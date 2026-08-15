"""Corrected Sol-style block-sparse attention for MiniMax H3 on MPS.

H3's self-attention is the term that makes long durations superlinear: every one
of the 50 DiT blocks attends over the whole packed [text | cond | audio | video]
sequence, which grows with clip length. Sol-Attn (arXiv:2607.24027) observes that
most of that score mass is concentrated in a small set of key blocks, routes to
them with a cheap block-centroid proxy, and — crucially — does not simply DROP
the rest: each rejected block contributes its centroid back to the softmax with a
`log(block_length)` multiplicity, so the rest of the sequence still reaches the
row as a coarse summary rather than being truncated away.

The routing policy is NVIDIA's released H3 one, adapted to Metal (`diag64`):

  * 64-token key blocks and 64-row query blocks;
  * one ANALYTIC threshold per query block, predicted from the diagonal
    covariance of that head's key centroids rather than from the proxy row's own
    spread — `mean + tau*sqrt(var + 1e-6)` in log2 units;
  * the whole `[0, video_start)` prefix (text + conditioning + audio) is one
    contiguous exact KV sink, rounded outward to a 64-token boundary, and every
    query block overlapping it is fully dense;
  * a local +-1 block band;
  * the first two transformer blocks stay dense;
  * no Morton reordering and no temporal-band heuristic.

The earlier empirical router — 32-row tiles thresholded on the proxy row's own
mean and variance — was rejected: it reached 13% density by discarding
information the model needed, and its visible symptom was static-background
morphing. It survives only for diagnostics, behind ASFP8_SOL_ROUTER=legacy_empirical,
and is never reachable from a preset or the node UI.

Two decisions make this practical here:

  * **Execution is a Metal kernel, not FlexAttention.** Measured on M5 Max,
    torch 2.13's MPS FlexAttention backend at H3 shapes (H=56, D=128, bf16, 26%
    block density) runs 5-9x SLOWER than dense SDPA, so no amount of routing
    tuning could pay for it. `sol_ext` — a Metal 4.1 TensorOps kernel that walks
    only the selected KV blocks — runs 2.2-4.6x faster than the flash41 dense
    kernel this package already ships, at every H3 shape from 1s to 10s.

  * **Routing and the correction run on the GPU, never in PyTorch.** The first
    implementation kept both in PyTorch, merging the correction afterwards from
    an exported log-sum-exp. That is algebraically clean but was measured at
    25.5 ms routing + 93.6 ms correction against an 85.8 ms kernel and 220 ms
    dense attention — the scaffolding cost more than the kernel saved, at every
    density. `diag64` splits it into `sol_diag_route` (one compact block mask per
    64-row query block, computed once per attention call) and
    `sol_masked_corrected_fwd` (the exact blocks plus the rejected blocks'
    centroids, in one online softmax). No `.item()` anywhere on the hot path.

Everything is opt-in through the experimental node; unsupported calls delegate to
the package's own dense router (so flash41 still serves them) with a once-only
reason. Kill switch for the kernel itself: ASFP8_SOL_EXT=0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
from typing import Any

import torch


TAG = "[AppleSilicon-Optimized/H3Sparse]"
STATE_KEY = "asfp8_h3_sparse"
WRAPPER_KEY = "asfp8_h3_sparse"

MODES = ("corrected", "hard_drop", "dense_reference", "oracle")
DIAGNOSTIC_MODES = ("hard_drop", "dense_reference", "oracle")

# oracle: which transformer blocks to check the production route against the
# Python reference on. Early, middle and last — route agreement is a property of
# the arithmetic, not the depth, so three depths are enough to catch a
# systematic disagreement while staying affordable inside a real render.
ORACLE_BLOCKS = (2, 25, 49)

# Fixed, non-user-facing routing policy.
BLOCK_SIZE = 64
LOCAL_RADIUS = 1
H3_BLOCK_COUNT = 50

# NVIDIA's thresholds are derived in log2 units (their proxy scores carry the
# exp2 of a base-2 softmax), so both the mean and the standard deviation are
# scaled by log2(e). Dropping this would not change the RANKING of blocks, but
# it does change what `tau` means: one estimated standard deviation is 1.4426x
# wider in log2 units, so the same tau selects a different set.
LOG2E = 1.4426950408889634
# Guards sqrt() against a degenerate all-identical key-centroid set, where the
# analytic variance is exactly zero and every block would tie with the mean.
VARIANCE_EPS = 1e-6

ROUTER_DIAG64 = "diag64"
ROUTER_LEGACY = "legacy_empirical"


def router_name() -> str:
    """Which routing policy this process uses. `diag64` unless overridden.

    The legacy empirical router is reachable ONLY through the environment, for
    one diagnostic cycle; no preset and no node widget selects it.
    """
    name = os.environ.get("ASFP8_SOL_ROUTER", ROUTER_DIAG64)
    return name if name in (ROUTER_DIAG64, ROUTER_LEGACY) else ROUTER_DIAG64
# Co-located blocks in this many latent frames each way are always exact.
# DEFAULT OFF: measured on a synthetic static shot, forcing the temporal band
# made frame-to-frame consistency WORSE (1.22x dense at taps=0 -> 1.46x at
# taps=1 -> 1.64x at taps=2) while costing 5-10% density. The spatial band is
# translation-invariant -- every tile forces the same relative offsets, so every
# frame is treated alike -- whereas the temporal band is position-dependent and
# has edge effects at the first and last frames, which is itself a source of
# frame-to-frame variation. Exposed so it can be re-checked on real activations.
TEMPORAL_TAPS = 0

# Correction score tiles are [H, chunk, n_blocks] fp32; cap their footprint.
_CORRECTION_BUDGET_BYTES = 192 << 20
_MIN_CHUNK_ROWS = 128


def _as_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().flatten()[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _kernel():
    from .sol_ext import loader
    return loader.module()


def block_means(x: torch.Tensor, block: int, valid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block means of [H, S, D] over the real tokens, plus the block lengths.

    The final block is ragged: its mean divides by the number of REAL tokens it
    holds, not by `block`. Getting that wrong biases both the routing proxy and
    the centroid correction of the one block that covers the end of the video.
    """
    heads, seq, dim = x.shape
    n_blocks = (valid + block - 1) // block
    padded = n_blocks * block
    lengths = torch.full((n_blocks,), block, dtype=torch.float32, device=x.device)
    tail = valid - (n_blocks - 1) * block
    lengths[-1] = float(tail)
    if padded != seq or valid != seq:
        buf = x.new_zeros((heads, padded, dim))
        buf[:, :valid] = x[:, :valid]
    else:
        buf = x
    sums = buf.reshape(heads, n_blocks, block, dim).sum(dim=2)
    return sums / lengths.view(1, -1, 1).to(sums.dtype), lengths


def key_centroid_stats(k_cent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head mean and POPULATION diagonal variance of the key centroids.

    `k_cent` is [H, nK, D]; both outputs are fp32 [H, D]. This is the whole of
    the analytic threshold's model of the key distribution: the proxy score of a
    random key block is treated as a sum of D independent terms, so its variance
    is the query centroid's squared coordinates weighted by the per-coordinate
    variance of the key centroids. Computed once per attention call — it does
    not depend on the query at all.
    """
    kc = k_cent.float()
    return kc.mean(dim=1), kc.var(dim=1, unbiased=False)


def route_blocks_diag_reference(
    q_cent: torch.Tensor,
    k_cent: torch.Tensor,
    scale: float,
    tau: float,
    sink_blocks: int,
    local_radius: int = LOCAL_RADIUS,
) -> torch.Tensor:
    """NVIDIA's released H3 block router. Returns bool [H, nQ, nK].

    This is the executable reference the Metal path is tested against, and the
    definition of the `diag64` policy:

      * proxy(i, j) = <q_cent[i], k_cent[j]> * scale * log2(e);
      * threshold(i) = mu(i) + tau * sqrt(sigma2(i) + 1e-6), where
        mu(i)     = <q_cent[i], key_mean> * scale * log2(e) and
        sigma2(i) = <q_cent[i]^2, key_var> * (scale * log2(e))^2;
      * selection is STRICT `proxy > threshold`;
      * every |i - j| <= local_radius pair is forced;
      * `[0, sink_blocks)` is a contiguous exact sink — every query attends to
        those key blocks, and every query block inside it is fully dense.

    The threshold is ANALYTIC: it is predicted from the key centroids' own
    diagonal covariance, not measured from the proxy row. That is the difference
    from the rejected empirical router, which took the mean and variance OF THE
    PROXY ROW ITSELF and so adapted its threshold to whatever spread each tile
    happened to have — tightening exactly where the row was flat and the
    approximation was least safe.

    Both centroid tensors must already be pooled with the real (ragged-aware)
    block lengths; see `block_means`.
    """
    qc = q_cent.float()
    kc = k_cent.float()
    s = float(scale) * LOG2E
    k_mean, k_var = key_centroid_stats(kc)

    proxy = s * (qc @ kc.transpose(-1, -2))                       # [H, nQ, nK]
    mu = s * (qc * k_mean.unsqueeze(1)).sum(-1)                   # [H, nQ]
    sigma2 = (qc.square() * k_var.unsqueeze(1)).sum(-1).clamp_min(0) * (s * s)
    threshold = mu + float(tau) * (sigma2 + VARIANCE_EPS).sqrt()
    selected = proxy > threshold.unsqueeze(-1)

    n_q, n_k = proxy.shape[-2], proxy.shape[-1]
    qi = torch.arange(n_q, device=proxy.device).view(-1, 1)
    kj = torch.arange(n_k, device=proxy.device).view(1, -1)
    selected |= ((qi - kj).abs() <= local_radius).unsqueeze(0)

    sink = max(0, min(int(sink_blocks), n_k))
    if sink > 0:
        # H3 packs [text | cond | audio | video]. The whole prefix is one exact
        # sink: it is a few percent of the sequence, sits outside every video
        # query's local band, and scores below the video-video proxy mean, so a
        # purely score-driven router would collapse entire conditioning streams
        # into one averaged vector.
        selected[:, :, :sink] = True
        selected[:, :min(sink, n_q), :] = True
    return selected


def sink_block_count(video_row_start: int, block: int) -> int:
    """Blocks covering `[0, video_row_start)`, rounded OUTWARD to the boundary.

    Rounding outward (rather than truncating) keeps the sink a superset of the
    real prefix: the block straddling the text/video boundary holds real prefix
    tokens, so treating it as approximate would drop conditioning rows that no
    other rule protects.
    """
    if video_row_start <= 0:
        return 0
    return (int(video_row_start) + block - 1) // block


def selection_to_lists(selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bool [H, nQ, nK] -> (int32 indices [H, nQ, MMAX], int32 counts [H, nQ]).

    Unselected entries are sorted to the end through a sentinel ABOVE every real
    index. Encoding selection as `selected * index` would alias block 0 with
    every rejected block, which is exactly the indexing bug in the CUDA
    implementation this routing idea comes from.
    """
    n_k = selected.shape[-1]
    counts = selected.sum(-1, dtype=torch.int32)
    order = torch.arange(n_k, device=selected.device).view(1, 1, -1).expand_as(selected)
    sentinel = torch.full_like(order, n_k)
    indices = torch.where(selected, order, sentinel).sort(dim=-1).values
    max_selected = max(1, int(counts.max().item()))
    indices = indices[..., :max_selected].clamp_max(n_k - 1)
    return indices.to(torch.int32).contiguous(), counts.contiguous()


def _correction_chunk(heads: int, n_blocks: int) -> int:
    per_row = heads * n_blocks * 4
    if per_row <= 0:
        return _MIN_CHUNK_ROWS
    return max(_MIN_CHUNK_ROWS, _CORRECTION_BUDGET_BYTES // per_row)


def merge_centroid_correction(
    query: torch.Tensor,
    out_exact: torch.Tensor,
    lse: torch.Tensor,
    k_mean: torch.Tensor,
    v_mean: torch.Tensor,
    lengths: torch.Tensor,
    selected: torch.Tensor,
    block: int,
    scale: float,
) -> torch.Tensor:
    """Fold every REJECTED block's centroid back into the exact attention.

    query/out_exact: [H, S, D]; lse: [H, S]; k_mean/v_mean: [H, nK, D];
    selected: [H, nQ, nK]. Returns [H, S, D] in query's dtype.
    """
    heads, seq, dim = query.shape
    n_blocks = k_mean.shape[1]
    dropped = ~selected
    log_len = lengths.log().view(1, 1, -1)
    k_mean_f = k_mean.float().transpose(-1, -2)
    v_mean_f = v_mean.float()
    result = torch.empty_like(out_exact)
    step = _correction_chunk(heads, n_blocks)
    for start in range(0, seq, step):
        stop = min(start + step, seq)
        q_chunk = query[:, start:stop].float()
        scores = scale * (q_chunk @ k_mean_f) + log_len          # [H, c, nK]
        rows = torch.arange(start, stop, device=query.device) // block
        scores = scores.masked_fill(~dropped[:, rows, :], float("-inf"))
        row_lse = lse[:, start:stop]
        peak = torch.maximum(row_lse, scores.max(dim=-1).values)
        peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
        exact_weight = torch.exp(row_lse - peak)
        drop_weight = torch.exp(scores - peak.unsqueeze(-1))
        numerator = (exact_weight.unsqueeze(-1) * out_exact[:, start:stop].float()
                     + drop_weight @ v_mean_f)
        denominator = (exact_weight + drop_weight.sum(-1)).clamp_min(1e-30)
        result[:, start:stop] = (numerator / denominator.unsqueeze(-1)).to(result.dtype)
    return result


def sparse_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected: torch.Tensor,
    scale: float,
    block: int,
    correct: bool = True,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """fp32 pure-PyTorch definition of the corrected sparse attention.

    This is the specification the Metal path is tested against: dense scores
    masked to the selected blocks, plus one centroid term per rejected block
    carrying `log(valid_block_length)`. [H, S, D] in, [H, S, D] out.

    `rows` restricts the computation to a subset of query rows (returning
    [H, len(rows), D]). The full form materializes an [H, S, S] score matrix,
    which is several GB at H3 sequence lengths; the oracle uses a row sample.
    """
    heads, seq, dim = query.shape
    q_f, k_f, v_f = query.float(), key.float(), value.float()
    k_mean, lengths = block_means(k_f, block, seq)
    v_mean, _ = block_means(v_f, block, seq)
    n_blocks = k_mean.shape[1]

    if rows is None:
        rows = torch.arange(seq, device=query.device)
    q_f = q_f[:, rows]
    scores = scale * (q_f @ k_f.transpose(-1, -2))               # [H, R, S]
    block_rows = rows // block
    token_mask = selected[:, block_rows, :].repeat_interleave(block, dim=-1)[:, :, :seq]
    scores = scores.masked_fill(~token_mask, float("-inf"))

    if correct:
        centroid = scale * (q_f @ k_mean.transpose(-1, -2)) + lengths.log().view(1, 1, -1)
        centroid = centroid.masked_fill(selected[:, block_rows, :], float("-inf"))
        scores = torch.cat([scores, centroid], dim=-1)
        v_f = torch.cat([v_f, v_mean], dim=-2)

    peak = scores.max(dim=-1, keepdim=True).values
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    weights = torch.exp(scores - peak)
    return ((weights @ v_f) / weights.sum(-1, keepdim=True).clamp_min(1e-30)).to(query.dtype)


def route_tiles(
    query_padded: torch.Tensor,
    k_mean: torch.Tensor,
    scale: float,
    tau: float,
    block: int,
    tile: int,
    prefix_rows: int,
    audio_rows: tuple[int, int],
    prefix_blocks: int,
    audio_blocks: tuple[int, int],
    local_radius: int = LOCAL_RADIUS,
    video_row_start: int = 0,
    rows_per_frame: int = 0,
    taps: int = 0,
    valid_rows: int | None = None,
) -> torch.Tensor:
    """One keep/drop decision per (query tile, key block) — the kernel's rule.

    The matmul works on a `tile`-row block, so a key block must be computed if
    any row in the tile wants it. Thresholding per row makes that union 91-100%
    dense in practice, which erases the sparsity; deciding from the tile's mean
    query keeps density where the two-stage path had it. Returns [H, nT, nK].
    """
    heads, padded, dim = query_padded.shape
    n_tiles = padded // tile
    q_tile = query_padded.float().reshape(heads, n_tiles, tile, dim).mean(dim=2)
    proxy = scale * (q_tile @ k_mean.float().transpose(-1, -2))    # [H, nT, nK]
    mean = proxy.mean(dim=-1, keepdim=True)
    var = proxy.var(dim=-1, unbiased=False, keepdim=True)
    selected = proxy > (mean + tau * var.clamp_min(0).sqrt())

    n_k = k_mean.shape[1]
    device = query_padded.device
    starts = torch.arange(n_tiles, device=device) * tile
    qb0 = (starts // block).view(-1, 1)
    qb1 = ((starts + tile - 1) // block).view(-1, 1)
    k_index = torch.arange(n_k, device=device).view(1, -1)
    band = (k_index >= qb0 - local_radius) & (k_index <= qb1 + local_radius)
    selected |= band.unsqueeze(0)
    for start, stop in ((0, prefix_blocks), audio_blocks):
        if stop > start:
            selected[:, :, start:stop] = True
    tile_dense = (starts < prefix_rows) | ((starts + tile > audio_rows[0])
                                           & (starts < audio_rows[1]))
    selected |= tile_dense.view(1, -1, 1)

    # Temporal band: the co-located block one/two latent frames away. The
    # spatial band above is local WITHIN a frame (a frame is rows_per_frame
    # rows, ~25 blocks), so without this nothing guarantees a static region any
    # reference in another frame — the visible symptom is background flicker.
    if rows_per_frame > 0 and taps > 0:
        limit = padded if valid_rows is None else valid_rows
        centers = starts + tile // 2
        is_video = centers >= video_row_start
        for m in range(1, taps + 1):
            for sign in (1, -1):
                target = centers + sign * m * rows_per_frame
                ok = is_video & (target >= video_row_start) & (target < limit)
                tb = (target // block).view(-1, 1)
                band_t = ((k_index >= tb - local_radius)
                          & (k_index <= tb + local_radius) & ok.view(-1, 1))
                selected |= band_t.unsqueeze(0)
    return selected


def sparse_attention_fused_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    tau: float,
    block: int,
    tile: int = 32,
    prefix_rows: int = 0,
    audio_rows: tuple[int, int] = (0, 0),
    prefix_blocks: int = 0,
    audio_blocks: tuple[int, int] = (0, 0),
    local_radius: int = LOCAL_RADIUS,
    video_row_start: int = 0,
    rows_per_frame: int = 0,
    taps: int = 0,
) -> torch.Tensor:
    """fp32 specification of the fused (Phase E) kernel.

    Tile-granular selection, per-ROW centroid correction. This is what the
    Metal kernel is parity-tested against. Inputs are the unpadded [H, S, D]
    tensors; padding is applied internally exactly as the kernel sees it.
    """
    heads, seq, dim = query.shape
    k_mean, lengths = block_means(key.float(), block, seq)
    v_mean, _ = block_means(value.float(), block, seq)
    n_blocks = k_mean.shape[1]
    padded = n_blocks * block
    pad = padded - seq
    q_pad = torch.nn.functional.pad(query, (0, 0, 0, pad))

    selected = route_tiles(q_pad, k_mean, scale, tau, block, tile, prefix_rows,
                           audio_rows, prefix_blocks, audio_blocks, local_radius,
                           video_row_start, rows_per_frame, taps, seq)
    rows = torch.arange(seq, device=query.device)
    sel_rows = selected[:, rows // tile, :]                       # [H, S, nK]

    q_f, k_f, v_f = query.float(), key.float(), value.float()
    scores = scale * (q_f @ k_f.transpose(-1, -2))
    token_mask = sel_rows.repeat_interleave(block, dim=-1)[:, :, :seq]
    scores = scores.masked_fill(~token_mask, float("-inf"))

    centroid = scale * (q_f @ k_mean.transpose(-1, -2)) + lengths.log().view(1, 1, -1)
    centroid = centroid.masked_fill(sel_rows, float("-inf"))
    joint = torch.cat([scores, centroid], dim=-1)
    values = torch.cat([v_f, v_mean], dim=-2)

    peak = joint.max(dim=-1, keepdim=True).values
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    weights = torch.exp(joint - peak)
    return ((weights @ values) / weights.sum(-1, keepdim=True).clamp_min(1e-30)).to(query.dtype)


@dataclass
class _RunStats:
    calls: int = 0
    sparse_calls: int = 0
    dense_calls: int = 0
    routed_calls: int = 0      # calls the router scored (includes dense_reference)
    evaluations: int = 0
    sparse_evaluations: int = 0
    dense_evaluations: int = 0
    reasons: set[str] = field(default_factory=set)
    # Selected-block totals stay on the DEVICE. Reading a density per attention
    # call cost one MPS synchronization per call — 50 blocks x 20 steps of
    # stalling the whole pipeline to print a number nobody reads until the run
    # ends. These accumulate asynchronously and are read exactly once.
    selected_blocks: torch.Tensor | None = None
    total_blocks: int = 0

    def add_selected(self, selected: torch.Tensor, total: int) -> None:
        """Accumulate a device-side count of selected blocks. Never syncs."""
        value = selected.detach().sum().to(torch.float32).reshape(())
        self.selected_blocks = (value if self.selected_blocks is None
                                else self.selected_blocks + value)
        self.total_blocks += int(total)

    def mean_density(self) -> float:
        """The run's effective density. THE one synchronization, at run end."""
        if self.selected_blocks is None or self.total_blocks <= 0:
            return 0.0
        return float(self.selected_blocks.item()) / float(self.total_blocks)


class H3SparseHolder:
    """Per-sampling-run configuration and evaluation state."""

    def __init__(
        self,
        mode: str = "corrected",
        tau: float = 1.0,
        start_percent: float = 0.50,
        end_percent: float = 0.85,
        dense_leading_blocks: int = 2,
        min_sequence_tokens: int = 8192,
        block_size: int = BLOCK_SIZE,
        verbose: bool = True,
        temporal_taps: int = TEMPORAL_TAPS,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if tau < 0:
            raise ValueError("tau must be non-negative")
        if not 0.0 <= start_percent <= end_percent <= 1.0:
            raise ValueError("percents must satisfy 0 <= start <= end <= 1")
        if dense_leading_blocks < 0:
            raise ValueError("dense_leading_blocks must be non-negative")
        if min_sequence_tokens < 0:
            raise ValueError("min_sequence_tokens must be non-negative")
        if block_size not in (32, 64, 128):
            raise ValueError("block_size must be 32, 64 or 128")
        if not 0 <= temporal_taps <= 2:
            raise ValueError("temporal_taps must be 0, 1 or 2")
        self.router = router_name()
        if self.router == ROUTER_DIAG64:
            # The two widgets below exist only because ComfyUI maps saved
            # workflow values positionally, so removing them would silently
            # shift every later value into the wrong slot. Neither is part of
            # the released policy, and both are refused rather than ignored.
            if block_size != BLOCK_SIZE:
                raise ValueError(
                    f"block_size must be {BLOCK_SIZE} for the diag64 router "
                    "(NVIDIA's released H3 policy pools 64-token blocks; the "
                    "analytic threshold is calibrated to that pooling). Set "
                    f"ASFP8_SOL_ROUTER={ROUTER_LEGACY} to use the rejected "
                    "empirical router, which accepted other block sizes.")
            if temporal_taps != 0:
                raise ValueError(
                    "temporal_taps must be 0 for the diag64 router: the "
                    "released policy has no temporal band, and forcing one "
                    "measured WORSE frame-to-frame consistency here, not better.")
        self.mode = mode
        self.tau = float(tau)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.dense_leading_blocks = int(dense_leading_blocks)
        self.min_sequence_tokens = int(min_sequence_tokens)
        self.block_size = int(block_size)
        self.verbose = bool(verbose)
        self.temporal_taps = int(temporal_taps)
        self.start_sigma: float | None = None
        self.end_sigma: float | None = None
        # per-evaluation
        self.block_index = 0
        self.prefix_rows: int | None = None
        self.target_rows: int | None = None
        self.audio_rows: int | None = None
        self.rows_per_frame: int | None = None
        self.sigma: float | None = None
        self.oracle: dict[float, list[tuple[float, float, float]]] = {}
        self.oracle_seen = 0
        self.disabled = False
        self.stats = _RunStats()

    def clone(self) -> "H3SparseHolder":
        return H3SparseHolder(
            self.mode, self.tau, self.start_percent, self.end_percent,
            self.dense_leading_blocks, self.min_sequence_tokens,
            self.block_size, self.verbose, self.temporal_taps,
        )

    def prepare_timesteps(self, model_sampling: Any) -> "H3SparseHolder":
        self.start_sigma = float(model_sampling.percent_to_sigma(self.start_percent))
        self.end_sigma = float(model_sampling.percent_to_sigma(self.end_percent))
        return self

    def begin_evaluation(self, video: torch.Tensor | None, audio: torch.Tensor | None,
                         sigma: float | None) -> None:
        """Reset per-model-evaluation state and record the target row counts.

        H3 packs [text | cond | audio | video] and the two target streams are
        always the last segments, so the prefix length any attention call sees is
        `S - (audio_rows + video_rows)`. Deriving it from the sampler-facing
        latents avoids reaching into the model's private layout.
        """
        self.block_index = 0
        self.sigma = sigma
        self.target_rows = None
        self.audio_rows = None
        self.rows_per_frame = None
        self.prefix_rows = None
        if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
            return
        if video.dim() != 5 or audio.dim() < 3:
            return
        patch_h = (video.shape[-2] + 1) // 2
        patch_w = (video.shape[-1] + 1) // 2
        video_rows = video.shape[0] * video.shape[-3] * patch_h * patch_w
        audio_rows = audio.shape[-2] * audio.shape[-1]
        if video_rows > 0 and audio_rows > 0:
            self.target_rows = int(video_rows + audio_rows)
            self.audio_rows = int(audio_rows)
            # one latent frame's worth of packed rows — the stride between a
            # token and its co-located token in the next frame
            self.rows_per_frame = int(patch_h * patch_w)
        self.stats.evaluations += 1
        if self.in_sparse_window():
            self.stats.sparse_evaluations += 1
        else:
            self.stats.dense_evaluations += 1

    def note(self, reason: str) -> None:
        if reason not in self.stats.reasons:
            self.stats.reasons.add(reason)
            if self.verbose:
                logging.info("%s dense fallback: %s", TAG, reason)

    def in_sparse_window(self) -> bool:
        """Both ends are protected.

        Early steps are where the model is hypersensitive (H3's measured video
        transform rate is 41x early against 0.33 late). Late steps are where
        frame-to-frame texture is settled, so approximating there shows up as
        flicker on static content.
        """
        if self.start_sigma is None or self.end_sigma is None or self.sigma is None:
            return False
        upper = max(self.start_sigma, self.end_sigma)
        lower = min(self.start_sigma, self.end_sigma)
        return lower <= self.sigma <= upper

    def summary(self) -> str:
        stats = self.stats
        # The ONE synchronization of the whole run.
        density = stats.mean_density()
        # `routed` is the honest "did the router engage" counter: in
        # dense_reference mode every routed call still returns dense, so
        # sparse=0 there is expected rather than a silent no-op.
        return (f"router={self.router} mode={self.mode} tau={self.tau:.3g} "
                f"evaluations={stats.evaluations} "
                f"(sparse={stats.sparse_evaluations} dense={stats.dense_evaluations}) "
                f"attention_calls={stats.calls} routed={stats.routed_calls} "
                f"sparse={stats.sparse_calls} dense={stats.dense_calls} "
                f"effective_density={density:.1%}")

    def oracle_report(self) -> list[str]:
        """Route-mask parity between the Metal kernel and the Python reference."""
        return oracle_block_report(self)


def _dense(comfy_fn, q, k, v, heads, kwargs):
    from . import attn_router_mps
    return attn_router_mps.dense_without_override(comfy_fn, q, k, v, heads, **kwargs)


def _eligible(holder: H3SparseHolder, q, k, v, kwargs) -> str | None:
    """Return a fallback reason, or None when the call can run sparse.

    Structural reasons are checked before the device capability gate so the
    reported reason is the most specific one, and so the whole policy is
    testable without an MPS device.
    """
    if holder.disabled:
        return "latched_off"
    if kwargs.get("mask") is not None:
        return "masked_attention"
    if not kwargs.get("skip_reshape", False):
        return "unexpected_layout"
    if not isinstance(q, torch.Tensor) or q.dim() != 4:
        return "unexpected_rank"
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        return "unsupported_dtype"
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return "batched"
    if k.shape != q.shape or v.shape != q.shape:
        return "cross_attention"
    if q.requires_grad or (torch.is_grad_enabled() and q.grad_fn is not None):
        return "training_tensor"
    dim = int(q.shape[-1])
    if dim > 128 or dim % 8:
        return "unsupported_head_dim"
    seq = int(q.shape[2])
    if holder.target_rows is None:
        return "unknown_layout"
    if seq <= holder.target_rows:
        return "not_main_sequence"
    if seq < holder.min_sequence_tokens:
        return "short_sequence"
    if not holder.in_sparse_window():
        return "protected_steps"
    if q.device.type != "mps":
        return "not_mps"
    return None


def sparse_attention_override(comfy_fn, q, k, v, heads, **kwargs):
    """`optimized_attention_override` entry point for one H3 attention call."""
    transformer_options = kwargs.get("transformer_options")
    holder = (transformer_options.get(STATE_KEY)
              if isinstance(transformer_options, dict) else None)
    if not isinstance(holder, H3SparseHolder):
        return _dense(comfy_fn, q, k, v, heads, kwargs)

    holder.stats.calls += 1
    reason = _eligible(holder, q, k, v, kwargs)
    if reason == "not_main_sequence":
        # Token-refiner attention: shorter than the packed target streams. Not a
        # DiT block, so it must not advance the block counter — otherwise a run
        # that refines text would shift every block index by two.
        holder.stats.dense_calls += 1
        return _dense(comfy_fn, q, k, v, heads, kwargs)

    # Anything that got this far is one of H3's 50 DiT block attentions, so the
    # counter advances even when this particular call ends up dense.
    block_index = holder.block_index
    holder.block_index += 1
    if reason is None:
        if block_index < holder.dense_leading_blocks:
            reason = "leading_block"
        elif block_index >= H3_BLOCK_COUNT:
            reason = "unexpected_block_index"

    if reason is not None:
        holder.note(reason)
        holder.stats.dense_calls += 1
        return _dense(comfy_fn, q, k, v, heads, kwargs)

    try:
        out, selected, total = _run_sparse(holder, q, k, v, kwargs.get("scale"))
    except _SparseUnavailable as exc:
        # Capability problems latch the route off for the whole run rather than
        # paying a failed setup on all 50 blocks of every remaining step.
        holder.disabled = True
        holder.note(f"kernel_unavailable ({exc})")
        holder.stats.dense_calls += 1
        return _dense(comfy_fn, q, k, v, heads, kwargs)

    holder.stats.add_selected(selected, total)
    holder.stats.routed_calls += 1
    if holder.mode in ("dense_reference", "oracle"):
        # Diagnostics: route (and, for oracle, score) but return exact output,
        # so the render itself doubles as the matched reference.
        holder.stats.dense_calls += 1
        return _dense(comfy_fn, q, k, v, heads, kwargs)
    holder.stats.sparse_calls += 1
    if not kwargs.get("skip_output_reshape", False):
        heads_n, seq, dim = out.shape
        return out.transpose(0, 1).reshape(1, seq, heads_n * dim)
    return out.unsqueeze(0)


def _sparse_attention_containers(q, k, v, heads, **kwargs):
    """PR #15479+ single-owner entry point for the sparse override."""
    from . import attn_router_mps

    q, k, v, consumed = attn_router_mps._take_attention_inputs(q, k, v)
    if not consumed:
        raise TypeError("container_function requires q, k, and v containers")
    try:
        # There is no underlying `func` argument in ComfyUI's container
        # override contract. dense_without_override can still reach either the
        # selected dense backend, the installed router, or ComfyUI's dispatcher.
        return sparse_attention_override(None, q, k, v, heads, **kwargs)
    finally:
        q = k = v = None


sparse_attention_override.container_function = _sparse_attention_containers
sparse_attention_override._asfp8_sparse_override = True


class _SparseUnavailable(RuntimeError):
    pass


def _pad_to_blocks(tensors, pad):
    if not pad:
        return list(tensors)
    return [torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in tensors]


def _video_row_start(holder: "H3SparseHolder", seq: int) -> int:
    """First row of the video stream — the end of the contiguous exact sink.

    H3 packs [text | cond | audio | video] with the two target streams last, so
    the prefix is `seq - target_rows` and the audio sits immediately after it.
    Everything before the video is one sink; there is no separate audio span.
    """
    prefix_rows = max(0, seq - int(holder.target_rows or 0))
    return prefix_rows + int(holder.audio_rows or 0)


def _run_diag(holder, kernel, query, key, value, scale, block,
              k_mean, v_mean, lengths, sink_blocks):
    """diag64: route on the GPU, then attend under that route.

    Returns ([H, S, D] output, selected counts [Hq, nQ] on device, total block
    slots). Nothing here synchronizes: the counts stay a device tensor and are
    read once, at the end of the sampling run.
    """
    heads, seq, dim = query.shape
    n_blocks = k_mean.shape[1]
    pad = n_blocks * block - seq
    query_p, key_p, value_p = _pad_to_blocks((query, key, value), pad)

    # fp32 centroids for routing: the threshold's variance term is a sum of D
    # squared products, which bf16 cannot hold to the precision the strict
    # `proxy > threshold` comparison needs. The ATTENTION path keeps the bf16
    # centroids — there they are just extra softmax columns.
    k_cent = k_mean.float().contiguous()
    cent_mean, cent_var = key_centroid_stats(k_cent)
    try:
        mask, counts = kernel.sol_diag_route(
            query_p.unsqueeze(0), k_cent, cent_mean.contiguous(),
            cent_var.contiguous(), scale, holder.tau, seq, block,
            sink_blocks, LOCAL_RADIUS)
    except RuntimeError as exc:
        raise _SparseUnavailable(str(exc)[:200]) from None

    total = counts.numel() * n_blocks
    if holder.mode in ("dense_reference", "oracle"):
        if holder.mode == "oracle":
            _score_oracle(holder, query, key, value, scale, block,
                          k_cent, sink_blocks, mask, counts)
        return query, counts, total

    try:
        out, _ = kernel.sol_masked_corrected_fwd(
            query_p.unsqueeze(0), key_p.unsqueeze(0), value_p.unsqueeze(0),
            k_mean.contiguous(), v_mean.contiguous(), lengths.log().contiguous(),
            mask, scale, seq, seq, block, 0, False,
            holder.mode != "hard_drop")
    except RuntimeError as exc:
        raise _SparseUnavailable(str(exc)[:200]) from None
    return out[0, :, :seq], counts, total


def _run_legacy(holder, kernel, query, key, value, scale, block,
                k_mean, v_mean, lengths, prefix_rows, audio_span):
    """The REJECTED empirical 32-row-tile router, for diagnostics only.

    Reachable through ASFP8_SOL_ROUTER=legacy_empirical and nothing else. Its
    threshold comes from the proxy row's own mean and variance, which is what
    let it reach 13% density by discarding structure the model needed; the
    visible symptom was static-background morphing.
    """
    heads, seq, dim = query.shape
    n_blocks = k_mean.shape[1]
    prefix_blocks = (prefix_rows + block - 1) // block
    audio_rows = int(holder.audio_rows or 0)
    a_r0, a_r1 = prefix_rows, prefix_rows + audio_rows
    a_b0, a_b1 = audio_span if audio_span else (0, 0)

    pad = n_blocks * block - seq
    query_p, key_p, value_p = _pad_to_blocks((query, key, value), pad)

    video_start = prefix_rows + audio_rows
    rows_per_frame = int(holder.rows_per_frame or 0) if holder.temporal_taps else 0
    try:
        out, stat = kernel.sol_fused_fwd(
            query_p.unsqueeze(0), key_p.unsqueeze(0), value_p.unsqueeze(0),
            k_mean.contiguous(), v_mean.contiguous(), lengths.log().contiguous(),
            scale, holder.tau, seq, seq, block, 0, False,
            prefix_rows, a_r0, a_r1, prefix_blocks, a_b0, a_b1, LOCAL_RADIUS,
            video_start, rows_per_frame, holder.temporal_taps)
    except RuntimeError as exc:
        raise _SparseUnavailable(str(exc)[:200]) from None
    return out[0, :, :seq], stat, stat.numel() * n_blocks


def _audio_block_span(holder: "H3SparseHolder", seq: int, block: int):
    """[start, stop) block range covering H3's packed audio segment, or None.

    The packed order is [text | cond | audio | video] with the two target
    streams last, so the audio rows begin exactly where the prefix ends.
    """
    if not holder.audio_rows or not holder.target_rows:
        return None
    prefix_rows = seq - int(holder.target_rows)
    if prefix_rows < 0:
        return None
    start = prefix_rows // block
    stop = (prefix_rows + int(holder.audio_rows) + block - 1) // block
    return (start, stop)


def mask_to_selection(mask: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """Unpack the kernel's compact bitmask [H, nQ, MW] to bool [H, nQ, nK]."""
    words = mask.to(torch.int64) & 0xFFFFFFFF
    bits = torch.arange(32, device=mask.device, dtype=torch.int64)
    unpacked = ((words.unsqueeze(-1) >> bits) & 1).to(torch.bool)
    return unpacked.reshape(mask.shape[0], mask.shape[1], -1)[..., :n_blocks]


def _score_oracle(holder, query, key, value, scale, block, k_cent,
                  sink_blocks, mask, counts):
    """Compare the PRODUCTION Metal route against the Python reference.

    Runs on real H3 activations and returns exact attention, so the render
    itself doubles as the matched dense reference. Sampled at three depths
    (early, middle, last transformer block) rather than every call: unpacking
    the mask and rebuilding the reference route is far too expensive to do 50
    times a step, and route agreement does not vary meaningfully call to call.
    """
    if holder.block_index not in ORACLE_BLOCKS:
        return
    holder.oracle_seen += 1
    seq = query.shape[1]
    n_blocks = k_cent.shape[1]
    q_cent, _ = block_means(query.float(), block, seq)
    want = route_blocks_diag_reference(q_cent, k_cent, scale, holder.tau,
                                       sink_blocks, LOCAL_RADIUS)
    got = mask_to_selection(mask, n_blocks)
    agree = float((got == want).float().mean().item())
    got_density = float(got.float().mean().item())
    want_density = float(want.float().mean().item())
    holder.oracle.setdefault(int(holder.block_index), []).append(
        (agree, got_density, want_density))


def oracle_block_report(holder) -> list[str]:
    """Route-mask parity, per sampled transformer block."""
    if not holder.oracle:
        return []
    lines = ["oracle: transformer_block -> metal/reference mask agreement "
             "and effective density (agreement must be >= 99.99%)"]
    for index in sorted(holder.oracle):
        samples = holder.oracle[index]
        count = len(samples)
        agree = sum(s[0] for s in samples) / count
        got = sum(s[1] for s in samples) / count
        want = sum(s[2] for s in samples) / count
        lines.append(
            f"oracle: block={index:<3d} agreement={agree:8.5%} "
            f"density metal={got:6.2%} reference={want:6.2%} "
            f"delta={100.0 * (got - want):+.3f}pp (n={count})")
    return lines


def _run_sparse(holder: H3SparseHolder, q, k, v, scale):
    """Route on the GPU, then attend under that route.

    Returns ([H, S, D] output, selected-block counts on device, total block
    slots). The counts are NEVER read here — the caller accumulates them and one
    synchronization at the end of the run turns them into a density.
    """
    kernel = _kernel()
    if kernel is None:
        raise _SparseUnavailable("sol_ext did not build")

    query, key, value = q[0], k[0], v[0]           # [H, S, D]
    heads, seq, dim = query.shape
    block = holder.block_size
    sc = float(scale) if scale is not None else dim ** -0.5

    k_mean, lengths = block_means(key, block, seq)
    v_mean, _ = block_means(value, block, seq)

    if holder.router == ROUTER_LEGACY:
        prefix_rows = max(0, seq - int(holder.target_rows or 0))
        return _run_legacy(holder, kernel, query, key, value, sc, block,
                           k_mean, v_mean, lengths, prefix_rows,
                           _audio_block_span(holder, seq, block))

    if not hasattr(kernel, "sol_diag_route"):
        raise _SparseUnavailable("sol_ext predates the diag64 router; rebuild it")
    sink_blocks = sink_block_count(_video_row_start(holder, seq), block)
    return _run_diag(holder, kernel, query, key, value, sc, block,
                     k_mean, v_mean, lengths, sink_blocks)


def h3_sparse_diffusion_wrapper(executor, *args, **kwargs):
    """Reset per-evaluation state and capture the target latent shapes."""
    transformer_options = kwargs.get("transformer_options")
    if not isinstance(transformer_options, dict):
        for candidate in reversed(args):
            if isinstance(candidate, dict) and STATE_KEY in candidate:
                transformer_options = candidate
                break
    holder = (transformer_options.get(STATE_KEY)
              if isinstance(transformer_options, dict) else None)
    if not isinstance(holder, H3SparseHolder):
        return executor(*args, **kwargs)
    pair = args[0] if args else None
    video = audio = None
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        video, audio = pair[0], pair[1]
    holder.begin_evaluation(video, audio, _as_float(transformer_options.get("sigmas")))
    return executor(*args, **kwargs)


def h3_sparse_sample_wrapper(executor, *args, **kwargs):
    """Own sparse state for one sample and release it on every exit path."""
    import comfy.model_patcher

    guider = executor.class_obj
    original_options = guider.model_options
    active: H3SparseHolder | None = None
    try:
        guider.model_options = comfy.model_patcher.create_model_options_clone(
            original_options)
        transformer_options = guider.model_options.setdefault("transformer_options", {})
        template = transformer_options.get(STATE_KEY)
        if not isinstance(template, H3SparseHolder):
            return executor(*args, **kwargs)
        active = template.clone().prepare_timesteps(
            guider.model_patcher.model.model_sampling)
        transformer_options[STATE_KEY] = active
        transformer_options["optimized_attention_override"] = sparse_attention_override
        logging.info(
            "%s enabled router=%s mode=%s tau=%.3g window=%.2f..%.2f "
            "dense_leading=%d min_seq=%d block=%d temporal_taps=%d",
            TAG, active.router, active.mode, active.tau, active.start_percent,
            active.end_percent, active.dense_leading_blocks,
            active.min_sequence_tokens, active.block_size, active.temporal_taps,
        )
        return executor(*args, **kwargs)
    finally:
        if active is not None:
            logging.info("%s %s", TAG, active.summary())
            for line in active.oracle_report():
                logging.info("%s %s", TAG, line)
            active.stats = _RunStats()
            active.oracle.clear()
        guider.model_options = original_options


def apply_h3_sparse(
    model: Any,
    mode: str,
    tau: float,
    start_percent: float,
    dense_leading_blocks: int,
    min_sequence_tokens: int,
    verbose: bool,
    block_size: int = BLOCK_SIZE,
    end_percent: float = 0.85,
    temporal_taps: int = TEMPORAL_TAPS,
):
    """Attach corrected sparse attention to a cloned MiniMax H3 ModelPatcher.

    Every rejection is raised before anything is imported, cloned or patched, so
    a refused combination leaves the incoming model exactly as it was.
    """
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = getattr(diffusion_model, "blocks", None)
    module_name = getattr(type(diffusion_model), "__module__", "")
    if blocks is None or "comfy.ldm.minimax" not in module_name:
        raise ValueError("H3 sparse attention only supports ComfyUI's MiniMax H3 model")
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        raise ValueError("H3 sparse attention requires an MPS (Apple Silicon) device")
    if _kernel() is None:
        raise ValueError(
            "H3 sparse attention requires the sol_ext Metal kernel, which did not "
            "build. Install the Metal toolchain (xcrun) and ninja into the ComfyUI "
            "venv, or leave this node out; ASFP8_SOL_EXT=0 also force-disables it.")

    current = model.model_options.get("transformer_options", {})
    if STATE_KEY in current:
        raise ValueError("H3 sparse attention is already attached to this model")
    existing = current.get("optimized_attention_override")
    composable_dense = bool(
        getattr(existing, "_asfp8_dense_backend", False))
    if (existing is not None and existing is not sparse_attention_override
            and not composable_dense):
        raise ValueError(
            "another attention override is already installed on this model; "
            "H3 sparse attention will not replace it")

    import comfy.patcher_extension

    patched = model.clone()
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options[STATE_KEY] = H3SparseHolder(
        mode=mode,
        tau=tau,
        start_percent=start_percent,
        end_percent=end_percent,
        dense_leading_blocks=dense_leading_blocks,
        min_sequence_tokens=min_sequence_tokens,
        block_size=block_size,
        verbose=verbose,
        temporal_taps=temporal_taps,
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        WRAPPER_KEY,
        h3_sparse_sample_wrapper,
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        h3_sparse_diffusion_wrapper,
    )
    return patched


class AppleSiliconH3SparseAttention:
    """Explicit ComfyUI node for corrected Sol-style sparse attention."""

    # Kept registered for saved workflows; the unified H3 node is the canonical
    # production-facing control surface.
    DEPRECATED = True
    CATEGORY = "Apple Silicon/approximate"
    DESCRIPTION = (
        "MiniMax H3 block-sparse self-attention on MPS, using NVIDIA's released "
        "H3 routing policy (diag64): 64-token blocks, an analytic threshold from "
        "the key centroids' diagonal covariance, a contiguous exact text/audio "
        "sink, a local +-1 band and two dense leading transformer blocks. A "
        "Metal 4.1 kernel computes only the selected blocks and folds every "
        "rejected block back through its centroid with log-length multiplicity. "
        "Approximate and opt-in; hard_drop, dense_reference and oracle are "
        "diagnostics, not production modes. Tuned for 20-step workflows."
    )
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (list(MODES), {"default": "corrected"}),
                "tau": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                }),
                # 0.50 protects the first TEN of twenty denoising evaluations.
                # H3's measured video transform rate is 41x early against 0.33
                # late, so the early steps are where an approximation shows.
                "start_percent": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "dense_leading_blocks": ("INT", {
                    "default": 2, "min": 0, "max": 49, "step": 1,
                }),
                "min_sequence_tokens": ("INT", {
                    "default": 8192, "min": 0, "max": 131072, "step": 512,
                }),
                "verbose": ("BOOLEAN", {"default": True}),
                # NEW WIDGETS GO AT THE END. ComfyUI maps saved workflow values
                # positionally, so inserting one mid-list silently shifts every
                # later value into the wrong slot (min_sequence_tokens=4096
                # landing in dense_leading_blocks, etc).
                "end_percent": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                # Present only for saved-workflow compatibility: the diag64
                # router REFUSES any value but the default for both of these,
                # rather than silently ignoring them. Removing the widgets would
                # shift every earlier value into the wrong slot on load.
                "temporal_taps": ("INT", {
                    "default": 0, "min": 0, "max": 2, "step": 1,
                }),
                "block_size": ([64, 32, 128], {"default": 64}),
            },
        }

    def patch(self, model, mode, tau, start_percent, dense_leading_blocks,
              min_sequence_tokens, verbose, end_percent=0.85, temporal_taps=0,
              block_size=BLOCK_SIZE):
        return (apply_h3_sparse(
            model, mode, tau, start_percent, dense_leading_blocks,
            min_sequence_tokens, verbose, end_percent=end_percent,
            temporal_taps=temporal_taps, block_size=int(block_size),
        ),)


NODE_CLASS_MAPPINGS = {
    "AppleSiliconH3SparseAttention": AppleSiliconH3SparseAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleSiliconH3SparseAttention": "H3 Sparse Attention (Apple Silicon, Experimental)",
}
