# Derived from https://github.com/veda-sparse/Veda-on-ComfyUI at fd59c7277ccc37ebf1a8f6823474b8c2ef2e33a0.
# Locally maintained Apple Silicon runtime; see h3_runtime/ORIGINS.json and package LICENSE.
"""Per-run geometry caches and sparse-work diagnostics for our Apple runtime.

The bounded execution loop and FP32 projection cache live in ../runtime.py.
No generic CUDA allocator heuristic or alternate attention loop is retained.
"""

from __future__ import annotations

import collections
import contextlib
import weakref

import torch

from . import bundle as veda_bundle
from . import h3_layout
from . import plans as veda_plans
from . import selection
from . import tiling

class Stats:
    """What the sparse calls computed, accumulated on the device and read
    once at the end of a run (reading per call would synchronize).

    Work is counted in 128x128 blocks per head: kept video blocks plus the
    always-dense global rows / columns, against the (S / 128)^2 blocks of
    full attention, so `compute_fraction` is the share of full-attention
    work actually done.
    """

    def __init__(self):
        self.kept = None
        self.video_pairs = 0
        self.work = None
        self.dense_work = 0.0

    def add(self, keep: torch.Tensor, layout: tiling.TileLayout) -> None:
        heads = keep.shape[0]
        kept = keep.sum()
        global_blocks = layout.n_global_tiles * (layout.n_tiles
                                                 + layout.n_video_tiles)
        work = kept + heads * global_blocks
        self.kept = kept if self.kept is None else self.kept + kept
        self.work = work if self.work is None else self.work + work
        self.video_pairs += heads * layout.n_video_tiles ** 2
        self.dense_work += heads * (layout.seq_len / tiling.TILE_SIZE) ** 2

    def kept_fraction(self) -> float | None:
        """Kept share of the video x video tile pairs."""
        if self.kept is None or not self.video_pairs:
            return None
        return float(self.kept) / self.video_pairs

    def compute_fraction(self) -> float | None:
        """Share of full-attention work done (all rows, all tokens)."""
        if self.work is None or not self.dense_work:
            return None
        return min(1.0, float(self.work) / self.dense_work)


class PhaseTimer:
    """Opt-in per-phase GPU time (CUDA events; read once at the end)."""

    def __init__(self):
        self._events = collections.defaultdict(list)

    @contextlib.contextmanager
    def __call__(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self._events[name].append((start, end))

    def summary(self) -> dict[str, float]:
        """Milliseconds per phase (synchronizes the device)."""
        torch.cuda.synchronize()
        return {name: sum(s.elapsed_time(e) for s, e in pairs)
                for name, pairs in self._events.items()}


def _no_timer(name: str):
    del name
    return contextlib.nullcontext()


class GeometryState:
    """Sparse attention for one predictor bundle and settings on a device.

    Args:
        bundle: The predictor bundle.
        generated: Budget of the generated (target) video's key tiles.
        reference: Budget of the reference / condition key tiles;
            keeps_all leaves the conditions global (full attention both
            ways), which is also what happens when the layout has none.
        backend: A `backends.base.Backend`.
        device: The device of q / k / v.
    """

    def __init__(self, bundle: veda_bundle.PredictorBundle,
                 generated: selection.Budget, reference: selection.Budget,
                 backend, device: torch.device):
        self.bundle = bundle
        self.generated = generated
        self.reference = reference
        self.backend = backend
        self.device = device
        self.stats = Stats()
        self.timer = _no_timer
        # Last call's chunking, for diagnostics: free memory, heads per
        # chunk, chunks per layer.
        self.chunking: dict[str, int] = {}
        self._specs = weakref.WeakKeyDictionary()
        self._tile_layouts: dict[tuple, tiling.TileLayout] = {}

    def enable_timing(self) -> PhaseTimer | None:
        """Times every phase from now on; returns the timer, or None off
        CUDA (no event timers there)."""
        if self.device.type != 'cuda':
            return None
        self.timer = PhaseTimer()
        return self.timer

    # -- caches ---------------------------------------------------------

    def layout_spec(self, layout) -> h3_layout.LayoutSpec:
        """LayoutSpec of a ComfyUI PackedLayout (cached per layout object).

        Raises:
            h3_layout.LayoutError: If the layout cannot be mapped.
        """
        try:
            spec = self._specs.get(layout)
        except TypeError:  # not weak-referenceable: describe every time
            return h3_layout.describe(layout)
        if spec is None:
            spec = h3_layout.describe(layout)
            self._specs[layout] = spec
        return spec

    def plan_for(self, spec: h3_layout.LayoutSpec) -> veda_plans.PlanChoice:
        return self.bundle.plans.select(spec.target.grid)

    def _tile_layout(self, spec: h3_layout.LayoutSpec,
                     shape: tiling.TileShape) -> tiling.TileLayout:
        tile_references = not self.reference.keeps_all
        key = (spec, shape, tile_references)
        layout = self._tile_layouts.get(key)
        if layout is None:
            spans = []
            if tile_references:
                spans = [tiling.TiledSpan(s.start, s.grid,
                                          tiling.least_padding_shape(s.grid))
                         for s in spec.references]
            spans.append(tiling.TiledSpan(spec.target.start, spec.target.grid,
                                          shape))
            layout = tiling.build_tile_layout(spans, spec.seq_len,
                                              self.device)
            self._tile_layouts[key] = layout
        return layout

    def reset(self) -> None:
        """Drops per-geometry caches (between sampling runs)."""
        self._tile_layouts.clear()
        self._specs = weakref.WeakKeyDictionary()
        self.stats = Stats()

