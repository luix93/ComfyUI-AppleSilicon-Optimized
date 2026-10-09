"""Hooks Veda into ComfyUI's MiniMax-H3 attention.

H3's `Attention.forward` (comfy/ldm/minimax/model.py) calls
`optimized_attention` with q / k / v already through the (LoRA-patched)
qkv projection, QK-norm and RoPE, and with `transformer_options` carrying
`block_index`, `minimax_h3_layout` and the sampling sigmas. Veda replaces
exactly that call through `transformer_options["optimized_attention_override"]`:

  * weights are untouched, so any LoRA, fine-tune or quantized H3 checkpoint
    works, and T2VA / FL2VA / R2VA conditioning nodes are unchanged;
  * no `patches_replace` slot is taken, so block patches such as H3's Fun
    ControlNet keep working;
  * calls Veda declines run the override that was installed before it (or
    ComfyUI's attention), so other attention nodes still apply there.

The override is re-installed on every step (ON_PREPARE_STATE), like
ComfyUI's own sparse attention node, so a node applied later cannot
silently replace it. Node text: a few readable lines (kernel, video and
plan, sparsity, share of full attention computed); `verbose` adds timing
and call diagnostics.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import weakref

import torch

import comfy.patcher_extension

from . import backends
from . import settings as veda_settings
from . import status as veda_status
from .core import engine as veda_engine
from .core import h3_layout
from .core import plans as veda_plans

try:  # ComfyUI with the comfy-aimdo allocation graph (0.38+)
    from comfy.model_prefetch import pause_malloc_graph as _pause_malloc_graph
except ImportError:  # older ComfyUI: nothing to pause
    _pause_malloc_graph = contextlib.nullcontext

_KEY = 'veda_sparse_attention'
_TIMED_PHASES = (('gather', 'gather'), ('score', 'score'),
                 ('select', 'select'), ('attend', 'kernel'),
                 ('scatter', 'scatter'))


def _is_interrupt(error: BaseException) -> bool:
    # comfy.model_management.InterruptProcessingException, without importing
    # it at module scope.
    return type(error).__name__ == 'InterruptProcessingException'


class _Run:
    """Per sampling run: what happened, for the summary on the node."""

    def __init__(self):
        self.calls = collections.Counter()
        self.evaluations = 0          # model calls (layer 0 reached)
        self.video = None             # e.g. '1344x768 · 5.2 s'
        self.failed = None            # error text if the sparse path failed
        self.announced: set = set()


class VedaPatch:
    """Options and runtime state of one patched model."""

    def __init__(self, bundle, settings: veda_settings.VedaSettings,
                 node_id: str | None):
        self.bundle = bundle
        self.settings = settings
        self.status = veda_status.NodeStatus(node_id)
        self.installed: set = set()
        self._engines: dict[str, veda_engine.VedaEngine | None] = {}
        self._timers: dict[str, veda_engine.PhaseTimer] = {}
        self._steps = weakref.WeakKeyDictionary()
        self.run = _Run()

    @property
    def calls(self) -> collections.Counter:
        return self.run.calls

    # -- per device ------------------------------------------------------

    def _engine(self, device: torch.device) -> veda_engine.VedaEngine | None:
        key = str(device)
        if key not in self._engines:
            resolution = backends.resolve(
                device,
                notify=self.status.show)
            if resolution.backend is None:
                self.status.warn(
                    f'Veda off: no sparse kernel works on '
                    f'{resolution.device.label}; using full attention\n'
                    f'{resolution.report()}')
                self._engines[key] = None
            else:
                engine = veda_engine.VedaEngine(
                    self.bundle, self.settings.generated,
                    self.settings.reference, resolution.backend, device)
                if self.settings.verbose:
                    timer = engine.enable_timing()
                    if timer is not None:
                        self._timers[key] = timer
                    logging.info('Veda: backends on %s: %s',
                                 resolution.device.label, resolution.report())
                self._engines[key] = engine
        return self._engines[key]

    def _step(self, transformer_options) -> int | None:
        sigmas = transformer_options.get('sigmas')
        schedule = transformer_options.get('sample_sigmas')
        if sigmas is None or schedule is None:
            return None
        try:
            return self._steps[sigmas]
        except (KeyError, TypeError):
            pass
        step = veda_settings.step_index(float(sigmas.flatten()[0]),
                                        schedule.flatten().tolist())
        try:
            self._steps[sigmas] = step
        except TypeError:
            pass
        return step

    # -- the override ------------------------------------------------------

    def make_override(self, previous):
        """The attention override; declined calls go to `previous`."""
        patch = self

        def override(func, q, k, v, heads, mask=None, attn_precision=None,
                     skip_reshape=False, skip_output_reshape=False,
                     **kwargs):
            def dense(reason: str):
                patch.run.calls[reason] += 1
                kw = dict(mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)
                if previous is None:
                    return func(q, k, v, heads, **kw)
                return previous(func, q, k, v, heads, **kw)

            options = kwargs.get('transformer_options') or {}
            layout = options.get('minimax_h3_layout')
            if (mask is not None or not skip_reshape or q.dim() != 4
                    or q.shape[0] != 1 or q.shape != k.shape
                    or q.shape != v.shape or layout is None
                    or getattr(layout, 'seq_len', None) != q.shape[2]):
                return dense('other attention')
            if options.get('block_index') == 0:
                patch.run.evaluations += 1
            reason = patch._dense_reason(q, options)
            if reason is not None:
                return dense(reason)
            try:
                with _pause_malloc_graph():
                    return patch._sparse(q, k, v, layout, options,
                                         skip_output_reshape, dense)
            except Exception as error:  # never lose the user's render
                if _is_interrupt(error):
                    raise
                patch.run.failed = f'{type(error).__name__}: {error}'
                logging.error('Veda: sparse attention failed', exc_info=True)
                patch.status.warn(
                    'Veda hit an error and finishes this run with full '
                    f'attention:\n{patch.run.failed}')
                return dense('error')

        return override

    def _dense_reason(self, q, options) -> str | None:
        """Why this H3 call runs full attention, or None to go sparse."""
        s, bundle = self.settings, self.bundle
        if self.run.failed:
            return 'error'
        layer = options.get('block_index')
        if not isinstance(layer, int) or layer >= bundle.num_layers:
            return 'layer outside the predictor'
        if q.shape[1] != bundle.num_heads or q.shape[3] != bundle.head_dim:
            self._announce(('shape', tuple(q.shape)),
                           f'Veda off: the model has {q.shape[1]} heads of '
                           f'dim {q.shape[3]}, the predictor expects '
                           f'{bundle.num_heads} x {bundle.head_dim}',
                           warn=True)
            return 'head mismatch'
        if layer in s.dense_layers:
            return 'full-attention layer'
        if s.dense_steps and self._step(options) in s.dense_steps:
            return 'full-attention step'
        if s.generated.keeps_all and s.reference.keeps_all:
            return 'sparsity 0%'
        return None

    def _sparse(self, q, k, v, layout, options, skip_output_reshape, dense):
        # ComfyUI records each block's allocations into a malloc graph
        # (comfy-aimdo) and expects everything allocated inside a block to
        # be gone by its end. Veda keeps device state across calls (tile
        # layouts, plan head groups, statistics) and its kernels allocate
        # their own workspaces, so the caller runs this with the graph
        # paused, like ComfyUI's own sparse attention node does for its
        # persistent state. Otherwise the process aborts natively.
        engine = self._engine(q.device)
        if engine is None:
            return dense('no sparse kernel')
        try:
            spec = engine.layout_spec(layout)
        except h3_layout.LayoutError as error:
            self._announce(('layout', str(error)),
                           f'Veda off for this video: cannot read the H3 '
                           f'layout ({error})', warn=True)
            return dense('layout')
        choice = engine.plan_for(spec)
        self.run.video = veda_plans.describe_grid(spec.target.grid)
        self._announce(('plan', spec.target.grid),
                       self._running_text(engine, spec, choice),
                       warn=not choice.exact)
        batch, heads, seq_len, dim = q.shape
        out = engine.attention(q[0].transpose(0, 1), k[0].transpose(0, 1),
                               v[0].transpose(0, 1), options['block_index'],
                               spec, choice.plan)
        self.run.calls['sparse'] += 1
        if skip_output_reshape:
            return out.transpose(0, 1).unsqueeze(0)
        return out.reshape(batch, seq_len, heads * dim)

    # -- node text ---------------------------------------------------------

    def _running_text(self, engine, spec, choice) -> str:
        lines = [f'Veda running · {engine.backend.display}',
                 f'Video: {veda_plans.describe_grid(spec.target.grid)}',
                 f'Tile plan: {choice.how}']
        lines.append(f'Sparsity: {self.settings.describe()}')
        if spec.references:
            count = len(spec.references)
            mode = ('full attention' if self.settings.reference.keeps_all
                    else 'tiled, ' + veda_settings.format_budget(
                        self.settings.reference) + ' sparse')
            lines.append(f'References: {count} span'
                         f'{"s" if count > 1 else ""} ({mode})')
        full = self.settings.describe_full_attention()
        if full:
            lines.append(f'Full attention: {full}')
        if self.settings.verbose and spec.skipped:
            lines.append('Untiled references: ' + '; '.join(spec.skipped))
        return '\n'.join(lines)

    def _summary(self) -> str | None:
        run = self.run
        engines = [e for e in self._engines.values() if e is not None]
        sparse = run.calls.get('sparse', 0)
        full = sum(n for r, n in run.calls.items()
                   if r not in ('sparse', 'other attention'))
        if not sparse and not full:
            return None
        backend = engines[0].backend.display if engines else 'full attention'
        headline = ('Veda done' if sparse and not run.failed
                    else 'Veda done, fell back to full attention')
        lines = [f'{headline} · {backend}']
        if run.video:
            lines.append(f'Video: {run.video}')
        work = [e.stats.compute_fraction() for e in engines]
        work = [w for w in work if w is not None]
        if work:
            share = sum(work) / len(work)
            if full:  # dense calls do all of their work
                share = (share * sparse + full) / (sparse + full)
            lines.append(f'Attention computed: {100 * share:.1f}% of full '
                         f'attention ({100 * (1 - share):.1f}% skipped)')
        configured = self.settings.describe_full_attention()
        if configured:
            lines.append(f'Full attention: {configured}')
        if run.failed:
            lines.append(f'Fell back to full attention after: {run.failed}')
        if self.settings.verbose:
            lines += self._diagnostics(engines)
        return '\n'.join(lines)

    def _diagnostics(self, engines) -> list[str]:
        run, lines = self.run, ['-- diagnostics --']
        for key, timer in self._timers.items():
            phases = timer.summary()
            total = sum(phases.values()) / 1000.0
            if not total:
                continue
            per_eval = total / max(1, run.evaluations)
            lines.append(f'Veda attention time: {total:.2f} s '
                         f'({per_eval:.2f} s per model call x '
                         f'{run.evaluations})')
            lines.append('  ' + ' · '.join(
                f'{label} {phases.get(name, 0.0) / 1000.0:.2f} s'
                for name, label in _TIMED_PHASES))
            self._timers[key] = self._engines[key].enable_timing()
        for engine in engines:
            kept = engine.stats.kept_fraction()
            if kept is not None:
                lines.append(f'Video tiles kept: {100 * kept:.1f}% of '
                             'video x video tile pairs')
            chunking = engine.chunking
            if chunking.get('chunks_per_layer'):
                free = chunking.get('free_bytes')
                lines.append(
                    f'Chunks: {chunking["chunks_per_layer"]} per layer, '
                    f'{chunking["heads_per_chunk"]} heads each'
                    + (f' ({free / 2**30:.1f} GB free)' if free else ''))
        reasons = ', '.join(f'{r} {n}' for r, n in sorted(run.calls.items())
                            if r != 'sparse')
        lines.append(f'Attention calls: {run.calls.get("sparse", 0)} sparse'
                     + (f'; full attention: {reasons}' if reasons else ''))
        lines.append(f'Predictor: {self.bundle.describe()}')
        return lines

    def _announce(self, key, text: str, warn: bool = False) -> None:
        if key in self.run.announced:
            return
        self.run.announced.add(key)
        (self.status.warn if warn else self.status.show)(text)

    # -- lifecycle ---------------------------------------------------------

    def install(self, transformer_options: dict) -> None:
        """Puts the override on top of whatever override is on the hook;
        idempotent once it is on top."""
        current = transformer_options.get('optimized_attention_override')
        if current in self.installed:
            return
        override = self.make_override(current)
        self.installed.add(override)
        transformer_options['optimized_attention_override'] = override

    def on_cleanup(self) -> None:
        """End of a sampling run: show the summary, reset per-run state."""
        summary = self._summary()
        if summary:
            self.status.show(summary)
        for engine in self._engines.values():
            if engine is not None:
                engine.stats = veda_engine.Stats()
        self.run = _Run()


def apply(model, bundle, settings: veda_settings.VedaSettings,
          node_id: str | None):
    """Returns a clone of `model` with Veda attention installed."""
    patch = VedaPatch(bundle, settings, node_id)
    patched = model.clone()
    patch.install(patched.model_options.setdefault('transformer_options', {}))
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, _KEY,
        lambda model_patcher, timestep, model_options: patch.install(
            model_options['transformer_options']))
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP, _KEY,
        lambda model_patcher: patch.on_cleanup())
    return patched, patch
