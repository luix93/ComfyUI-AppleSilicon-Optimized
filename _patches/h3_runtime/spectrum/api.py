"""Our H3 node's stable Spectrum interface; GPL-3.0-or-later derived code."""
from .config import SpectrumH3Config

def make_config(*, protected, blend_weight=0.5, verbose=False):
    return SpectrumH3Config(
        enabled=True, blend_weight=float(blend_weight), degree=1,
        ridge_lambda=0.1, window_size=2.0, flex_window=0.75,
        warmup_steps=4 if protected else 1,
        tail_actual_steps=2 if protected else 1, max_history=8,
        debug=bool(verbose), history_storage="vram",
        bootstrap_first_forecast=not protected,
        anchor_residual_feedback=False, selective_rollback_correction=False,
        offline_smoothing_replay=True, audio_blend_weight=0.0,
        offline_archive_storage="vram", model_aware_mode="off",
        model_aware_trust_shrinkage=False,
        model_aware_replay_generic_correction=False,
    ).validate()

def apply(model, *, protected, blend_weight=0.5, verbose=False):
    # Keep the proven compatibility hook order, scoped to our runtime classes.
    # Loading config/predictor code alone never installs these hooks.
    from . import compat
    from .minimax_h3 import install_h3_wrapper, require_native_minimax_h3
    from .runtime import SpectrumH3Runtime
    from .sampling import install_sampler_wrappers
    require_native_minimax_h3(model)
    config = make_config(protected=protected, blend_weight=blend_weight, verbose=verbose)
    patched = model.clone()
    require_native_minimax_h3(patched)
    install_sampler_wrappers(patched, SpectrumH3Runtime(config))
    install_h3_wrapper(patched)
    return patched
