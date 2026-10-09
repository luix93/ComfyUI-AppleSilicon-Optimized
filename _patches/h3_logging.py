"""Concise work summaries. These estimates exclude routing/replay/cache overhead."""


def call_summary(name, skipped, total):
    computed = total - skipped
    if total <= 0 or computed <= 0:
        return f"MiniMax H3 {name} - skipped {skipped}/{total} model calls (speedup unavailable)"
    return (f"MiniMax H3 {name} - skipped {skipped}/{total} model calls "
            f"(estimated {total / computed:.2f}x DiT speedup)")


def block_summary(name, skipped, total):
    computed = total - skipped
    if total <= 0 or computed <= 0:
        return f"MiniMax H3 {name} - skipped {skipped}/{total} transformer blocks (speedup unavailable)"
    return (f"MiniMax H3 {name} - skipped {skipped}/{total} transformer blocks "
            f"(estimated {total / computed:.2f}x block-compute speedup)")


def spectrum_summary(stats):
    # Rollback/discarded actual calls cost compute, but do not add native steps.
    # Offline smoothing reuses archived outputs: those calls are NOT extra DiT
    # executions and must not be counted as additional saved native calls.
    actual = stats.actual_transformer_calls
    forecast = max(0, stats.forecast_model_calls - stats.speculative_forecast_calls)
    extra = stats.discarded_actual_calls + stats.replayed_transformer_calls
    native = max(0, actual - extra) + forecast
    saved = native - actual
    if native <= 0 or actual <= 0:
        return f"MiniMax H3 Spectrum - {actual} DiT calls (speedup unavailable)"
    return (f"MiniMax H3 Spectrum - saved {saved}/{native} DiT calls "
            f"(estimated {native / actual:.2f}x DiT speedup; replay/predictor overhead excluded)")
