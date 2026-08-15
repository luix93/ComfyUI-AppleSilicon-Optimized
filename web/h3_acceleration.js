// Dynamic controls for the unified MiniMax H3 acceleration node.
//
// The Python schema remains complete so workflow JSON is stable and API
// execution needs no frontend. In the visual editor, inactive groups collapse
// to keep the node compact and the internal input names receive readable labels.
import { app } from "../../scripts/app.js";

const NODE_CLASS = "AppleSiliconH3UnifiedAcceleration";
const HIDDEN_TYPE = "converted-widget:asfp8-h3-hidden";
const MIN_WIDTH = 410;

const LABELS = {
    cache_mode: "Cache Setup",
    cache_reuse_threshold: "Reuse Threshold",
    cache_start_percent: "Start Percent",
    cache_end_percent: "End Percent",
    cache_max_consecutive_skips: "Maximum Consecutive Skips",
    cache_audio_weight: "Audio Veto Weight",
    cache_verbose: "Cache Logging",
    token_reduction: "Token Reduction",
    token_reduction_scale: "Update Scale",
    token_early_evaluations: "Early Evaluations",
    token_detail_restore_block: "Later Restore Block",
    token_coarse_restore_block: "Early Restore Block",
    token_verbose: "Token Reduction Logging",
    sparse_attention: "Sparse Attention",
    sparse_mode: "Sparse Mode",
    sparse_tau: "Routing Tau",
    sparse_start_percent: "Start Percent",
    sparse_end_percent: "End Percent",
    sparse_dense_leading_blocks: "Dense Leading Blocks",
    sparse_min_sequence_tokens: "Minimum Sequence Tokens",
    sparse_verbose: "Sparse Logging",
};

const CACHE_SETTINGS = [
    "cache_reuse_threshold",
    "cache_start_percent",
    "cache_end_percent",
    "cache_max_consecutive_skips",
    "cache_audio_weight",
    "cache_verbose",
];
const CACHE_TUNING = CACHE_SETTINGS.slice(0, -1);
const TOKEN_SETTINGS = [
    "token_reduction_scale",
    "token_early_evaluations",
    "token_detail_restore_block",
    "token_coarse_restore_block",
    "token_verbose",
];
const SPARSE_SETTINGS = [
    "sparse_mode",
    "sparse_tau",
    "sparse_start_percent",
    "sparse_end_percent",
    "sparse_dense_leading_blocks",
    "sparse_min_sequence_tokens",
    "sparse_verbose",
];

function widget(node, name) {
    return node.widgets?.find((candidate) => candidate.name === name);
}

function hideWidget(node, target) {
    if (!target || target.__asfp8Hidden) return;
    target.__asfp8Hidden = true;
    target.__asfp8OriginalType = target.type;
    target.__asfp8OriginalComputeSize = target.computeSize;
    target.type = HIDDEN_TYPE;
    target.computeSize = () => [0, -4];
    for (const linked of target.linkedWidgets || []) hideWidget(node, linked);
}

function showWidget(node, target) {
    if (!target || !target.__asfp8Hidden) return;
    target.__asfp8Hidden = false;
    target.type = target.__asfp8OriginalType;
    target.computeSize = target.__asfp8OriginalComputeSize;
    for (const linked of target.linkedWidgets || []) showWidget(node, linked);
}

function setGroupVisible(node, names, visible) {
    for (const name of names) {
        const target = widget(node, name);
        if (visible) showWidget(node, target);
        else hideWidget(node, target);
    }
}

function refresh(node) {
    const cacheMode = widget(node, "cache_mode")?.value ?? "disabled";
    const cacheEnabled = cacheMode !== "disabled";
    const cacheIsOracle = cacheMode === "oracle" || cacheMode === "block_oracle";
    setGroupVisible(node, CACHE_SETTINGS, cacheEnabled);
    // Oracle modes always run the exact model; thresholds, windows and skip
    // counts do not affect them. Keep only their diagnostic logging control.
    if (cacheEnabled && cacheIsOracle) setGroupVisible(node, CACHE_TUNING, false);

    setGroupVisible(
        node,
        TOKEN_SETTINGS,
        Boolean(widget(node, "token_reduction")?.value),
    );
    setGroupVisible(
        node,
        SPARSE_SETTINGS,
        Boolean(widget(node, "sparse_attention")?.value),
    );

    const computed = node.computeSize?.() || node.size || [MIN_WIDTH, 100];
    node.setSize?.([Math.max(MIN_WIDTH, node.size?.[0] || 0), computed[1]]);
    node.setDirtyCanvas?.(true, true);
}

function installToggle(node, name) {
    const target = widget(node, name);
    if (!target || target.__asfp8ToggleInstalled) return;
    target.__asfp8ToggleInstalled = true;
    const original = target.callback;
    target.callback = function (...args) {
        const result = original?.apply(this, args);
        refresh(node);
        return result;
    };
}

function prepare(node) {
    if (node.__asfp8H3Prepared) return;
    node.__asfp8H3Prepared = true;
    for (const target of node.widgets || []) {
        if (LABELS[target.name]) target.label = LABELS[target.name];
    }
    installToggle(node, "cache_mode");
    installToggle(node, "token_reduction");
    installToggle(node, "sparse_attention");
    refresh(node);
}

app.registerExtension({
    name: "AppleSiliconFP8.H3AccelerationControls",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalCreated?.apply(this, args);
            prepare(this);
            // Saved widget values can be assigned immediately after creation.
            queueMicrotask(() => refresh(this));
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = originalConfigure?.apply(this, args);
            queueMicrotask(() => refresh(this));
            return result;
        };
    },
});
