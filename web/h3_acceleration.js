import { app } from "../../scripts/app.js";
import { buildH3Panel } from "./h3_panel.js";

const NODE_CLASS = "AppleSiliconH3UnifiedAcceleration";
const MIN_WIDTH = 400;
const OFF = { cache_mode: "disabled", token_reduction: false, sparse_attention: false };
const FBC = { cache_mode: "first_block_cache", cache_reuse_threshold: 0.20,
    cache_start_percent: 0.15, cache_end_percent: 0.95,
    cache_max_consecutive_skips: 2, cache_audio_weight: 1 };
const SOL = { sparse_attention: true, sparse_mode: "corrected", sparse_backend: "int8",
    sparse_topk_ratio: 0, sparse_tau: 1.3, sparse_start_percent: 0.20, sparse_end_percent: 0.90,
    sparse_dense_leading_blocks: 2, sparse_min_sequence_tokens: 8192,
    sparse_dense_layers: "0,1,33,34,35", sparse_window_mode: "step_fraction" };
const VSA = { ...OFF, sparse_attention: true, sparse_backend: "vsa_metal",
    sparse_start_percent: 0, sparse_end_percent: 1, sparse_dense_leading_blocks: 0,
    sparse_dense_layers: "", sparse_min_sequence_tokens: 12288,
    vsa_keep_percent: 10, vsa_head_chunk: 4 };

export const PRESETS = [
    { label: "MotionCache · recommended", settings: { ...OFF, cache_mode: "motion_cache", motion_reuse_threshold: 0.15, motion_strength: 1, motion_warmup_steps: 4, motion_max_consecutive_skips: 2, motion_start_percent: 0.15, motion_end_percent: 0.95 },
        lines: ["Closest content preservation in our tests.", "MotionCache 0.15 · first 4 evaluations fully computed.", "Included · no extra nodes."] },
    { label: "Exact · no approximation", settings: OFF,
        lines: ["Disables this node's approximate accelerators.", "Your exact Apple Silicon kernels remain active.", "Other nodes and environment options still apply."] },
    { label: "VSA Metal · FastH3 Omni (128)", settings: { ...VSA, vsa_tile_size: 128, vsa_reference_keep_percent: 10 },
        lines: ["FastH3 trained gates · 128-token tiles · 10% keep.", "All steps/layers eligible; reference videos routed separately.", "Experimental · cache and token reduction Off."] },
    { label: "Spectrum · protected", settings: { ...OFF, cache_mode: "spectrum_protected", spectrum_blend_weight: 0.5 },
        lines: ["Faster, with some observed motion changes.", "First 4 / last 2 evaluations fully computed.", "Included · no extra nodes."] },
    { label: "Spectrum · fast draft", settings: { ...OFF, cache_mode: "spectrum_fast", spectrum_blend_weight: 0.5 },
        lines: ["Fastest recent preset; larger appearance changes.", "First / last evaluation fully computed.", "Included · no extra nodes."] },
    { label: "FirstBlockCache · 0.20", settings: { ...OFF, ...FBC },
        lines: ["Fast residual reuse; composition can change.", "Window 0.15–0.95 · at most 2 consecutive skips.", "Built in · no extra node package."] },
    { label: "SOL Metal · fast", settings: { ...OFF, ...SOL },
        lines: ["INT8 sparse video attention · tau 1.3.", "At 20 steps: first 4 / last 2 stay dense.", "Protected layers 0, 1, 33, 34, 35 · built in."] },
    { label: "Dense Metal · protected", settings: { ...OFF, ...SOL, sparse_backend: "dense_int8_v3" },
        lines: ["All keys retained; Kitchen-style INT8 numerics (rotation, per-tile P).", "At 20 steps: first 4 / last 2 and protected layers stay exact.", "Lowest INT8 error measured · ~8% slower attention than v2 · M5 only."] },
    { label: "Dense Metal · balanced", settings: { ...OFF, ...SOL, sparse_backend: "dense_int8_v3",
        sparse_start_percent: 0.10, sparse_end_percent: 0.95 },
        lines: ["Kitchen-style INT8 between protected and full.", "At 20 steps: first 2 / last 1 and protected layers stay exact.", "17 of 20 steps quantized · M5 only."] },
    { label: "Dense Metal · full experimental", settings: { ...OFF, ...SOL, sparse_backend: "dense_int8_v3",
        sparse_start_percent: 0, sparse_end_percent: 1, sparse_dense_leading_blocks: 0, sparse_dense_layers: "" },
        lines: ["Dense Metal (INT8 v3) attention at every step and layer.", "Visual references can be approximated with the separate switch.", "Experimental · output video/audio can change."] },
    { label: "SOL Metal · hybrid fast", settings: { ...OFF, ...SOL, sparse_backend: "int8_hybrid" },
        lines: ["SOL Fast plus dense INT8 in its protected layers.", "First 4 / last 2 sampling steps stay exact at 20 steps.", "Small additional gain; quality changes can accumulate."] },
    { label: "SOL Metal · top-k wide", settings: { ...OFF, ...SOL, sparse_topk_ratio: 0.15,
        sparse_start_percent: 0.10, sparse_end_percent: 0.95 },
        lines: ["15% video-block budget + global/local blocks and pooled tail.", "At 20 steps: first 2 / last 1 stay dense; protected layers.", "Experimental · background morphing reported; prefer SOL Fast for that case."] },
    { label: "SOL Metal · top-k all-step draft", settings: { ...OFF, ...SOL, sparse_topk_ratio: 0.10,
        sparse_start_percent: 0, sparse_end_percent: 1, sparse_dense_leading_blocks: 0,
        sparse_dense_layers: "" },
        lines: ["10% video-block budget + global/local blocks and pooled tail.", "Every step and layer eligible · fastest SOL recipe.", "Draft · larger changes to composition, motion and audio possible."] },
    { label: "Token reduction · protected", settings: { ...OFF, token_reduction: true,
        token_reduction_scale: 1, token_phase_policy: "evaluation_count", token_early_evaluations: 10, token_coarse_restore_block: 12,
        token_detail_restore_block: 8, token_start_percent: 0.20, token_end_percent: 0.90 },
        lines: ["Short reduced span: early/later restore at blocks 12/8.", "At 20 steps: first 4 / last 2 retain full spatial tokens.", "Experimental · protection does not guarantee identical detail."] },
    { label: "Token reduction · balanced", settings: { ...OFF, token_reduction: true,
        token_reduction_scale: 1, token_phase_policy: "evaluation_count", token_early_evaluations: 10, token_coarse_restore_block: 30,
        token_detail_restore_block: 20, token_start_percent: 0.10, token_end_percent: 0.95 },
        lines: ["Early/later restore at blocks 30/20.", "At 20 steps: first 2 / last 1 retain full spatial tokens.", "Experimental · between Light and Fast in reduced compute."] },
    { label: "SOL Metal + FirstBlockCache", settings: { ...OFF, ...SOL, ...FBC,
        sparse_tau: 0.5, cache_reuse_threshold: 0.10, cache_start_percent: 0.25,
        cache_end_percent: 0.85, cache_max_consecutive_skips: 1 },
        lines: ["Measured hybrid · SOL tau 0.5 + cache 0.10.", "Cache window 0.25–0.85 · at most 1 skip.", "Quality changes can accumulate · built in."] },
];


// The backend widgets stay in their original order for API and workflow
// serialization. A single DOM widget owns the entire visible layout.
export function accelerationControls(node) {
    const backend = node.widgets.filter(w => w.options?.serialize !== false);
    const byName = new Map(backend.map(w => [w.name, w]));
    node.properties ??= {};
    const connected = name => {
        const index = node.inputs?.findIndex(i => i.widget?.name === name) ?? -1;
        return index >= 0 && (node.isInputConnected ? node.isInputConnected(index) : node.inputs[index].link != null);
    };
    const get = name => byName.get(name)?.value;
    const set = (name, value) => { if (!connected(name) && byName.has(name)) byName.get(name).value = value; };
    const write = settings => Object.entries(settings).forEach(([name, value]) => set(name, value));
    return {
        backend, get, connected,
        set(name, value) {
            if (name === "sparse_attention" && value && !node.properties.asfp8_h3_sparse_configured) {
                write(SOL);
                node.properties.asfp8_h3_sparse_configured = true;
            }
            set(name, value);
        },
        preset(label) {
            const preset = PRESETS.find(p => p.label === label);
            if (preset && !backend.some(w => connected(w.name))) {
                write(preset.settings);
                if (preset.settings.sparse_attention) node.properties.asfp8_h3_sparse_configured = true;
            }
        },
        selectedPreset() {
            if (backend.some(w => connected(w.name))) return "Custom combination";
            if (get("sparse_attention") && ["int8", "int8_hybrid", "dense_int8", "dense_floatpv", "dense_int8_v2", "dense_int8_v3"].includes(get("sparse_backend")) && get("sparse_approximate_references")) return "Custom combination";
            return PRESETS.find(p => Object.entries(p.settings).every(([k,v]) => get(k) === v))?.label || "Custom combination";
        },
        changed() { node.graph?.change?.(); node.setDirtyCanvas?.(true, true); },
    };
}
function prepare(node) {
    if (node.__asfp8H3UI || !node.widgets?.some(w => w.name === "cache_mode")) return;
    const controls = accelerationControls(node);
    for (const w of controls.backend) {
        // Nodes 2.0 uses computeLayoutSize; computeSize alone leaves blank rows.
        w.hidden = true;
        w.computeSize = () => [0, -4];
        w.computeLayoutSize = () => ({ minWidth: 0, minHeight: 0, maxHeight: 0 });
    }
    if (!document.querySelector('link[data-asfp8-h3-panel]')) {
        const link = document.createElement("link");
        link.rel = "stylesheet"; link.href = new URL("./h3_panel.css", import.meta.url).href;
        link.dataset.asfp8H3Panel = "true"; document.head.append(link);
    }
    let height = 365;
    const panel = buildH3Panel(controls, PRESETS, node.__asfp8H3InputSpecs, () => node.properties, resize);
    const dom = node.addDOMWidget("H3 controls", "asfp8-h3-panel", panel.element, {
        serialize: false, hideOnZoom: false, margin: 0,
        getMinHeight: () => height, getMaxHeight: () => height, getHeight: () => height,
    });
    dom.computeSize = () => [MIN_WIDTH, height];
    function resize() {
        // offset/scroll heights are in CSS pixels, unaffected by graph zoom.
        const measured = Math.ceil(panel.content.scrollHeight);
        if (measured < 100) return;
        height = measured + 4;
        const size = node.computeSize();
        node.setSize([Math.max(MIN_WIDTH, node.size?.[0] || 0), size[1]]);
        node.setDirtyCanvas?.(true, true);
    }
    const observer = new ResizeObserver(() => resize());
    observer.observe(panel.content);
    const removed = node.onRemoved;
    node.onRemoved = function (...args) {
        observer.disconnect();
        return removed?.apply(this, args);
    };
    node.__asfp8H3UI = { controls, panel, resize };
    const serialize = node.onSerialize;
    node.onSerialize = function (data) {
        serialize?.apply(this, arguments);
        data.widgets_values = controls.backend.map(w => w.value);
    };
    // Widget callbacks from connected primitives should update the panel too.
    for (const w of controls.backend) {
        const original = w.callback;
        w.callback = function (...args) {
            const result = original?.apply(this, args);
            panel.refresh();
            return result;
        };
    }
    panel.refresh();
    requestAnimationFrame(resize);
}
// The four Veda inputs sat between sparse_memory_budget_gib and sparse_topk_ratio.
// Workflows saved before their removal carry four extra positional values.
const REMOVED_AFTER = "sparse_memory_budget_gib", REMOVED_COUNT = 4;
export function migrateRemovedWidgets(node, info) {
    const values = info?.widgets_values;
    const backend = node.widgets?.filter(w => w.options?.serialize !== false) || [];
    if (!Array.isArray(values) || values.length !== backend.length + REMOVED_COUNT) return;
    const at = backend.findIndex(w => w.name === REMOVED_AFTER) + 1;
    if (at <= 0) return;
    values.splice(at, REMOVED_COUNT);
    backend.forEach((w, i) => { w.value = values[i]; });
    // Veda selections run SOL INT8 (the server maps them too).
    const sparse = backend.find(w => w.name === "sparse_backend");
    if (sparse && String(sparse.value).startsWith("veda_")) sparse.value = "int8";
}
const isH3 = node => (node.comfyClass || node.constructor?.comfyClass || node.type) === NODE_CLASS;
app.registerExtension({
    name: "AppleSiliconFP8.H3AccelerationControls",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;
        nodeType.prototype.__asfp8H3InputSpecs = { ...nodeData.input.required, ...nodeData.input.optional };
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            migrateRemovedWidgets(this, args[0]);
            const result = configure?.apply(this, args);
            // A loaded workflow owns its sparse values, even when switched off.
            this.properties.asfp8_h3_sparse_configured = true;
            this.__asfp8H3UI?.panel.refresh();
            requestAnimationFrame(() => this.__asfp8H3UI?.resize());
            return result;
        };
        const connections = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const result = connections?.apply(this, args);
            queueMicrotask(() => this.__asfp8H3UI?.panel.refresh());
            return result;
        };
    },
    nodeCreated(node) { if (isH3(node)) prepare(node); },
    loadedGraphNode(node) {
        if (!isH3(node)) return;
        prepare(node);
        node.__asfp8H3UI?.panel.refresh();
        requestAnimationFrame(() => node.__asfp8H3UI?.resize());
    },
});
