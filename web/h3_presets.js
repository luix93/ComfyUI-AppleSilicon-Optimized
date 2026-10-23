import { app } from "../../scripts/app.js";

// Use ComfyUI's standard combo/toggle widgets: compact in both node renderers,
// with normal keyboard navigation, primitive connections and serialization.
export const CLASS_NAME = "AppleSiliconH3QuickAcceleration";
// Removed or renamed attention recipes, as in minimax_h3_presets.SPARSE_ALIASES.
export const SPARSE_ALIASES = {
    "VSA Metal · ComfyUI (64)": "VSA Metal · FastH3 Omni (128)",
    "Dense INT8 · protected": "Dense Metal · protected",
    "Dense INT8 · full draft": "Dense Metal · full experimental",
    "Dense INT8 v2 · protected": "Dense Metal · protected",
    "Dense INT8 v2 · full experimental": "Dense Metal · full experimental",
    "Dense QK INT8 / PV BF16 · protected": "Dense Metal · protected",
    "Dense QK INT8 / PV BF16 · full experimental": "Dense Metal · full experimental",
    "SOL INT8 · hybrid balanced": "SOL Metal · balanced",
    "Veda Metal · protected": "SOL Metal · fast",
    "Veda Metal · fast draft": "SOL Metal · top-k wide",
    "SOL INT8 · conservative": "SOL Metal · conservative",
    "SOL INT8 · balanced": "SOL Metal · balanced",
    "SOL INT8 · fast": "SOL Metal · fast",
    "SOL INT8 · top-k wide": "SOL Metal · top-k wide",
    "SOL INT8 · top-k all-step draft": "SOL Metal · top-k all-step draft",
    "SOL INT8 · hybrid fast": "SOL Metal · hybrid fast",
    "Dense INT8 v3 · protected": "Dense Metal · protected",
    "Dense INT8 v3 · balanced": "Dense Metal · balanced",
    "Dense INT8 v3 · full experimental": "Dense Metal · full experimental",
};
export function migrateQuickPresets(graph) {
    for (const node of graph.nodes || []) {
        if (node.type !== CLASS_NAME || !node.widgets_values) continue;
        if (node.widgets_values[2] === "Fast · experimental") {
            node.widgets_values[2] = "Fast protected · experimental";
        }
        const sparse = SPARSE_ALIASES[node.widgets_values[1]];
        if (sparse) node.widgets_values[1] = sparse;
    }
    for (const subgraph of graph.definitions?.subgraphs || []) migrateQuickPresets(subgraph);
}
export function prepareQuickNode(node) {
    const labels = { cache: "Temporal cache", sparse: "Attention acceleration", tokens: "Token reduction", log_activity: "Detailed logs", approximate_references: "Approximate reference / guide attention" };
    for (const widget of node.widgets || []) {
        if (labels[widget.name]) widget.label = labels[widget.name];
    }
    node.color = "#303c50";
    node.bgcolor = "#202835";
    const size = node.computeSize?.();
    if (size) node.setSize?.([Math.max(390, size[0]), size[1]]);
}
app.registerExtension({
    name: "AppleSilicon.H3QuickPresets",
    beforeConfigureGraph(graph) { migrateQuickPresets(graph); },
    nodeCreated(node) {
        if ((node.comfyClass || node.constructor?.comfyClass || node.type) === CLASS_NAME) prepareQuickNode(node);
    },
});
