import { app } from "../../scripts/app.js";

// Use ComfyUI's standard combo/toggle widgets: compact in both node renderers,
// with normal keyboard navigation, primitive connections and serialization.
export const CLASS_NAME = "AppleSiliconH3QuickAcceleration";
export function migrateQuickPresets(graph) {
    for (const node of graph.nodes || []) {
        if (node.type === CLASS_NAME && node.widgets_values?.[2] === "Fast · experimental") {
            node.widgets_values[2] = "Fast protected · experimental";
        }
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
