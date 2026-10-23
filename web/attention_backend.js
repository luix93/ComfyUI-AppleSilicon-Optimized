import { app } from "../../scripts/app.js";

// Attention Backend (Apple Silicon): former backend names, as in
// _patches/attention_backend.LEGACY_BACKENDS. The server accepts them too.
export const CLASS_NAME = "AppleSiliconAttentionBackend";
export const LEGACY_BACKENDS = { "int8_dense_accurate": "dense_metal", "int8_dense_approximate": "dense_metal_v2" };
export function migrateAttentionBackends(graph) {
    for (const node of graph.nodes || []) {
        const renamed = node.type === CLASS_NAME && LEGACY_BACKENDS[node.widgets_values?.[0]];
        if (renamed) node.widgets_values[0] = renamed;
    }
    for (const subgraph of graph.definitions?.subgraphs || []) migrateAttentionBackends(subgraph);
}
app.registerExtension({
    name: "AppleSilicon.AttentionBackendNames",
    beforeConfigureGraph(graph) { migrateAttentionBackends(graph); },
});
