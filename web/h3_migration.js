import { app } from "../../scripts/app.js";

export const unifiedDefaults = {
    cache_mode: "disabled", cache_reuse_threshold: 0.20,
    cache_start_percent: 0.15, cache_end_percent: 0.95,
    cache_max_consecutive_skips: 2, cache_audio_weight: 1, cache_verbose: true,
    token_reduction: false, token_reduction_scale: 1, token_early_evaluations: 10,
    token_detail_restore_block: 30, token_coarse_restore_block: 40, token_verbose: false,
    sparse_attention: false, sparse_mode: "corrected", sparse_tau: 1,
    sparse_start_percent: 0.20, sparse_end_percent: 0.85,
    sparse_dense_leading_blocks: 2, sparse_min_sequence_tokens: 0, sparse_verbose: true,
};
const cacheModes = ["first_block_cache", "first_block_cache_per_step", "oracle", "block_oracle"];
const legacy = new Set([
    "AppleSiliconH3Cache", "AppleSiliconH3StructuralAcceleration",
    "AppleSiliconH3SparseAttention", "AppleSiliconH3Acceleration", "AppleSiliconH3TurboLoRA",
]);

export function migrateH3Workflow(graph, warn = console.warn) {
    for (const node of graph.nodes ?? []) {
        const oldLoRA = node.type === "AppleSiliconH3LoRA" && typeof node.widgets_values?.[1] === "number";
        if (!legacy.has(node.type) && !oldLoRA) continue;
        const refuse = (reason) => warn(`H3 node ${node.id}: ${reason}. Replace this retired node explicitly; its settings were not changed.`);
        if (node.inputs?.some((input) => input.name !== "model" && input.link != null)) {
            refuse("connected widget inputs require manual migration"); continue;
        }
        const v = node.widgets_values ?? [];
        let values;
        if (oldLoRA || node.type === "AppleSiliconH3TurboLoRA") {
            values = [JSON.stringify([{
                enabled: true, lora_name: v[0], strength: v[1] ?? 1,
                validation: oldLoRA ? "standard" : "turbo",
            }]), v[2] ?? "fast_runtime"];
            node.type = "AppleSiliconH3LoRA";
        } else {
            const out = { ...unifiedDefaults };
            if (node.type === "AppleSiliconH3Cache") {
                if (!cacheModes.includes(v[6]) || (v[5] ?? 0) !== 0) {
                    refuse("the retired cache experiment is not supported by H3 Acceleration"); continue;
                }
                Object.assign(out, { cache_mode: v[6], cache_reuse_threshold: v[0],
                    cache_start_percent: v[1], cache_end_percent: v[2], cache_max_consecutive_skips: v[3],
                    cache_audio_weight: v[4], cache_verbose: v[7] ?? true });
            } else if (node.type === "AppleSiliconH3StructuralAcceleration") {
                if (!v[0]) { refuse("the retired layer-thinning mode is unsupported"); continue; }
                Object.assign(out, { token_reduction: true, token_reduction_scale: v[2],
                    token_early_evaluations: v[4], token_detail_restore_block: v[5],
                    token_coarse_restore_block: v[6], token_verbose: v[7] ?? false });
            } else if (node.type === "AppleSiliconH3SparseAttention") {
                if (!["corrected", "dense_reference", "oracle"].includes(v[0]) ||
                    (v[7] ?? 0) !== 0 || Number(v[8] ?? 64) !== 64) {
                    refuse("the retired sparse experiment is unsupported"); continue;
                }
                Object.assign(out, { sparse_attention: true, sparse_mode: v[0], sparse_tau: v[1],
                    sparse_start_percent: v[2], sparse_dense_leading_blocks: v[3],
                    sparse_min_sequence_tokens: v[4], sparse_verbose: v[5] ?? true,
                    sparse_end_percent: v[6] ?? 0.85 });
            } else if ((v[0] ?? "exact") !== "exact") {
                refuse("that old preset was never calibrated"); continue;
            }
            values = Object.values(out);
            node.type = "AppleSiliconH3UnifiedAcceleration";
        }
        node.inputs = (node.inputs ?? []).filter((input) => input.name === "model");
        node.widgets_values = values;
        // Old registry metadata can otherwise make the editor suggest reinstalling removed nodes.
        if (node.properties?.["Node name for S&R"]) node.properties["Node name for S&R"] = node.type;
    }
    for (const subgraph of graph.definitions?.subgraphs ?? []) migrateH3Workflow(subgraph, warn);
}

app.registerExtension({
    name: "AppleSiliconFP8.H3NodeMigration",
    beforeConfigureGraph(graph) {
        migrateH3Workflow(graph, (detail) => {
            console.warn(detail);
            app.extensionManager?.toast?.add({ severity: "warn", summary: "H3 workflow migration", detail, life: 15000 });
        });
    },
});
