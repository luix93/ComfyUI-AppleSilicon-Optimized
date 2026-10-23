import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const source = (await readFile(new URL("../web/attention_backend.js", import.meta.url), "utf8"))
    .replace(/^import \{ app \}.*$/m, "const app = { registerExtension() {} };");
const {migrateAttentionBackends, LEGACY_BACKENDS} = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));

test("former Attention Backend names migrate in graphs and subgraphs", async () => {
    const python = await readFile(new URL("../_patches/attention_backend.py", import.meta.url), "utf8");
    assert.match(python, /LEGACY_BACKENDS = \{"int8_dense_accurate": BACKEND_INT8_ACCURATE,\s+"int8_dense_approximate": BACKEND_INT8\}/);
    assert.deepEqual(LEGACY_BACKENDS, {"int8_dense_accurate": "dense_metal", "int8_dense_approximate": "dense_metal_v2"});
    const node = () => ({type: "AppleSiliconAttentionBackend", widgets_values: ["int8_dense_accurate"]});
    const graph = {nodes: [node(), {type: "AppleSiliconAttentionBackend", widgets_values: ["int8_dense_approximate"]}, {type: "OtherNode", widgets_values: ["int8_dense_accurate"]}], definitions: {subgraphs: [{nodes: [node()]}]}};
    migrateAttentionBackends(graph);
    assert.equal(graph.nodes[0].widgets_values[0], "dense_metal");
    assert.equal(graph.definitions.subgraphs[0].nodes[0].widgets_values[0], "dense_metal");
    assert.equal(graph.nodes[1].widgets_values[0], "dense_metal_v2");
    assert.equal(graph.nodes[2].widgets_values[0], "int8_dense_accurate");
});
