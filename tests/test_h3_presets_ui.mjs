import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const source = (await readFile(new URL("../web/h3_presets.js", import.meta.url), "utf8"))
    .replace(/^import \{ app \}.*$/m, "const app = { registerExtension() {} };");
const {prepareQuickNode, migrateQuickPresets, SPARSE_ALIASES} = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
test("quick node uses five standard serialized controls without hidden rows", () => {
    const node = { widgets: ["cache", "sparse", "tokens", "log_activity", "approximate_references"].map(name => ({name, value: name})),
        computeSize: () => [200, 160], setSize(size) { this.size = size; } };
    prepareQuickNode(node);
    assert.deepEqual(node.size, [390,160]);
    assert.deepEqual(node.widgets.map(w => w.label), ["Temporal cache","Attention acceleration","Token reduction","Detailed logs","Approximate reference / guide attention"]);
    assert.deepEqual(node.widgets.map(w => w.value), ["cache","sparse","tokens","log_activity","approximate_references"]);
    assert.ok(node.widgets.every(w => !w.hidden && !w.computeLayoutSize));
});
test("preparation leaves connected widgets and unknown controls intact", () => {
    const node = { widgets: [{name:"cache", value:"Off"},{name:"future",value:42}], computeSize: () => [450,170],
        setSize(size) { this.size=size; } };
    prepareQuickNode(node);
    assert.equal(node.widgets[0].value, "Off");
    assert.equal(node.widgets[1].value, 42);
    assert.deepEqual(node.size,[450,170]);
});

test("old Fast graphs and subgraphs migrate only to protected", () => {
    const legacy = () => ({type:"AppleSiliconH3QuickAcceleration", widgets_values:["Off","Off","Fast · experimental",false]});
    const graph={nodes:[legacy(),{type:"OtherNode",widgets_values:["Off","Off","Fast · experimental"]}],definitions:{subgraphs:[{nodes:[legacy()]}]}};
    migrateQuickPresets(graph);
    assert.equal(graph.nodes[0].widgets_values[2],"Fast protected · experimental");
    assert.equal(graph.definitions.subgraphs[0].nodes[0].widgets_values[2],"Fast protected · experimental");
    assert.equal(graph.nodes[1].widgets_values[2],"Fast · experimental");
    assert.deepEqual(graph.nodes[0].widgets_values.slice(0,2),["Off","Off"]);
});

test("removed attention presets migrate to the same replacements as the server", async () => {
    const python = await readFile(new URL("../_patches/minimax_h3_presets.py", import.meta.url), "utf8");
    const block = python.slice(python.indexOf("SPARSE_ALIASES = {"), python.indexOf("TOKEN_PRESETS = {"));
    const server = Object.fromEntries([...block.matchAll(/"([^"]+)": "([^"]+)"/g)].map(m => [m[1], m[2]]));
    assert.deepEqual(SPARSE_ALIASES, server);
    const graph = {nodes: [{type: "AppleSiliconH3QuickAcceleration", widgets_values: ["Off", "Veda Metal · protected", "Off", false]}]};
    migrateQuickPresets(graph);
    assert.equal(graph.nodes[0].widgets_values[1], "SOL Metal · fast");
});
