import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const source = (await readFile(new URL("../web/h3_presets.js", import.meta.url), "utf8"))
    .replace(/^import \{ app \}.*$/m, "const app = { registerExtension() {} };");
const {prepareQuickNode} = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
test("quick node uses four standard serialized controls without hidden rows", () => {
    const node = { widgets: ["cache", "sparse", "tokens", "log_activity"].map(name => ({name, value: name})),
        computeSize: () => [200, 160], setSize(size) { this.size = size; } };
    prepareQuickNode(node);
    assert.deepEqual(node.size, [390,160]);
    assert.deepEqual(node.widgets.map(w => w.label), ["Temporal cache","Attention acceleration","Token reduction","Detailed logs"]);
    assert.deepEqual(node.widgets.map(w => w.value), ["cache","sparse","tokens","log_activity"]);
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
