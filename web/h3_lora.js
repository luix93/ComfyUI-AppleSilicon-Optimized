import { app } from "../../scripts/app.js";

// Store one ordered list for both workflow serialization and API execution.
// Autogrow currently accepts sockets, not repeated groups of inline widgets.
export function prepareLoRA(node, files) {
    const value = node.widgets.find((widget) => widget.name === "loras");
    value.type = "converted-widget:h3-loras";
    value.computeSize = () => [0, -4];
    let entries = [];
    const extra = new Set();
    const save = () => {
        value.value = JSON.stringify(entries);
        node.graph?.change?.();
        node.setDirtyCanvas?.(true, true);
    };
    const add = (type, name, initial, callback, options = {}) => {
        const widget = node.addWidget(type, name, initial, callback, { ...options, serialize: false });
        extra.add(widget);
        return widget;
    };
    const rebuild = () => {
        node.widgets = node.widgets.filter((widget) => !extra.has(widget));
        extra.clear();
        entries.forEach((entry, index) => {
            const prefix = `LoRA ${index + 1}`;
            add("toggle", `${prefix} enabled`, entry.enabled ?? true, (v) => { entry.enabled = v; save(); });
            // Keep missing saved filenames visible so execution can report them.
            const choices = files.includes(entry.lora_name) ? files : [entry.lora_name, ...files].filter(Boolean);
            add("combo", `${prefix} file`, entry.lora_name, (v) => { entry.lora_name = v; save(); }, { values: choices });
            add("number", `${prefix} weight`, entry.strength ?? 1, (v) => { entry.strength = v; save(); },
                { min: -10, max: 10, step: 0.1, precision: 2 });
            add("combo", `${prefix} validation`, entry.validation ?? "standard", (v) => { entry.validation = v; save(); },
                { values: ["standard", "turbo"] });
            add("button", `Remove LoRA ${index + 1}`, null, () => {
                entries.splice(index, 1); save(); rebuild();
            });
        });
        add("button", "Add LoRA", null, () => {
            entries.push({ enabled: true, lora_name: files[0] ?? "", strength: 1, validation: "standard" });
            save(); rebuild();
        });
        node.setSize([Math.max(390, node.size?.[0] ?? 0), node.computeSize()[1]]);
        node.setDirtyCanvas?.(true, true);
    };
    const restore = () => {
        // Never replace malformed saved input with an empty, silently bypassed list.
        try {
            const parsed = JSON.parse(value.value);
            if (!Array.isArray(parsed) || parsed.some((entry) => !entry || typeof entry !== "object" || Array.isArray(entry))) {
                throw new Error("H3 LoRA selection must be a list of entries");
            }
            entries = parsed;
            rebuild();
        } catch (error) {
            console.error("H3 LoRA: cannot restore saved selection", error);
        }
    };
    value.callback = restore;
    value.serializeValue = () => value.value;
    const serialize = node.onSerialize;
    node.onSerialize = function (info) {
        const result = serialize?.call(this, info);
        info.widgets_values = [value.value, this.widgets.find((widget) => widget.name === "mode").value];
        return result;
    };
    const configure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = configure?.apply(this, args);
        restore();
        return result;
    };
    restore();
}

app.registerExtension({
    name: "AppleSiliconFP8.H3LoRAStack",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AppleSiliconH3LoRA") return;
        const files = nodeData.input.required.loras[1].h3_lora_files ?? [];
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            prepareLoRA(this, files);
            return result;
        };
    },
});
