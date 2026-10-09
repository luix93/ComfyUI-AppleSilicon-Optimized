import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AppleSilicon.H3LatentUpscale",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AppleSiliconH3LatentUpscale") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = original?.apply(this, args);
            const mode = this.widgets?.find(w => w.name === "mode");
            const refresh = () => {
                const names = {scale: ["scale"], dimensions: ["target_width", "target_height"], megapixels: ["megapixels"]};
                for (const w of this.widgets || []) {
                    if (!["scale", "target_width", "target_height", "megapixels"].includes(w.name)) continue;
                    if (!w.__asfp8Original) w.__asfp8Original = {type: w.type, computeSize: w.computeSize};
                    const show = (names[mode?.value] || []).includes(w.name);
                    w.type = show ? w.__asfp8Original.type : "converted-widget:asfp8-h3-hidden";
                    w.computeSize = show ? w.__asfp8Original.computeSize : () => [0, -4];
                }
                this.setSize?.(this.computeSize());
                this.setDirtyCanvas?.(true, true);
            };
            const callback = mode?.callback;
            if (mode) mode.callback = function (...values) { const r = callback?.apply(this, values); refresh(); return r; };
            const configured = this.onConfigure;
            this.onConfigure = function (...values) { const r = configured?.apply(this, values); refresh(); return r; };
            refresh();
            return result;
        };
    },
});
