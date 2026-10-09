import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { buildPromptStudio } from "./prompt_studio_panel.js";
import { generateConnected, promptStudioTarget } from "./prompt_studio_graph.js";

const NODE = "AppleSiliconPromptStudio";
const WIDTH = 450;
async function transport(path, data) {
    const response = await api.fetchApi(`/asfp8/prompt-studio${path}`, data === undefined ? {} : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
    });
    let result;
    try { result = await response.json(); }
    catch { throw new Error("Prompt Studio is unavailable. Restart ComfyUI after installing the node."); }
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
    return result;
}
function prepare(node) {
    if (node.__asPromptStudio) return;
    const widgets = node.widgets.filter(w => w.options?.serialize !== false);
    const byName = new Map(widgets.map(w => [w.name, w]));
    for (const w of widgets) {
        w.hidden = true;
        w.computeSize = () => [0, -4];
        w.computeLayoutSize = () => ({ minWidth: 0, minHeight: 0, maxHeight: 0 });
    }
    if (!document.querySelector('link[data-as-prompt-studio]')) {
        const link = document.createElement("link"); link.rel = "stylesheet";
        link.href = new URL("./prompt_studio.css", import.meta.url).href;
        link.dataset.asPromptStudio = "true"; document.head.append(link);
    }
    const connected = name => node.inputs?.some(i => (i.name === name || i.widget?.name === name) && i.link != null);
    const controls = {
        get: name => byName.get(name)?.value,
        set(name, value) { if (!connected(name) && byName.has(name)) byName.get(name).value = value; },
        connected: () => ["idea", "images", "video", "audio"].filter(connected),
        generateConnected: (settings, jobId, signal, onStatus) => generateConnected({
            api, getPrompt: () => app.graphToPrompt(),
            getTarget: output => promptStudioTarget(output, node, app.rootGraph || app.graph),
            settings, jobId, signal, onStatus,
        }),
        changed() { node.graph?.change?.(); node.setDirtyCanvas?.(true, true); },
    };
    let height = 430;
    const panel = buildPromptStudio(controls, transport);
    const dom = node.addDOMWidget("Prompt Studio", "as-prompt-studio", panel.element, {
        serialize: false, hideOnZoom: false, margin: 0,
        getMinHeight: () => height, getMaxHeight: () => height, getHeight: () => height,
    });
    dom.computeSize = () => [WIDTH, height];
    function resize() {
        const measured = Math.ceil(panel.content.scrollHeight);
        if (measured < 100) return;
        height = measured + 4;
        node.setSize([Math.max(WIDTH, node.size?.[0] || 0), node.computeSize()[1]]);
        node.setDirtyCanvas?.(true, true);
    }
    const observer = new ResizeObserver(resize); observer.observe(panel.content);
    const removed = node.onRemoved;
    node.onRemoved = function (...args) { observer.disconnect(); panel.destroy(); return removed?.apply(this, args); };
    const serialize = node.onSerialize;
    node.onSerialize = function (data) { serialize?.apply(this, arguments); data.widgets_values = widgets.map(w => w.value); };
    node.__asPromptStudio = { panel, resize };
    const connectionsChanged = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = connectionsChanged?.apply(this, args);
        panel.connectionsChanged();
        return result;
    };
    requestAnimationFrame(resize);
}
const matches = node => (node.comfyClass || node.constructor?.comfyClass || node.type) === NODE;
app.registerExtension({
    name: "AppleSilicon.PromptStudio",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configure?.apply(this, args);
            this.__asPromptStudio?.panel.refresh(); requestAnimationFrame(() => this.__asPromptStudio?.resize());
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message, ...args) {
            const result = executed?.call(this, message, ...args);
            // Button-generated results are applied by the panel after its
            // draft-revision check, not by this generic queue callback.
            if (message.studio_job?.[0]) return result;
            for (const name of ["prompt", "negative_prompt"]) {
                const w = this.widgets?.find(w => w.name === name);
                if (w && typeof message[name]?.[0] === "string") w.value = message[name][0];
            }
            this.__asPromptStudio?.panel.refresh(); this.graph?.change?.();
            return result;
        };
    },
    nodeCreated(node) { if (matches(node)) prepare(node); },
    loadedGraphNode(node) { if (matches(node)) { prepare(node); node.__asPromptStudio.panel.refresh(); } },
});
