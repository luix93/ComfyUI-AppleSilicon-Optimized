// A Generate click evaluates the connected inputs and this node only.
export function promptStudioTarget(output, node, root) {
    const matches = Object.keys(output).filter(id => {
        if (output[id].class_type !== "AppleSiliconPromptStudio") return false;
        let graph = root, found;
        for (const part of id.split(":")) {
            found = graph?.getNodeById(part);
            graph = found?.subgraph;
        }
        return found === node;
    });
    if (matches.length !== 1) throw new Error("Cannot identify this Prompt Studio instance. Keep it outside shared subgraphs when generating from the button.");
    return matches[0];
}

export function generationPrompt(output, target, settings, jobId) {
    if (output[target]?.class_type !== "AppleSiliconPromptStudio") throw new Error("Prompt Studio is muted, bypassed or missing from the executable graph.");
    const prompt = {};
    function visit(id) {
        if (Object.hasOwn(prompt, id)) return;
        const original = output[id];
        if (!original) throw new Error(`A connected input node (${id}) is missing.`);
        prompt[id] = structuredClone(original);
        for (const value of Object.values(original.inputs || {})) {
            if (Array.isArray(value) && value.length === 2 && Number.isInteger(value[1]) && Object.hasOwn(output, value[0])) visit(String(value[0]));
        }
    }
    visit(String(target));
    prompt[target].inputs.mode = "generate_on_queue";
    prompt[target].inputs.settings = JSON.stringify({ ...settings, _studio_job: jobId });
    return prompt;
}

async function json(api, path, data) {
    const response = await api.fetchApi(path, data === undefined ? {} : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
    });
    const result = await response.json();
    if (!response.ok) {
        const details = Object.values(result.node_errors || {}).flatMap(n => (n.errors || []).map(e => e.message + (e.details ? `: ${e.details}` : "")));
        throw new Error(details.join("\n") || result.error?.message || result.error || `Request failed (${response.status}).`);
    }
    return result;
}

export async function generateConnected({ api, getPrompt, getTarget, settings, jobId, signal, onStatus }) {
    let queued = false;
    const cancelled = () => { if (signal.aborted) throw new Error("Generation cancelled."); };
    const progress = event => {
        if (event.detail?.prompt_id === jobId) onStatus("Preparing connected inputs and writing your prompt…");
    };
    api.addEventListener?.("execution_start", progress);
    try {
        onStatus("Preparing connected inputs…");
        const { output } = await getPrompt();
        cancelled();
        const target = getTarget(output);
        const prompt = generationPrompt(output, target, settings, jobId);
        await json(api, "/prompt", { prompt, prompt_id: jobId, client_id: api.clientId,
            partial_execution_targets: [target] });
        queued = true;
        onStatus("Queued · your connected references will be included.");
        while (true) {
            cancelled();
            const history = await json(api, `/history/${jobId}`);
            cancelled();
            const record = history[jobId];
            if (record) {
                const failure = record.status?.messages?.find(([type]) => type === "execution_error" || type === "execution_interrupted");
                if (failure) throw new Error(failure[1]?.exception_message || "Generation cancelled.");
                const result = record.outputs?.[target]?.studio_result?.[0];
                if (result?.prompt) return result;
                throw new Error("Prompt Studio produced no result. Restart ComfyUI to load the updated node.");
            }
            await new Promise(resolve => {
                const done = () => { clearTimeout(timer); signal.removeEventListener("abort", done); resolve(); };
                const timer = setTimeout(done, 400);
                signal.addEventListener("abort", done, { once: true });
            });
        }
    } finally {
        api.removeEventListener?.("execution_start", progress);
        if (queued && signal.aborted) await json(api, `/api/jobs/${jobId}/cancel`, {}).catch(() => {});
    }
}
