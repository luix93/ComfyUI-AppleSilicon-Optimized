import assert from "node:assert/strict";
import { test } from "node:test";
import { generationPrompt, generateConnected, promptStudioTarget } from "../web/prompt_studio_graph.js";

const graph = () => ({
    1: { class_type: "LoadImage", inputs: { image: "synthetic.png" } },
    2: { class_type: "AppleSiliconPromptStudio", inputs: { images: ["1", 0], mode: "use_saved", settings: "{}", prompt: "saved", negative_prompt: "" } },
    3: { class_type: "DownstreamDiffusion", inputs: { text: ["2", 0] } },
    4: { class_type: "UnrelatedOutput", inputs: {} },
});

test("Generate isolates ancestors and preserves stored workflow mode", () => {
    const original = graph(), before = structuredClone(original);
    const result = generationPrompt(original, "2", { idea: "Describe this" }, "one");
    assert.deepEqual(Object.keys(result), ["1", "2"]);
    assert.equal(result[2].inputs.mode, "generate_on_queue");
    assert.equal(JSON.parse(result[2].inputs.settings)._studio_job, "one");
    assert.deepEqual(original, before);
});

test("Target resolution handles root nodes and unique subgraph instances", () => {
    const node = {}, root = { getNodeById: id => id === "2" ? node : undefined };
    assert.equal(promptStudioTarget(graph(), node, root), "2");
    const subgraph = { getNodeById: id => id === "9" ? { subgraph: root } : undefined };
    assert.equal(promptStudioTarget({ "9:2": graph()[2] }, node, subgraph), "9:2");
    assert.throws(() => promptStudioTarget({}, node, root), /identify/);
});

test("Click queues only the target, returns its result, and supports repeated generation", async () => {
    const calls = [];
    const api = { clientId: "test-client", async fetchApi(path, options) {
        calls.push([path, options?.body && JSON.parse(options.body)]);
        return { ok: true, json: async () => path === "/prompt" ? {} : {
            [path.split("/").at(-1)]: { outputs: { 2: { studio_result: [{ prompt: "Result", model: "fixture" }] } } },
        } };
    } };
    for (const jobId of ["job-a", "job-b"]) {
        const result = await generateConnected({ api, getPrompt: async () => ({ output: graph() }), getTarget: () => "2",
            settings: {}, jobId, signal: new AbortController().signal, onStatus() {} });
        assert.equal(result.prompt, "Result");
    }
    const posts = calls.filter(([path]) => path === "/prompt");
    assert.deepEqual(posts.map(([, body]) => body.prompt_id), ["job-a", "job-b"]);
    for (const [, body] of posts) {
        assert.deepEqual(body.partial_execution_targets, ["2"]);
        assert.deepEqual(Object.keys(body.prompt), ["1", "2"]);
        assert.equal(body.extra_data, undefined); // No full-workflow metadata upload.
    }
});

test("Cancellation targets only this job, including cancel during submission", async () => {
    const controller = new AbortController(), paths = [];
    const api = { async fetchApi(path) {
        paths.push(path);
        if (path === "/prompt") controller.abort();
        return { ok: true, json: async () => ({}) };
    } };
    await assert.rejects(generateConnected({ api, getPrompt: async () => ({ output: graph() }), getTarget: () => "2",
        settings: {}, jobId: "own-job", signal: controller.signal, onStatus() {} }), /cancelled/);
    assert.deepEqual(paths, ["/prompt", "/api/jobs/own-job/cancel"]);
});

test("Upstream failures reach the composer instead of replacing the saved prompt", async () => {
    const api = { async fetchApi(path) { return { ok: true, json: async () => path === "/prompt" ? {} : {
        broken: { status: { messages: [["execution_error", { exception_message: "Synthetic loader failure" }]] } },
    } }; } };
    await assert.rejects(generateConnected({ api, getPrompt: async () => ({ output: graph() }), getTarget: () => "2",
        settings: {}, jobId: "broken", signal: new AbortController().signal, onStatus() {} }), /Synthetic loader failure/);
});
