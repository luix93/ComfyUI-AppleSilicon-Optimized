import { prepareMedia } from "./prompt_studio_media.js";

const E = (tag, cls = "", text = "") => { const e = document.createElement(tag); e.className = cls; e.textContent = text; return e; };
const button = (label, action, cls = "") => { const b = E("button", `ps-button ${cls}`, label); b.type = "button"; b.onclick = action; return b; };
const field = (label, control, hint = "") => { const el = E("label", "ps-field"); el.append(E("span", "ps-label", label), control); if (hint) el.append(E("span", "ps-hint", hint)); return el; };
const input = (value, change, type = "text", placeholder = "") => { const e = E("input", "ps-input"); e.type = type; e.value = value ?? ""; e.placeholder = placeholder; e.oninput = () => change(type === "number" ? Number(e.value) : e.value); return e; };
const area = (value, change, placeholder = "", rows = 4) => { const e = E("textarea", "ps-input"); e.value = value || ""; e.placeholder = placeholder; e.rows = rows; e.oninput = () => change(e.value); return e; };
const select = (value, options, change) => { const e = E("select", "ps-input"); for (const item of options) { const [v, text] = Array.isArray(item) ? item : [item, item]; const o = E("option", "", text); o.value = v; e.append(o); } e.value = value; e.onchange = () => change(e.value); return e; };
const check = (label, value, change) => { const box = input("", () => change(box.checked), "checkbox"); box.checked = Boolean(value); const wrap = E("label", "ps-check"); wrap.append(box, E("span", "", label)); return wrap; };
const grid = (...children) => { const el = E("div", "ps-grid"); el.append(...children); return el; };
const row = (...children) => { const el = E("div", "ps-row"); el.append(...children); return el; };
const heading = (title, description) => { const el = E("div", "ps-section-title"); el.append(E("h2", "", title), E("p", "ps-muted", description)); return el; };
const kinds = [["image", "Image"], ["video", "Video"], ["audio", "Audio"]];
const providerDefaults = { omlx: ["oMLX", "http://localhost:8000/v1"], lmstudio: ["LM Studio", "http://localhost:1234/v1"], ollama: ["Ollama", "http://localhost:11434/v1"], openai: ["OpenAI compatible", "http://localhost:8000/v1"] };
const INITIAL = { version: 1, connection: "omlx", idea: "", style: "", detail: "balanced", aspect_ratio: "model default", duration: 5, additional: "", temperature: 0.7, seed: -1, frame_count: 4, reference_fps: 24, image_batch_mode: "auto", references: [] };

export function buildPromptStudio(controls, api) {
    let ws = null, dialog = null, tab = "compose", busy = false, job = null, controller = null, connectedJob = false, status = "", error = false, generationRevision = 0;
    let selectedProfile = null, selectedConnection = null, librarySearch = "", libraryKind = "all", favorites = false;
    let modalStatus, modalBody, generateButton, modalGenerate, inlineBrief, inlinePrompt, inlineProfile, inlineModel, inlineMode;
    let discovering = false;
    let settings = readSettings();
    const root = E("div", "asps ps-node");
    const content = E("div", "ps-node-content"); root.append(content);
    const stop = event => { if (!(event.type === "keydown" && (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s")) event.stopPropagation(); };
    ["pointerdown", "dblclick", "keydown", "wheel"].forEach(name => root.addEventListener(name, stop));
    function readSettings() {
        try { return { ...INITIAL, ...JSON.parse(controls.get("settings") || "{}") }; }
        catch { return { ...INITIAL }; }
    }
    function commit() { controls.set("settings", JSON.stringify(settings)); controls.changed(); generationRevision++; }
    function set(name, value) { settings[name] = value; commit(); }
    function say(text, failed = false) { status = text; error = failed; refreshStatus(); }
    function refreshStatus() {
        inlineStatus.textContent = status || (controls.get("mode") === "generate_on_queue" ? "Generates a fresh prompt when queued." : "Saved text → STRING outputs");
        inlineStatus.classList.toggle("ps-error", error);
        if (modalStatus) { modalStatus.textContent = status || "Changes to this draft are saved in your workflow."; modalStatus.classList.toggle("ps-error", error); }
        if (generateButton) { generateButton.textContent = busy ? "Cancel generation" : "Generate prompt"; generateButton.classList.toggle("ps-busy", busy); }
        if (modalGenerate) modalGenerate.textContent = busy ? "Cancel generation" : "Generate prompt";
    }
    async function request(path, body) { return api(path, body); }
    async function load() {
        ws = await request("/workspace");
        if (!settings.profile) { settings.profile = structuredClone(ws.profiles[0]); settings.negative = settings.profile?.negative || ""; commit(); }
        refreshInline();
    }
    async function mutate(section, item, deleteId) {
        ws = await request("/workspace", { revision: ws.revision, section, ...(deleteId ? { delete: deleteId } : { item }) });
        refreshInline();
    }
    const attempt = action => async (...args) => { try { await action(...args); } catch (e) { say(e.message, true); } };
    function applyProfile(id) {
        const p = ws?.profiles.find(p => p.id === id);
        if (p) { settings.profile = structuredClone(p); settings.negative = p.negative || ""; commit(); }
        refreshInline(); renderModal();
    }
    function queueToggle() {
        const toggle = check("Generate on queue", controls.get("mode") === "generate_on_queue", value => {
            controls.set("mode", value ? "generate_on_queue" : "use_saved");
            controls.changed(); refreshInline();
        });
        toggle.className = "ps-queue-toggle";
        toggle.title = "On: generate a fresh prompt when the workflow runs. Off: use the prompt shown above.";
        toggle.querySelector("input").setAttribute("role", "switch");
        return toggle;
    }
    function modelOptions(connection = ws?.connections.find(c => c.id === settings.connection)) {
        const selected = settings.model || connection?.model || "";
        const ids = [...new Set([...(connection?.models || []), connection?.model, selected].filter(Boolean))];
        return { selected, options: [["", "Choose a model…"], ...ids.map(id => [id, id])] };
    }
    function chooseModel(value) { set("model", value); refreshInline(); renderModal(); }
    async function discoverModels(connectionId) {
        const data = await request("/models", { connection: connectionId });
        ws = data.workspace;
        refreshInline();
        return data.models;
    }
    async function generate() {
        if (busy) {
            controller?.abort();
            if (!connectedJob) await request("/cancel", { job_id: job });
            say("Cancelling…"); return;
        }
        connectedJob = Boolean(controls.connected?.().length);
        busy = true; job = crypto.randomUUID(); controller = new AbortController(); const revision = generationRevision;
        say(connectedJob ? "Preparing connected references…" : "Writing your prompt…");
        try {
            const result = connectedJob
                ? await controls.generateConnected(structuredClone(settings), job, controller.signal, say)
                : await request("/generate", { job_id: job, settings: structuredClone(settings) });
            if (controller.signal.aborted) { say("Generation cancelled."); return; }
            if (revision !== generationRevision) {
                await saveEntry(result.prompt, result.negative_prompt, "Generated before draft changed", generationSettings);
                say("Draft changed during generation. The result was kept in your library.");
                return;
            }
            controls.set("prompt", result.prompt); controls.set("negative_prompt", result.negative_prompt); controls.changed();
            const used = [result.visual_count ? `${result.visual_count} visual reference${result.visual_count === 1 ? "" : "s"}` : "", result.audio_count ? "audio" : ""].filter(Boolean).join(" + ");
            say(`${result.model} · ${result.elapsed}s${used ? ` · ${used} sent` : ""} · ready to use`); refreshInline(); renderModal();
        } catch (e) {
            if (controller.signal.aborted) { say("Generation cancelled."); return; }
            throw e;
        } finally { busy = false; job = null; controller = null; refreshStatus(); }
    }
    let generationSettings;
    const runGeneration = attempt(async () => { if (!busy) generationSettings = structuredClone(settings); await generate(); });
    async function saveEntry(prompt, negative, title, config = settings) {
        if (!prompt?.trim()) throw new Error("Generate or type a prompt before saving it.");
        await mutate("library", { id: crypto.randomUUID(), title: title || settings.idea?.slice(0, 80) || prompt.slice(0, 80),
            prompt, negative_prompt: negative || "", kind: config.profile?.kind || "image", profile: config.profile?.name || "Custom", settings: structuredClone(config), favorite: false, created: new Date().toISOString() });
        say("Saved to your prompt library.");
    }
    const saveCurrent = attempt(() => saveEntry(controls.get("prompt"), controls.get("negative_prompt")));
    function outputEditor(rows = 10) {
        const wrap = E("div", "ps-output");
        const title = row(E("span", "ps-eyebrow", "YOUR PROMPT"), button("Copy", attempt(async () => { await navigator.clipboard.writeText(controls.get("prompt") || ""); say("Prompt copied."); }), "ps-quiet"));
        wrap.append(title, area(controls.get("prompt"), v => { controls.set("prompt", v); controls.changed(); generationRevision++; if (inlinePrompt) inlinePrompt.value = v; }, "Your generated prompt appears here. You can also write or paste one.", rows),
            field("Negative prompt", area(controls.get("negative_prompt"), v => { controls.set("negative_prompt", v); controls.changed(); generationRevision++; }, "Optional exclusions", 2)),
            row(button("Save to library", saveCurrent), E("span", "ps-hint", "Editable · two STRING outputs")));
        return wrap;
    }
    inlineProfile = select("", [], applyProfile);
    inlineModel = select("", [], chooseModel);
    inlineModel.setAttribute("aria-label", "LLM model");
    const refreshModels = button("↻", attempt(async () => {
        if (discovering) return;
        discovering = true; refreshModels.disabled = true;
        say("Refreshing models…");
        try {
            const models = await discoverModels(settings.connection);
            say(`${models.length} model${models.length === 1 ? "" : "s"} saved for this connection.`);
            renderModal();
        } finally { discovering = false; refreshModels.disabled = false; }
    }), "ps-model-refresh");
    refreshModels.title = "Refresh available models"; refreshModels.setAttribute("aria-label", "Refresh models");
    inlineBrief = area(settings.idea, v => set("idea", v), "A quiet observatory above the clouds, just before sunrise…", 3);
    inlinePrompt = area(controls.get("prompt"), v => { controls.set("prompt", v); controls.changed(); generationRevision++; }, "Generate a prompt, or paste your own.", 5);
    inlineMode = queueToggle();
    generateButton = button("Generate prompt", runGeneration, "ps-primary");
    const inlineReferences = E("div", "ps-input-references");
    const inlineUpload = input("", () => {}, "file"); inlineUpload.multiple = true; inlineUpload.accept = "image/*,video/*,audio/*"; inlineUpload.hidden = true;
    inlineUpload.onchange = attempt(async () => { await attachFiles(inlineUpload.files); inlineUpload.value = ""; });
    const inlineStatus = E("div", "ps-status"); inlineStatus.setAttribute("role", "status");
    content.append(field("Target profile", inlineProfile), field("LLM model", row(inlineModel, refreshModels)),
        field("Your idea", inlineBrief), inlineReferences,
        row(button("+ Attach reference", () => inlineUpload.click(), "ps-quiet"), button("Open Studio ↗", attempt(() => open()), "ps-quiet")), inlineUpload,
        field("Prompt", inlinePrompt), generateButton, inlineMode, inlineStatus);

    function optionsFor(element, value, options) {
        element.replaceChildren();
        if (value && !options.some(o => o[0] === value)) options = [[value, `${value} · workflow copy`], ...options];
        for (const [id, label] of options) { const o = E("option", "", label); o.value = id; element.append(o); }
        element.value = value || "";
    }
    function refreshInline() {
        inlineBrief.value = settings.idea || ""; inlinePrompt.value = controls.get("prompt") || "";
        inlineMode.querySelector("input").checked = controls.get("mode") === "generate_on_queue";
        optionsFor(inlineProfile, settings.profile?.id, (ws?.profiles || []).map(p => [p.id, p.name]));
        const connection = ws?.connections.find(c => c.id === settings.connection);
        const models = modelOptions(connection);
        optionsFor(inlineModel, models.selected, models.options);
        inlineModel.title = [models.selected, connection?.name].filter(Boolean).join(" · ");
        refreshReferences();
        refreshStatus();
    }
    function refreshReferences() {
        const connected = controls.connected?.() || [];
        inlineReferences.replaceChildren();
        const labels = { idea: "Brief connected", images: "Images connected", video: "Video connected", audio: "Audio connected" };
        for (const name of connected) inlineReferences.append(E("span", "ps-reference-chip", labels[name]));
        if (settings.references.length) inlineReferences.append(E("span", "ps-reference-chip", `${settings.references.length} attached`));
        if (connected.length || settings.references.length) inlineReferences.append(E("span", "ps-hint", "Included with your idea when you press Generate."));
        inlineBrief.disabled = connected.includes("idea");
        inlineBrief.placeholder = connected.includes("idea") ? "Your brief comes from the connected text node." : "Describe what you want the LLM to write…";
    }
    async function attachFiles(files, audioStart = 0, audioDuration = 15) {
        if (!ws) await load();
        for (const file of files) {
            say(`Preparing ${file.name}…`);
            const refs = await prepareMedia(file, settings.frame_count, audioStart, audioDuration);
            const existing = ws.references.filter(r => settings.references.includes(r.id));
            if ([...existing, ...refs].filter(r => r.kind !== "audio").length > 8 || [...existing, ...refs].filter(r => r.kind === "audio").length > 1) throw new Error("Attach up to 8 images / frames and one audio excerpt. Remove a reference before adding more.");
            for (const ref of refs) { await mutate("references", ref); settings.references.push(ref.id); commit(); }
        }
        say("References ready. Press Generate to send them with your idea."); refreshInline(); renderModal();
    }
    async function open() {
        if (dialog) { dialog.showModal(); return; }
        await load();
        dialog = E("dialog", "asps ps-dialog");
        dialog.setAttribute("aria-label", "Apple Silicon Prompt Studio");
        const header = E("header", "ps-header");
        const identity = E("div", "ps-identity"); identity.append(E("span", "ps-brand-mark", "✦"), heading("Prompt Studio", "APPLE SILICON · YOUR LOCAL CREATIVE WORKSPACE"));
        const close = button("Close", () => dialog.close(), "ps-quiet");
        header.append(identity, close);
        const layout = E("div", "ps-layout"), nav = E("nav", "ps-nav"); nav.setAttribute("aria-label", "Studio sections");
        for (const [id, label, description] of [["compose", "Compose", "Shape an idea"], ["library", "Library", "Keep the good ones"], ["profiles", "Profiles", "Speak each model’s language"], ["connections", "Connections", "Your local LLMs"], ["workspace", "Workspace", "Back up & restore"]]) {
            const b = button("", () => { tab = id; renderModal(); }, "ps-nav-button");
            b.dataset.tab = id; b.append(E("strong", "", label), E("small", "", description)); nav.append(b);
        }
        nav.append(E("p", "ps-nav-note", "Prompt generation only. Your image, video and audio models run in the workflow."));
        modalBody = E("main", "ps-main"); layout.append(nav, modalBody);
        const footer = E("footer", "ps-footer"); modalStatus = E("span", "ps-status"); modalStatus.setAttribute("role", "status");
        modalGenerate = button("Generate prompt", runGeneration, "ps-primary"); footer.append(modalStatus, modalGenerate);
        dialog.append(header, layout, footer); document.body.append(dialog);
        dialog.addEventListener("close", () => { dialog.remove(); dialog = null; modalBody = null; modalStatus = null; modalGenerate = null; refreshInline(); });
        renderModal(); dialog.showModal();
    }
    function renderModal() {
        if (!dialog || !ws) return;
        dialog.querySelectorAll("[data-tab]").forEach(el => { el.classList.toggle("active", el.dataset.tab === tab); el.setAttribute("aria-current", el.dataset.tab === tab ? "page" : "false"); });
        modalBody.replaceChildren();
        ({ compose: composeView, library: libraryView, profiles: profilesView, connections: connectionsView, workspace: workspaceView })[tab]();
        modalBody.scrollTop = 0;
        refreshStatus();
    }
    function composeView() {
        modalBody.append(heading("From a spark to a scene.", "Choose a target, describe your idea, and let your local model develop it."));
        const left = E("div", "ps-stack"), right = E("div", "ps-stack");
        const profiles = ws.profiles.map(p => [p.id, p.name]);
        if (settings.profile?.id && !ws.profiles.some(p => p.id === settings.profile.id)) profiles.unshift([settings.profile.id, `${settings.profile.name} · workflow copy`]);
        left.append(grid(field("Target profile", select(settings.profile?.id, profiles, applyProfile)),
            field("LLM connection", select(settings.connection, ws.connections.map(c => [c.id, c.name]), v => { delete settings.model; set("connection", v); refreshInline(); renderModal(); }))),
            field("LLM model", select(modelOptions().selected, modelOptions().options, chooseModel)),
            field("Your idea", area(settings.idea, v => { set("idea", v); inlineBrief.value = v; }, "What would you like to create?", 4)),
            grid(field("Style", input(settings.style, v => set("style", v), "text", "Cinematic, editorial, ambient…")),
                field("Detail", select(settings.detail, ["concise", "balanced", "rich"], v => set("detail", v)))));
        const kind = settings.profile?.kind || "image";
        if (kind !== "audio") left.append(field("Aspect ratio", select(settings.aspect_ratio, ["model default", "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "21:9"], v => set("aspect_ratio", v))));
        if (kind !== "image") left.append(field("Duration · seconds", input(settings.duration, v => set("duration", v), "number")));
        left.append(field("Additional direction", area(settings.additional, v => set("additional", v), "Camera movement, mood, exact text, sound, things to preserve…", 3)),
            field("Things to avoid", area(settings.negative, v => set("negative", v), "Profile exclusions, or your own", 2)));
        const advanced = E("details", "ps-details"); advanced.append(E("summary", "", "Generation settings"),
            grid(field("Temperature · 0–2", input(settings.temperature, v => set("temperature", v), "number")), field("Seed · −1 for random", input(settings.seed, v => set("seed", v), "number"))),
            queueToggle(),
            E("p", "ps-hint", "Generate includes connected inputs automatically. It runs their source nodes and Prompt Studio, stopping before downstream generation. A seed requests repeatability; the LLM server controls determinism."));
        left.append(advanced);
        right.append(outputEditor(12), referencesView());
        const columns = E("div", "ps-compose-columns"); columns.append(left, right); modalBody.append(columns);
    }
    function referencesView() {
        const section = E("section", "ps-references");
        section.append(row(E("h3", "", "References"), E("span", "ps-pill", `${settings.references.length} attached`)),
            E("p", "ps-hint", "Images and sampled video frames need a vision LLM. Audio needs a model and server supporting input_audio."));
        const connected = controls.connected?.() || [];
        if (connected.length) section.append(E("p", "ps-connected-note", `Connected inputs: ${connected.join(", ")}. These are included automatically alongside attachments.`));
        const thumbs = E("div", "ps-thumbs");
        for (const id of settings.references) {
            const ref = ws.references.find(r => r.id === id), card = E("div", "ps-reference");
            if (ref?.kind !== "audio" && ref) { const img = E("img"); img.src = `data:${ref.mime};base64,${ref.data}`; img.alt = ref.name; card.append(img); }
            else if (ref) { const audio = E("audio"); audio.controls = true; audio.src = `data:audio/wav;base64,${ref.data}`; card.append(audio); }
            card.append(E("span", "", ref?.name || "Missing reference"), button("Remove", () => { set("references", settings.references.filter(i => i !== id)); renderModal(); }, "ps-quiet")); thumbs.append(card);
        }
        const upload = input("", () => {}, "file"); upload.multiple = true; upload.accept = "image/*,video/*,audio/*"; upload.setAttribute("aria-label", "Upload references");
        let audioStart = 0, audioDuration = 15;
        upload.onchange = attempt(async () => { await attachFiles(upload.files, audioStart, audioDuration); upload.value = ""; });
        const sampling = E("details", "ps-details"); sampling.append(E("summary", "", "Reference sampling"),
            grid(field("Video frames · 1–8", select(settings.frame_count, [1, 2, 3, 4, 6, 8].map(n => [String(n), String(n)]), v => set("frame_count", Number(v)))), field("Connected frame batch · FPS", input(settings.reference_fps, v => set("reference_fps", v), "number"))),
            field("Connected IMAGE batch", select(settings.image_batch_mode, [["auto", "Automatic"], ["images", "Separate images · include every image"], ["video", "Video frames · sample across the batch"]], v => set("image_batch_mode", v)), "Automatic includes every image in batches of up to 8; larger batches are sampled as video."),
            grid(field("Audio start · seconds", input(0, v => { audioStart = v; }, "number")), field("Audio length · up to 30s", input(15, v => { audioDuration = v; }, "number"))),
            E("p", "ps-hint", "Images are prepared at up to 1024 px. VIDEO uses the clip’s frame rate and samples across its duration. Connect AUDIO separately to include sound. Uploaded audio is a mono 24 kHz WAV excerpt; files up to 32 MiB. Connected AUDIO uses the first 30 seconds."));
        section.append(thumbs, field("Add image, video or audio", upload), sampling);
        return section;
    }
    function libraryView() {
        modalBody.append(heading("Your collection of good ideas.", "Search, favorite, and bring any saved prompt back into the composer."));
        const search = input(librarySearch, value => { librarySearch = value; renderCards(); }, "search", "Search prompts, profile names and titles…");
        search.setAttribute("aria-label", "Search library");
        const filters = row(search, select(libraryKind, [["all", "All types"], ...kinds], v => { libraryKind = v; renderCards(); }), check("Favorites", favorites, v => { favorites = v; renderCards(); }));
        const cards = E("div", "ps-library-grid");
        modalBody.append(filters, cards);
        function renderCards() {
            cards.replaceChildren();
            const entries = [...ws.library].reverse().filter(e => (libraryKind === "all" || e.kind === libraryKind) && (!favorites || e.favorite) && [e.title, e.prompt, e.negative_prompt, e.profile].join(" ").toLowerCase().includes(librarySearch.toLowerCase()));
            if (!entries.length) { cards.append(E("p", "ps-empty", ws.library.length ? "No prompts match these filters." : "Your library starts with one good prompt. Generate something, then choose Save to library.")); return; }
            for (const entry of entries) {
                const card = E("article", "ps-library-card"), star = button(entry.favorite ? "★" : "☆", attempt(async () => { await mutate("library", { ...entry, favorite: !entry.favorite }); renderCards(); }), "ps-star");
                star.setAttribute("aria-label", entry.favorite ? "Remove favorite" : "Add favorite"); star.setAttribute("aria-pressed", Boolean(entry.favorite));
                card.append(row(E("span", "ps-pill", entry.kind), star), E("h3", "", entry.title), E("p", "ps-excerpt", entry.prompt), E("small", "ps-muted", entry.profile || "Custom"),
                    row(button("Reuse prompt", () => { settings = { ...INITIAL, ...structuredClone(entry.settings) }; controls.set("prompt", entry.prompt); controls.set("negative_prompt", entry.negative_prompt || ""); commit(); tab = "compose"; refreshInline(); renderModal(); say("Prompt restored to this node."); }),
                        button("Delete", attempt(async () => { if (confirm("Delete this saved prompt?")) { await mutate("library", null, entry.id); renderCards(); } }), "ps-quiet")));
                cards.append(card);
            }
        }
        renderCards();
    }
    function profilesView() {
        modalBody.append(heading("A voice for every model.", "Profiles describe the target generation model. Your LLM connection writes the prompt."));
        const list = E("div", "ps-item-list"), editor = E("div", "ps-stack");
        const p = structuredClone(ws.profiles.find(p => p.id === selectedProfile) || ws.profiles[0] || { id: crypto.randomUUID(), name: "New profile", kind: "image", instructions: "", negative: "" });
        selectedProfile = p.id;
        for (const profile of ws.profiles) list.append(button(profile.name, () => { selectedProfile = profile.id; renderModal(); }, `ps-list-button ${profile.id === p.id ? "active" : ""}`));
        list.append(button("+ New profile", attempt(async () => { const id = crypto.randomUUID(); await mutate("profiles", { id, name: "New profile", kind: "image", instructions: "", negative: "" }); selectedProfile = id; renderModal(); })));
        editor.append(field("Profile name", input(p.name, v => { p.name = v; })), field("Output type", select(p.kind, kinds, v => { p.kind = v; })),
            field("Instructions", area(p.instructions, v => { p.instructions = v; }, "How should the LLM write prompts for this model?", 9)),
            field("Default negative prompt", area(p.negative, v => { p.negative = v; }, "Optional exclusions", 3)),
            row(button("Save profile", attempt(async () => { await mutate("profiles", p); if (settings.profile?.id === p.id) { settings.profile = structuredClone(p); commit(); } say("Profile saved. Use Apply to draft to reset its negative prompt too."); renderModal(); }), "ps-primary"),
                button("Apply to draft", () => { settings.profile = structuredClone(p); settings.negative = p.negative; commit(); tab = "compose"; refreshInline(); renderModal(); }),
                button("Delete", attempt(async () => { if (confirm("Delete this profile? Existing workflow drafts keep their copy.")) { await mutate("profiles", null, p.id); selectedProfile = null; renderModal(); } }), "ps-quiet")));
        const layout = E("div", "ps-edit-layout"); layout.append(list, editor); modalBody.append(layout);
    }
    function connectionsView() {
        modalBody.append(heading("Bring your own local model.", "Start your server, save its connection, then discover models. No extra Python client is needed."));
        const list = E("div", "ps-item-list"), editor = E("div", "ps-stack");
        const c = structuredClone(ws.connections.find(c => c.id === selectedConnection) || ws.connections.find(c => c.id === settings.connection) || ws.connections[0] || { id: crypto.randomUUID(), name: "New connection", provider: "openai", base_url: "http://localhost:8000/v1", model: "" });
        selectedConnection = c.id; let secret;
        for (const connection of ws.connections) list.append(button(connection.name, () => { selectedConnection = connection.id; renderModal(); }, `ps-list-button ${connection.id === c.id ? "active" : ""}`));
        list.append(button("+ New connection", attempt(async () => { const id = crypto.randomUUID(); await mutate("connections", { id, name: "New connection", provider: "openai", base_url: "http://localhost:8000/v1", model: "" }); selectedConnection = id; renderModal(); })));
        const name = input(c.name, v => { c.name = v; }), url = input(c.base_url, v => { c.base_url = v; });
        const model = input(c.model, v => { c.model = v; }, "text", "Load a model in your server, then refresh the list");
        model.setAttribute("list", `ps-models-${c.id}`); const modelList = E("datalist"); modelList.id = `ps-models-${c.id}`;
        modelList.replaceChildren(...(c.models || []).map(id => { const o = E("option"); o.value = id; return o; }));
        const modelsSelect = select(c.model, [["", "Choose a discovered model…"], ...[...new Set([...(c.models || []), c.model].filter(Boolean))].map(id => [id, id])], v => { if (v) { c.model = v; model.value = v; } });
        modelsSelect.setAttribute("aria-label", "Discovered models");
        const key = input("", v => { secret = v; }, "password", c.has_key ? "Saved · leave blank to keep" : "Optional API key"); key.autocomplete = "new-password";
        const save = async () => {
            ws = await request("/workspace", { revision: ws.revision, section: "connections", item: c, ...(secret !== undefined ? { api_key: secret } : {}) });
            const saved = ws.connections.find(i => i.id === c.id);
            Object.assign(c, saved); c.models = saved.models || [];
            secret = undefined; key.value = ""; key.placeholder = c.has_key ? "Saved · leave blank to keep" : "Optional API key";
            refreshInline();
        };
        editor.append(grid(field("Name", name), field("Server", select(c.provider, Object.entries(providerDefaults).map(([id, d]) => [id, d[0]]), v => { c.provider = v; [c.name, c.base_url] = providerDefaults[v]; name.value = c.name; url.value = c.base_url; }))),
            field("API base URL", url, "Runs from the ComfyUI machine. Use /v1, or the API path provided by your server."),
            grid(field("API key", key, "Stored privately on this machine. Excluded from workflows and backups."), field("Or API key environment variable", input(c.key_env, v => { c.key_env = v; }), "Takes precedence over a saved key. Set it before starting ComfyUI.")),
            row(button("Save & discover models", attempt(async () => {
                await save(); say("Checking the model list…"); c.models = await discoverModels(c.id);
                modelList.replaceChildren(...c.models.map(id => { const o = E("option"); o.value = id; return o; }));
                optionsFor(modelsSelect, c.model, [["", c.models.length ? "Choose a discovered model…" : "No models returned"], ...c.models.map(id => [id, id])]);
                say(`${c.models.length} model${c.models.length === 1 ? "" : "s"} saved. You can now switch models directly on the node.`);
            })), modelsSelect), field("LLM model ID", model), modelList,
            field("Max output tokens", input(c.max_tokens || 2048, v => { c.max_tokens = v; }, "number"), "Include room for reasoning. Start with 4096–8192 if your model thinks before answering."),
            E("p", "ps-hint", "References are sent automatically. Choose a vision-capable model for images or video frames, and an audio-capable model/server for audio. A text-only model can still write prompts without references."),
            row(button("Save connection", attempt(async () => { await save(); settings.connection = c.id; delete settings.model; commit(); refreshInline(); say("Connection saved and selected for this draft."); renderModal(); }), "ps-primary"),
                button("Clear saved key", attempt(async () => { secret = ""; await save(); say("Saved key removed."); })),
                button("Delete", attempt(async () => { if (confirm("Delete this connection and its saved key?")) { await mutate("connections", null, c.id); selectedConnection = null; renderModal(); } }), "ps-quiet")));
        const layout = E("div", "ps-edit-layout"); layout.append(list, editor); modalBody.append(layout);
    }
    function workspaceView() {
        modalBody.append(heading("Keep your workspace with you.", "Profiles, connections, saved prompts and prepared references travel together in one JSON file."));
        const stats = E("div", "ps-stats");
        for (const [n, name] of [[ws.library.length, "Saved prompts"], [ws.profiles.length, "Profiles"], [ws.references.length, "References"]]) { const s = E("div", "ps-stat"); s.append(E("strong", "", String(n)), E("span", "", name)); stats.append(s); }
        const file = input("", () => {}, "file"); file.accept = "application/json,.json"; file.setAttribute("aria-label", "Restore workspace JSON");
        file.onchange = attempt(async () => {
            const picked = file.files[0]; if (!picked) return;
            if (picked.size > 48 * 1024 * 1024) throw new Error("Backup exceeds the 48 MiB workspace limit.");
            const backup = JSON.parse(await picked.text());
            if (!confirm("Replace this installation’s Prompt Studio workspace with this backup? Export a backup first if you want to keep the current library.")) { file.value = ""; return; }
            ws = await request("/restore", { revision: ws.revision, workspace: backup }); say("Workspace restored. Re-enter API keys on a different machine."); renderModal(); refreshInline();
        });
        modalBody.append(stats, E("p", "ps-muted", "This workspace is shared by users of this ComfyUI installation. API keys stay in a separate private server file and are never exported. Your current draft lives in the ComfyUI workflow; save it to the library to include it in a workspace backup."),
            row(button("Export workspace JSON", attempt(async () => { const data = await request("/backup"); const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })); const a = E("a"); a.href = url; a.download = `prompt-studio-${new Date().toISOString().slice(0, 10)}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 3000); say("Workspace backup exported."); }), "ps-primary"),
                button("Reload workspace", attempt(async () => { await load(); say("Workspace reloaded."); renderModal(); }))), field("Restore a workspace backup", file),
            heading("Reference storage", "Remove prepared references you no longer need. References used by library entries are protected."));
        const refs = E("div", "ps-storage-list");
        for (const ref of ws.references) refs.append(row(E("span", "", ref.name), E("small", "ps-muted", `${Math.round(ref.data.length * 0.75 / 1024)} KB`), button("Delete reference", attempt(async () => { if (settings.references.includes(ref.id)) throw new Error("Remove this reference from the current draft first."); if (confirm("Delete this reference? Other unsaved workflow drafts may still use it.")) { await mutate("references", null, ref.id); renderModal(); } }), "ps-quiet")));
        modalBody.append(refs);
    }
    load().catch(e => say(e.message, true));
    refreshStatus();
    return { element: root, content, refresh() { settings = readSettings(); refreshInline(); renderModal(); },
        connectionsChanged() { generationRevision++; refreshReferences(); renderModal(); },
        destroy() { dialog?.close(); controller?.abort(); if (job && !connectedJob) request("/cancel", { job_id: job }).catch(() => {}); }, open };
}
