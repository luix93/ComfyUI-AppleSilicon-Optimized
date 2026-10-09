const CACHE_NAMES = {
    disabled: "Off — exact", motion_cache: "MotionCache", spectrum_protected: "Spectrum · protected",
    spectrum_fast: "Spectrum · fast", first_block_cache: "FirstBlockCache",
    first_block_cache_per_step: "FirstBlockCache · per step", oracle: "Cache oracle · diagnostic",
    block_oracle: "Block oracle · diagnostic",
};
const CACHE_HELP = {
    disabled: "No residual reuse or forecasting.",
    motion_cache: "Motion-aware reuse. Best content preservation in our tests.",
    spectrum_protected: "Forecasts between actual evaluations; protects first 4 / last 2.",
    spectrum_fast: "More forecasting. Faster, with larger appearance changes.",
    first_block_cache: "Reuses block residuals when the first block changes little.",
    first_block_cache_per_step: "A more aggressive first-block reuse policy.",
    oracle: "Measures cache error without skipping model work.",
    block_oracle: "Measures block error without skipping model work.",
};
const ATTENTION_NAMES = {bf16: "SOL · BF16", int8: "SOL · INT8 (M5)", dense_int8: "Dense · INT8 (M5)", int8_hybrid: "SOL + dense INT8 (M5)",
    veda_metal: "Veda · Metal (experimental)", veda_mlx: "Veda · MLX (experimental)"};
const FIELDS = {
    cache_reuse_threshold: "Reuse threshold", cache_start_percent: "Start fraction", cache_end_percent: "End fraction",
    cache_max_consecutive_skips: "Maximum skips", cache_audio_weight: "Audio veto weight", cache_verbose: "Log cache activity",
    motion_reuse_threshold: "Reuse threshold", motion_strength: "Motion strength", motion_warmup_steps: "Warmup evaluations",
    motion_max_consecutive_skips: "Maximum skips", motion_start_percent: "Start fraction", motion_end_percent: "End fraction",
    spectrum_blend_weight: "Video blend weight",
    sparse_backend: "Attention method", sparse_mode: "Mode", sparse_tau: "Routing tau", sparse_window_mode: "Schedule",
    sparse_start_percent: "Start fraction", sparse_end_percent: "End fraction", sparse_dense_leading_blocks: "Dense leading blocks",
    sparse_dense_layers: "Protected dense layers", sparse_min_sequence_tokens: "Minimum tokens", sparse_verbose: "Log sparse activity",
    sparse_memory_budget_gib: "Temporary memory (GiB)", sparse_topk_ratio: "Top-k video fraction",
    veda_predictor: "Veda predictor", veda_sparsity: "Video sparsity (%)",
    veda_reference_sparsity: "Reference sparsity (%)", veda_dense_steps: "Additional dense steps",
    token_reduction_scale: "Update scale", token_early_evaluations: "Early evaluations", token_detail_restore_block: "Later restore block",
    token_coarse_restore_block: "Early restore block", token_start_percent: "Start step fraction", token_end_percent: "End step fraction", token_verbose: "Log token activity",
};
function element(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text) el.textContent = text;
    return el;
}
function options(select, values) {
    for (const [value, label] of values) {
        const option = element("option", "", label); option.value = value; select.append(option);
    }
}
export function buildH3Panel(controls, presets, specs, properties, resized) {
    const root = element("div", "asfp8-h3");
    const content = element("div", "asfp8-h3-content"); root.append(content);
    // Keep typing/selection inside the controls out of graph shortcuts/dragging.
    for (const event of ["pointerdown", "dblclick"]) root.addEventListener(event, e => e.stopPropagation());
    root.addEventListener("keydown", e => {
        // Let ComfyUI save while an input is focused; keep editing shortcuts local.
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") return;
        e.stopPropagation();
    });
    const header = element("div", "h3-top");
    header.append(element("span", "h3-eyebrow", "H3 ACCELERATION"));
    const badge = element("span", "h3-badge"); header.append(badge); content.append(header);
    const presetRow = element("label", "h3-preset");
    presetRow.append(element("span", "", "Starting point"));
    const preset = element("select"); preset.setAttribute("aria-label", "Starting point");
    options(preset, [["Custom combination", "Custom combination"], ...presets.map(p => [p.label,p.label])]);
    presetRow.append(preset); content.append(presetRow);
    const intro = element("p", "h3-intro", "Combine a cache method with the switches below."); content.append(intro);
    const cards = element("div", "h3-cards"); content.append(cards);
    const inputs = new Map();
    const details = new Map();
    const groups = {};
    function changed(name, value) {
        controls.set(name, value); controls.changed(); refresh();
    }
    function card(name, title, description, toggleName) {
        const box = element("section", "h3-card"); box.dataset.section = name;
        const top = element("div", "h3-card-heading");
        top.append(element("h4", "", title));
        if (toggleName) {
            const toggle = element("input", "h3-switch"); toggle.type = "checkbox";
            toggle.setAttribute("role", "switch"); toggle.setAttribute("aria-label", title);
            toggle.addEventListener("change", () => changed(toggleName, toggle.checked));
            inputs.set(toggleName, toggle); top.append(toggle);
        }
        box.append(top);
        const help = element("p", "h3-help", description); box.append(help);
        cards.append(box); return {box, help, top};
    }
    function tune(parent, name, names) {
        const panel = element("details", "h3-tune");
        panel.open = Boolean(properties()[`asfp8_h3_tune_${name}`]);
        panel.append(element("summary", "", "Settings"));
        const grid = element("div", "h3-fields"); panel.append(grid); parent.append(panel);
        details.set(name,panel);
        panel.addEventListener("toggle", () => {
            properties()[`asfp8_h3_tune_${name}`] = panel.open;
            resized();
        });
        for (const key of names) {
            const declaration = specs[key]; if (!declaration) continue;
            const [type, info = {}] = declaration;
            const label = element("label", "h3-field"); label.dataset.field = key;
            label.append(element("span", "", FIELDS[key] || key));
            let input;
            if (Array.isArray(type)) {
                input = element("select"); options(input, type.map(v=>[v,key === "sparse_backend" ? ATTENTION_NAMES[v] || v : v]));
            } else {
                input = element("input");
                input.type = type === "BOOLEAN" ? "checkbox" : ["INT","FLOAT"].includes(type) ? "number" : "text";
                if (input.type === "number") {
                    input.min = info.min ?? ""; input.max = info.max ?? ""; input.step = info.step ?? (type === "INT" ? 1 : "any");
                }
                if (input.type === "checkbox") label.classList.add("h3-check-field");
            }
            input.setAttribute("aria-label", FIELDS[key] || key); input.title = info.tooltip || "";
            // Commit valid typing immediately: saving or collapsing a section
            // must not depend on the browser delivering a later blur event.
            if (input.type === "number" || input.type === "text") input.addEventListener("input", () => {
                if (input.type === "number" && (!input.checkValidity() || !Number.isFinite(input.valueAsNumber))) return;
                controls.set(key, input.type === "number" ? input.valueAsNumber : input.value);
                controls.changed();
                preset.value = controls.selectedPreset();
            });
            input.addEventListener("change", () => {
                if (input.type === "number" && (!input.checkValidity() || !Number.isFinite(input.valueAsNumber))) {
                    input.reportValidity(); input.value = controls.get(key); return;
                }
                changed(key, input.type === "checkbox" ? input.checked : input.type === "number" ? input.valueAsNumber : input.value);
            });
            inputs.set(key,input); label.append(input);
            if (key === "sparse_memory_budget_gib") label.append(element("small", "h3-help", "0 = Auto · or set a GiB ceiling. Allocated only as needed."));
            grid.append(label);
        }
        return panel;
    }
    const cache = card("cache", "Temporal cache", "");
    const method = element("select", "h3-method"); method.setAttribute("aria-label", "Temporal cache");
    options(method,Object.entries(CACHE_NAMES));
    method.addEventListener("change",()=>changed("cache_mode",method.value));
    inputs.set("cache_mode",method); cache.box.insertBefore(method,cache.help);
    tune(cache.box,"cache",["cache_reuse_threshold","cache_start_percent","cache_end_percent","cache_max_consecutive_skips","cache_audio_weight",
        "motion_reuse_threshold","motion_strength","motion_warmup_steps","motion_max_consecutive_skips","motion_start_percent","motion_end_percent",
        "spectrum_blend_weight","cache_verbose"]);
    groups.sparse = card("sparse", "Attention acceleration", "Optional attention arithmetic or sparsity. Adds approximation.", "sparse_attention");
    tune(groups.sparse.box,"sparse",["sparse_backend","sparse_mode","sparse_topk_ratio","sparse_tau","sparse_window_mode","sparse_start_percent","sparse_end_percent",
        "sparse_dense_layers","sparse_dense_leading_blocks","sparse_min_sequence_tokens","sparse_memory_budget_gib","sparse_verbose",
        "veda_predictor","veda_sparsity","veda_reference_sparsity","veda_dense_steps"]);
    groups.tokens = card("tokens", "Token reduction", "Experimental · reduces spatial detail during computation.", "token_reduction");
    tune(groups.tokens.box,"tokens",["token_reduction_scale","token_early_evaluations","token_detail_restore_block","token_coarse_restore_block","token_start_percent","token_end_percent","token_verbose"]);
    const footer = element("div", "h3-footer");
    const status = element("p", "h3-status"); status.setAttribute("aria-live","polite");
    const note = element("p", "h3-note");
    footer.append(status,note);
    content.append(footer);
    preset.addEventListener("change",()=>{controls.preset(preset.value);controls.changed();refresh();});
    function refresh() {
        for (const [name,panel] of details) panel.open = Boolean(properties()[`asfp8_h3_tune_${name}`]);
        for (const [key,input] of inputs) {
            const value = controls.get(key);
            if (input.type === "checkbox") input.checked = Boolean(value); else input.value = value ?? "";
            input.disabled = controls.connected(key);
            if (input.disabled) input.title = "Controlled by a connected input";
        }
        preset.value = controls.selectedPreset();
        preset.disabled = controls.backend.some(w=>controls.connected(w.name));
        const mode = controls.get("cache_mode"); cache.help.textContent = CACHE_HELP[mode] || "Custom cache method.";
        const attention = controls.get("sparse_backend");
        const veda = attention?.startsWith("veda_");
        note.textContent = veda && controls.get("sparse_attention")
            ? "Experimental · preview predictor trained for 8-step Turbo. Other shapes use the nearest plan. Compare appearance, motion and audio."
            : "Built in · no extra nodes. Presets tested at 20 steps without Turbo LoRA. Combinations need a quality check.";
        groups.sparse.help.textContent = veda
            ? "Learned tile selection. Video sparsity controls skipped connections; 0% reference sparsity keeps references dense. Predictor required in models/veda."
            : attention === "dense_int8" ? "Every token is retained. INT8 QK/PV changes arithmetic; protected steps and layers use the exact backend."
            : attention === "int8_hybrid" ? "SOL in the active window, with dense INT8 in protected layers. Protected sampling steps stay exact."
            : "SOL attention on computed evaluations. Adds approximation.";
        const cacheActive = mode !== "disabled"; cache.box.dataset.active = String(cacheActive);
        for (const [key,input] of inputs) {
            const label = input.closest(".h3-field"); if (!label) continue;
            if (key.startsWith("motion_")) label.hidden = mode !== "motion_cache";
            else if (key.startsWith("spectrum_")) label.hidden = !mode?.startsWith("spectrum_");
            else if (key.startsWith("cache_") && key !== "cache_verbose") label.hidden = !(mode?.startsWith("first_block_cache") || ["oracle","block_oracle"].includes(mode));
            else if (key.startsWith("veda_")) label.hidden = !veda;
            else if (key === "sparse_topk_ratio") label.hidden = !["int8","int8_hybrid"].includes(attention);
            else if (key === "sparse_tau") label.hidden = Boolean(veda) || attention === "dense_int8" || (["int8","int8_hybrid"].includes(attention) && controls.get("sparse_topk_ratio") > 0);
            else if (key === "sparse_mode") label.hidden = Boolean(veda);
            else if (key === "sparse_memory_budget_gib") label.hidden = !["int8","int8_hybrid","dense_int8"].includes(attention) && !veda;
        }
        details.get("cache").hidden = !cacheActive;
        for (const [group,key] of [["sparse","sparse_attention"],["tokens","token_reduction"]]) {
            const active = Boolean(controls.get(key));
            groups[group].box.dataset.active = String(active); details.get(group).hidden = !active;
        }
        const active = [];
        if (cacheActive) active.push(CACHE_NAMES[mode] || mode);
        if (controls.get("sparse_attention")) active.push(veda ? "Veda" : attention === "dense_int8" ? "Dense INT8" : attention === "int8_hybrid" ? "SOL hybrid" : "SOL");
        if (controls.get("token_reduction")) active.push("token reduction");
        badge.textContent = active.length ? "APPROXIMATE" : "EXACT";
        badge.dataset.exact = String(!active.length);
        status.textContent = active.length ? active.join(" + ") : "Exact kernels active · all approximations off";
        if (preset.disabled) status.textContent += " · linked inputs";
        requestAnimationFrame(resized);
    }
    return {element:root,content,refresh};
}
