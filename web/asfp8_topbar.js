// ComfyUI-AppleSilicon-Optimized: "Free RAM" topbar button.
//
// One click = ComfyUI's own POST /free with both flags: unload all models,
// reset the execution cache, gc.collect(), and torch.mps.empty_cache() —
// which releases the resident MPS allocator pool (deliberately kept hot by
// patch #30 during normal operation) back to the OS. The prompt-worker
// processes the flags immediately even when idle (set_flag notifies its
// condition variable), so memory drops within a second or two.
//
// Placement: injected next to the queue action bar (top-right area). The
// same action is also registered as a proper command ("Free RAM (unload
// models + MPS cache)") so it stays reachable from the command palette even
// if a frontend update changes the topbar DOM and the injection misses.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CMD_ID = "asfp8.freeMemory";
const BTN_ID = "asfp8-free-ram-btn";

async function freeMemory() {
    const btn = document.getElementById(BTN_ID);
    if (btn) btn.classList.add("asfp8-busy");
    try {
        await api.fetchApi("/free", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ unload_models: true, free_memory: true }),
        });
        // The worker applies the flags asynchronously; give it a beat before
        // declaring success so the toast doesn't lie.
        await new Promise((r) => setTimeout(r, 1500));
        app.extensionManager?.toast?.add?.({
            severity: "success",
            summary: "RAM freed",
            detail: "Models unloaded, caches and MPS pool released.",
            life: 3000,
        });
    } catch (e) {
        app.extensionManager?.toast?.add?.({
            severity: "error",
            summary: "Free RAM failed",
            detail: String(e),
            life: 5000,
        });
    } finally {
        if (btn) btn.classList.remove("asfp8-busy");
    }
}

function makeButton() {
    const btn = document.createElement("button");
    btn.id = BTN_ID;
    btn.title = "Unload all models and release cached RAM (AppleSilicon-Optimized)";
    btn.textContent = "Free RAM";
    btn.style.cssText = [
        "display:inline-flex", "align-items:center", "gap:4px",
        "margin-left:6px", "padding:4px 10px", "border-radius:6px",
        "border:1px solid var(--border-color, #4a4a4a)",
        "background:var(--comfy-menu-bg, #202020)",
        "color:var(--fg-color, #ddd)", "font-size:12px", "cursor:pointer",
        "white-space:nowrap",
    ].join(";");
    btn.onmouseenter = () => (btn.style.filter = "brightness(1.35)");
    btn.onmouseleave = () => (btn.style.filter = "");
    btn.onclick = freeMemory;
    return btn;
}

// Candidate anchors, most specific first. The Desktop 2.0 topbar renders the
// queue controls inside an "actionbar" component; fall back to the generic
// menu containers used by older frontends.
const ANCHORS = [
    '[class*="actionbar"]',
    ".comfyui-menu-right",
    ".comfyui-menu",
];

function inject() {
    if (document.getElementById(BTN_ID)) return true;
    for (const sel of ANCHORS) {
        const host = document.querySelector(sel);
        if (host) {
            host.appendChild(makeButton());
            return true;
        }
    }
    return false;
}

app.registerExtension({
    name: "AppleSiliconFP8.FreeRAM",
    commands: [
        {
            id: CMD_ID,
            label: "Free RAM (unload models + MPS cache)",
            icon: "pi pi-eraser",
            function: freeMemory,
        },
    ],
    menuCommands: [
        { path: ["Workflow"], commands: [CMD_ID] },
    ],
    setup() {
        // The topbar mounts after extension setup in some frontend versions;
        // retry briefly, then keep a light observer as a safety net so the
        // button survives topbar re-renders (tab switches etc.).
        let tries = 0;
        const t = setInterval(() => {
            if (inject() || ++tries > 40) clearInterval(t);
        }, 250);
        const obs = new MutationObserver(() => {
            if (!document.getElementById(BTN_ID)) inject();
        });
        obs.observe(document.body, { childList: true, subtree: true });
    },
});
