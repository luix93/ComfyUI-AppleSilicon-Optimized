"""Prompt Studio: local LLM prompt generation, workspace routes and STRING outputs."""
import asyncio
import json
import math
from pathlib import Path
import uuid

from aiohttp import web

from .client import generate, request_json, settings_from_json
from .media import tensor_references, video_references
from .store import MAX_WORKSPACE_BYTES, Workspace


def workspace():
    import folder_paths
    # Shared across workflows in this ComfyUI installation, like model connections.
    return Workspace(Path(folder_paths.get_user_directory()) / "__apple_silicon_prompt_studio")


class AppleSiliconPromptStudio:
    CATEGORY = "Apple Silicon/Prompting"
    DESCRIPTION = "Create and organize image, video and audio prompts using a local OpenAI-compatible LLM. Open Studio to configure connections, references and profiles. Outputs editable text only."
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt")
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["use_saved", "generate_on_queue"], {"default": "use_saved"}),
            "settings": ("STRING", {"default": '{"version":1}', "multiline": True}),
            "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
        }, "optional": {
            "idea": ("STRING", {"forceInput": True, "tooltip": "Brief sent to the LLM when you press Generate or generate on queue. Overrides the Studio brief."}),
            "images": ("IMAGE", {"tooltip": "Sent with your brief when you press Generate. Up to 8 images are all included; longer video batches are sampled."}),
            "audio": ("AUDIO", {"tooltip": "First audio item, up to its first 30 seconds. Requires an audio-capable LLM."}),
            "video": ("VIDEO", {"tooltip": "Video reference. Generate samples frames across the clip and sends them with your brief to a vision LLM."}),
        }}

    @classmethod
    def IS_CHANGED(cls, mode, **kwargs):
        return float("nan") if mode == "generate_on_queue" else "saved"

    async def execute(self, mode, settings, prompt, negative_prompt, idea=None, images=None, audio=None, video=None):
        job = ""
        if mode == "use_saved":
            if not prompt.strip():
                raise ValueError("Open Prompt Studio to generate or type a prompt, or choose Generate on queue.")
            result = {"prompt": prompt, "negative_prompt": negative_prompt}
        elif mode == "generate_on_queue":
            config = settings_from_json(settings)
            job = config.get("_studio_job", "")
            if idea is not None:
                config["idea"] = idea
            frames = int(config.get("frame_count", 4))
            fps = float(config.get("reference_fps", 24))
            if not 1 <= frames <= 8 or not math.isfinite(fps) or fps <= 0:
                raise ValueError("Select 1–8 reference frames and a positive reference FPS.")
            refs = tensor_references(images, audio, frames, fps, config.get("image_batch_mode", "auto"))
            if video is not None:
                refs.extend(video_references(video, frames))
            # Poll cancellation while the external server is working.
            import comfy.model_management as mm
            task = asyncio.create_task(generate(workspace(), config, refs))
            try:
                while not task.done():
                    await asyncio.wait({task}, timeout=0.5)
                    mm.throw_exception_if_processing_interrupted()
                result = await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        else:
            raise ValueError("Unknown Prompt Studio execution mode.")
        return {"ui": {"prompt": [result["prompt"]], "negative_prompt": [result["negative_prompt"]],
                       "studio_result": [result], "studio_job": [job]},
                "result": (result["prompt"], result["negative_prompt"])}


NODE_CLASS_MAPPINGS = {"AppleSiliconPromptStudio": AppleSiliconPromptStudio}
NODE_DISPLAY_NAME_MAPPINGS = {"AppleSiliconPromptStudio": "Apple Silicon Prompt Studio"}


def install_routes(server):
    if getattr(server, "_asfp8_prompt_studio_routes", False):
        return
    server._asfp8_prompt_studio_routes = True
    routes = server.routes
    jobs = {}

    async def body(request):
        raw = bytearray()
        async for chunk in request.content.iter_chunked(65536):
            raw.extend(chunk)
            if len(raw) > MAX_WORKSPACE_BYTES:
                raise ValueError("Request exceeds 48 MiB. Use smaller references.")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def guarded(handler):
        async def wrapped(request):
            try:
                return await handler(request)
            except (ValueError, KeyError, TypeError) as e:
                return web.json_response({"error": str(e)}, status=400)
        return wrapped

    @routes.get("/asfp8/prompt-studio/workspace")
    @guarded
    async def get_workspace(request):
        return web.json_response(workspace().read())

    @routes.post("/asfp8/prompt-studio/workspace")
    @guarded
    async def update_workspace(request):
        return web.json_response(workspace().update(await body(request)))

    @routes.get("/asfp8/prompt-studio/backup")
    @guarded
    async def backup(request):
        return web.json_response(workspace().read(False), headers={"Content-Disposition": 'attachment; filename="prompt-studio-backup.json"'})

    @routes.post("/asfp8/prompt-studio/restore")
    @guarded
    async def restore(request):
        data = await body(request)
        return web.json_response(workspace().restore(data["workspace"], data["revision"]))

    @routes.post("/asfp8/prompt-studio/models")
    @guarded
    async def models(request):
        data = await body(request)
        store = workspace()
        connection, key = store.connection(data.get("connection"))
        result = await request_json(connection, key, "/models")
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise ValueError("Server returned an invalid model list.")
        models = list(dict.fromkeys(m["id"] for m in result["data"] if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]))
        saved = store.cache_models(connection["id"], connection["base_url"], models)
        return web.json_response({"models": models, "workspace": saved})

    @routes.post("/asfp8/prompt-studio/generate")
    @guarded
    async def preview(request):
        data = await body(request)
        job_id = str(uuid.UUID(data["job_id"]))
        owner = server.user_manager.get_request_user_id(request)
        identity = (owner, job_id)
        if identity in jobs:
            raise ValueError("This generation is already running.")
        task = asyncio.create_task(generate(workspace(), settings_from_json(data["settings"])))
        jobs[identity] = task
        try:
            return web.json_response(await task)
        except asyncio.CancelledError:
            return web.json_response({"error": "Generation cancelled."}, status=409)
        finally:
            jobs.pop(identity, None)

    @routes.post("/asfp8/prompt-studio/cancel")
    @guarded
    async def cancel(request):
        data = await body(request)
        owner = server.user_manager.get_request_user_id(request)
        task = jobs.get((owner, str(uuid.UUID(data["job_id"]))))
        if task:
            task.cancel()
        return web.json_response({"cancelled": task is not None})
