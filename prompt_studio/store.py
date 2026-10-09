"""Small, atomic workspace store. Credentials never enter the public workspace."""
import copy
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import urlsplit
import uuid

MAX_WORKSPACE_BYTES = 48 * 1024 * 1024
LOCK = threading.RLock()

PROFILES = [
    {"id": "image", "name": "Image · general", "kind": "image", "instructions": "Write a clear image prompt: subject, composition, environment, lighting, materials, color and style. Describe visible details. Avoid filler and invented technical settings.", "negative": ""},
    {"id": "flux", "name": "FLUX / Z-Image / Krea", "kind": "image", "instructions": "Write natural-language image direction in coherent sentences. Put the subject and composition first, then lighting, texture and style. Quote any exact text to render. Do not use weighted tags or a negative prompt unless requested.", "negative": ""},
    {"id": "sdxl", "name": "SDXL", "kind": "image", "instructions": "Write a concise, concrete image prompt with subject, framing, lighting, visual style and key details. A compact descriptive phrase list is appropriate. Put exclusions in the negative prompt.", "negative": ""},
    {"id": "video", "name": "Video · general", "kind": "video", "instructions": "Describe a single coherent shot: starting composition, subject action over time, camera movement, light and atmosphere. Keep identities and objects consistent. Make motion achievable within the requested duration.", "negative": ""},
    {"id": "h3", "name": "MiniMax H3", "kind": "video", "instructions": "Write a natural-language video prompt with a clear scene, subject action, camera movement, timing and visual detail. If sound is requested, describe the soundscape and quote dialogue with its speaker. Do not invent special tokens or model parameters.", "negative": ""},
    {"id": "wan", "name": "Wan / LTX", "kind": "video", "instructions": "Describe the scene and subjects, then the action and camera movement in temporal order. Use concrete natural language, with stable lighting and identity. For image-to-video, focus on motion while preserving the reference composition. Include sound only if requested.", "negative": ""},
    {"id": "audio", "name": "Audio · sound design", "kind": "audio", "instructions": "Describe sound sources, environment, texture, timing, dynamics, spatial placement and transitions. Use audible details rather than visual metaphors. Respect the requested duration. Avoid unwanted speech or music unless requested.", "negative": ""},
    {"id": "music", "name": "Audio · music", "kind": "audio", "instructions": "Describe musical genre, mood, instrumentation, rhythm, tempo, arrangement and production character. Specify vocals or instrumental as requested. Do not invent lyrics unless asked.", "negative": ""},
]


def defaults():
    return {"format": "apple-silicon-prompt-studio", "version": 1, "revision": 0,
            "profiles": copy.deepcopy(PROFILES), "connections": [
                {"id": "omlx", "name": "oMLX", "provider": "omlx", "base_url": "http://localhost:8000/v1", "model": "", "key_env": ""},
                {"id": "lmstudio", "name": "LM Studio", "provider": "lmstudio", "base_url": "http://localhost:1234/v1", "model": "", "key_env": ""},
                {"id": "ollama", "name": "Ollama", "provider": "ollama", "base_url": "http://localhost:11434/v1", "model": "", "key_env": ""},
            ], "library": [], "references": []}


def endpoint(value):
    value = str(value).strip().rstrip("/")
    u = urlsplit(value)
    if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ValueError("Use an http(s) API base URL without credentials, query parameters or fragments.")
    # Accept the commonly pasted root URL as well as a proxy's custom API path.
    return value + "/v1" if not u.path else value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", value):
        raise ValueError("Invalid workspace item ID.")
    return value


def validate_workspace(data):
    if not isinstance(data, dict) or data.get("format") != "apple-silicon-prompt-studio" or data.get("version") != 1:
        raise ValueError("This is not a supported Prompt Studio backup (version 1).")
    clean = {k: copy.deepcopy(data.get(k, [])) for k in ("profiles", "connections", "library", "references")}
    clean.update(format=data["format"], version=1, revision=int(data.get("revision", 0)))
    for section, items in list(clean.items())[:4]:
        if not isinstance(items, list):
            raise ValueError(f"Invalid {section} in workspace.")
        ids = set()
        for item in items:
            if not isinstance(item, dict) or identifier(item.get("id")) in ids:
                raise ValueError(f"Duplicate or invalid ID in {section}.")
            ids.add(item["id"])
            if section in ("profiles", "connections", "references") and not isinstance(item.get("name"), str):
                raise ValueError(f"Missing name in {section}.")
            if section == "profiles":
                if item.get("kind") not in ("image", "video", "audio"):
                    raise ValueError("Profile type must be image, video or audio.")
                if not all(isinstance(item.get(k, ""), str) for k in ("instructions", "negative")):
                    raise ValueError("Profile instructions and negative prompts must be text.")
            elif section == "connections":
                item["base_url"] = endpoint(item.get("base_url", ""))
                if not all(isinstance(item.get(k, ""), str) for k in ("model", "key_env")):
                    raise ValueError("Connection model and environment variable must be text.")
                models = item.get("models", [])
                if not isinstance(models, list) or any(not isinstance(m, str) or not m for m in models):
                    raise ValueError("Cached models must be a list of model IDs.")
                if item.get("key_env") and not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", item["key_env"]):
                    raise ValueError("Invalid API key environment variable name.")
                for key in ("api_key", "key", "has_key", "authorization"):
                    item.pop(key, None)
            elif section == "library":
                if not isinstance(item.get("prompt"), str) or not isinstance(item.get("settings", {}), dict):
                    raise ValueError("Invalid library entry.")
            elif section == "references":
                # Full media validation is shared by uploads and restore.
                from .media import validate_reference
                validate_reference(item)
    if len(json.dumps(clean).encode()) > MAX_WORKSPACE_BYTES:
        raise ValueError("Workspace exceeds 48 MiB. Remove unused references or library entries first.")
    return clean


class Workspace:
    def __init__(self, root):
        self.root = Path(root)

    def _read(self, name, fallback):
        file = self.root / name
        return json.loads(file.read_text()) if file.exists() else fallback

    def _write(self, name, data):
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self.root, prefix=".studio-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(data, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.root / name)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def read(self, public=True):
        with LOCK:
            data = self._read("workspace.json", defaults())
            if public:
                keys = self._read("credentials.json", {})
                for c in data["connections"]:
                    c["has_key"] = bool(keys.get(c["id"]))
            return data

    def connection(self, connection_id):
        with LOCK:
            c = next((c for c in self.read(False)["connections"] if c["id"] == connection_id), None)
            if c is None:
                raise ValueError("Choose a saved connection in Prompt Studio → Connections.")
            key = self._read("credentials.json", {}).get(c["id"], "")
            if c.get("key_env"):
                key = os.environ.get(c["key_env"], "")
                if not key:
                    raise ValueError(f"Set {c['key_env']} before starting ComfyUI, or use a saved API key.")
            return c, key

    def update(self, body):
        with LOCK:
            data = self.read(False)
            if body.get("revision") != data["revision"]:
                raise ValueError("Workspace changed in another window. Reload the workspace and retry.")
            section = body.get("section")
            if section not in ("profiles", "connections", "library", "references"):
                raise ValueError("Unknown workspace section.")
            items = data[section]
            if body.get("delete"):
                item_id = identifier(body["delete"])
                if section == "references":
                    for entry in data["library"]:
                        if item_id in entry.get("settings", {}).get("references", []):
                            raise ValueError("This reference is used by a saved prompt. Remove that library entry first.")
                data[section] = [i for i in items if i["id"] != item_id]
            else:
                item = copy.deepcopy(body["item"])
                item.setdefault("id", uuid.uuid4().hex)
                identifier(item["id"])
                position = next((n for n, old in enumerate(items) if old["id"] == item["id"]), None)
                if section == "connections" and position is not None and items[position]["base_url"] != endpoint(item["base_url"]):
                    item.pop("models", None)
                if position is None:
                    items.append(item)
                else:
                    data[section] = [item if n == position else old for n, old in enumerate(items)]
            data["revision"] += 1
            clean = validate_workspace(data)
            if section == "connections":
                keys = self._read("credentials.json", {})
                if body.get("delete"):
                    keys.pop(body["delete"], None)
                else:
                    old = next((c for c in items if c["id"] == item["id"]), None)
                    if old and old["base_url"] != endpoint(item["base_url"]):
                        keys.pop(item["id"], None)
                    if "api_key" in body:
                        if not isinstance(body["api_key"], str):
                            raise ValueError("API key must be text.")
                        keys[item["id"]] = body["api_key"]
                self._write("credentials.json", keys)
            self._write("workspace.json", clean)
            return self.read()

    def cache_models(self, connection_id, base_url, models):
        with LOCK:
            data = self.read(False)
            connection = next((c for c in data["connections"] if c["id"] == connection_id), None)
            if connection is None or connection["base_url"] != base_url:
                raise ValueError("Connection changed during discovery. Refresh the model list again.")
            connection["models"] = list(dict.fromkeys(models))
            return self.update({"revision": data["revision"], "section": "connections", "item": connection})

    def restore(self, backup, revision):
        # A restore never redirects an existing credential to a new endpoint.
        clean = validate_workspace(backup)
        with LOCK:
            old = self.read(False)
            if revision != old["revision"]:
                raise ValueError("Workspace changed. Reload before restoring.")
            old_endpoints = {c["id"]: c["base_url"] for c in old["connections"]}
            valid = {c["id"] for c in clean["connections"] if old_endpoints.get(c["id"]) == c["base_url"]}
            keys = {k: v for k, v in self._read("credentials.json", {}).items() if k in valid}
            clean["revision"] = old["revision"] + 1
            self._write("credentials.json", keys)
            self._write("workspace.json", clean)
            return self.read()
