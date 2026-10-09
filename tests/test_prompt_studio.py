import asyncio
import base64
import copy
import io
import json
import math
import stat
import sys
from types import SimpleNamespace
import uuid
import wave

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
import pytest
import torch

from prompt_studio import AppleSiliconPromptStudio, install_routes
from prompt_studio import client
from prompt_studio.media import image_reference, tensor_references, video_references, validate_reference
from prompt_studio.store import Workspace, defaults, endpoint, validate_workspace
import prompt_studio


def update(store, section, item, **extras):
    return store.update({"revision": store.read()["revision"], "section": section, "item": item, **extras})


def configured(tmp_path, **overrides):
    store = Workspace(tmp_path)
    connection = dict(defaults()["connections"][0], model="fixture-model", **overrides)
    update(store, "connections", connection)
    config = {"connection": connection["id"], "idea": "An observatory above the clouds", "profile": defaults()["profiles"][0]}
    return store, config


def test_credentials_do_not_enter_workflows_or_backups_and_endpoint_change_clears_them(tmp_path):
    store, _ = configured(tmp_path)
    c = store.read()["connections"][0]
    update(store, "connections", c, api_key="private-test-key")
    assert "private-test-key" not in json.dumps(store.read())
    assert "private-test-key" not in json.dumps(store.read(False))
    assert store.read()["connections"][0]["has_key"]
    assert stat.S_IMODE((tmp_path / "credentials.json").stat().st_mode) == 0o600
    assert store.connection(c["id"])[1] == "private-test-key"
    c["base_url"] = "http://localhost:1234/v1"
    update(store, "connections", c)
    assert store.connection(c["id"])[1] == ""


def test_restore_is_validated_before_writing_and_cannot_redirect_saved_keys(tmp_path):
    store, _ = configured(tmp_path)
    c = store.read()["connections"][0]
    update(store, "connections", c, api_key="secret")
    before = store.read(False)
    invalid = copy.deepcopy(before)
    invalid["profiles"][0]["id"] = "../elsewhere"
    with pytest.raises(ValueError):
        store.restore(invalid, before["revision"])
    assert before == store.read(False)
    changed = copy.deepcopy(before)
    changed["connections"][0]["base_url"] = "https://example.com/v1"
    changed["connections"][0]["api_key"] = "imported-key"
    store.restore(changed, before["revision"])
    assert store.connection(c["id"])[1] == ""
    with pytest.raises(ValueError, match="changed"):
        store.restore(before, before["revision"])


def test_model_discovery_cache_survives_reload_and_backup_but_not_endpoint_change(tmp_path):
    store, _ = configured(tmp_path / "a")
    connection, _ = store.connection("omlx")
    state = store.cache_models("omlx", connection["base_url"], ["vision-a", "text-b", "vision-a"])
    assert state["connections"][0]["models"] == ["vision-a", "text-b"]
    assert state["connections"][0]["model"] == "fixture-model"
    reloaded = Workspace(tmp_path / "a")
    restored = Workspace(tmp_path / "b")
    restored.restore(reloaded.read(False), 0)
    assert restored.connection("omlx")[0]["models"] == ["vision-a", "text-b"]
    changed = dict(state["connections"][0], base_url="http://localhost:1234/v1")
    update(reloaded, "connections", changed)
    assert "models" not in reloaded.connection("omlx")[0]
    with pytest.raises(ValueError, match="changed during discovery"):
        reloaded.cache_models("omlx", connection["base_url"], ["stale-model"])
    assert "models" not in reloaded.connection("omlx")[0]


def test_model_discovery_route_persists_only_successful_lists(tmp_path, monkeypatch):
    store, _ = configured(tmp_path)
    monkeypatch.setattr(prompt_studio, "workspace", lambda: store)
    async def run():
        response = {"data": [{"id": "vision-a"}, {"id": "vision-a"}, {"id": "text-b"}, {"id": None}]}
        async def discovery(*args):
            return response
        monkeypatch.setattr(prompt_studio, "request_json", discovery)
        server = SimpleNamespace(routes=web.RouteTableDef())
        install_routes(server)
        app = web.Application(); app.add_routes(server.routes)
        async with TestClient(TestServer(app)) as http:
            result = await http.post("/asfp8/prompt-studio/models", json={"connection": "omlx"})
            assert result.status == 200
            data = await result.json()
            assert data["models"] == ["vision-a", "text-b"]
            assert data["workspace"]["connections"][0]["models"] == data["models"]
            response = {"error": "discovery unavailable"}
            failed = await http.post("/asfp8/prompt-studio/models", json={"connection": "omlx"})
            assert failed.status == 400
            assert store.connection("omlx")[0]["models"] == data["models"]
    asyncio.run(run())


@pytest.mark.parametrize("url", ["file:///tmp/foo", "http://user:pass@localhost/v1", "https://localhost/v1?key=x", "http://localhost/#x", "localhost:8000"])
def test_invalid_endpoints(url):
    with pytest.raises(ValueError):
        endpoint(url)


def test_endpoint_and_workspace_roundtrip(tmp_path):
    assert endpoint(" http://localhost:8000/ ") == "http://localhost:8000/v1"
    assert endpoint("https://host/proxy/api/") == "https://host/proxy/api"
    a = Workspace(tmp_path / "a")
    ref = image_reference(Image.new("RGB", (12, 12), "green"), "reference")
    update(a, "references", ref)
    entry = {"id": "saved", "prompt": "A green garden", "negative_prompt": "noise", "favorite": True, "settings": {"references": [ref["id"]]}}
    update(a, "library", entry)
    with pytest.raises(ValueError, match="used by a saved prompt"):
        a.update({"revision": a.read()["revision"], "section": "references", "delete": ref["id"]})
    b = Workspace(tmp_path / "b")
    b.restore(a.read(False), 0)
    assert b.read(False)["references"] == a.read(False)["references"]
    assert b.read(False)["library"] == a.read(False)["library"]


def completion(content, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}


def test_completion_formats_and_explicit_negative_preserved():
    assert client.parse_completion(completion('<think>reasoning</think>```json\n{"prompt":"Forest","negative_prompt":"fog"}\n```'), "noise") == {"prompt": "Forest", "negative_prompt": "noise, fog"}
    assert client.parse_completion(completion("A forest."), "noise")["negative_prompt"] == "noise"
    assert client.parse_completion(completion([{"type": "text", "text": '{"prompt":"Hi"}'}]))["prompt"] == "Hi"
    for value in [completion("{broken"), completion('{"prompt":""}'), completion("<think>unfinished"), completion(None), completion("half", "length")]:
        with pytest.raises(ValueError):
            client.parse_completion(value)


def test_reference_preparation_is_bounded_and_rejects_invalid_data():
    images = torch.linspace(0, 1, 11 * 10 * 10 * 3).reshape(11, 10, 10, 3)
    audio = {"waveform": torch.zeros(1, 2, 32 * 8000), "sample_rate": 8000}
    refs = tensor_references(images, audio, frame_count=3, fps=10)
    assert [r["timestamp"] for r in refs[:3]] == [0.0, 0.5, 1.0]
    for ref in refs:
        validate_reference(ref)
    with wave.open(io.BytesIO(base64.b64decode(refs[-1]["data"]))) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getnframes()) == (24000, 1, 720000)
    large = image_reference(Image.new("RGB", (2200, 1600)), "large")
    with Image.open(io.BytesIO(base64.b64decode(large["data"]))) as image:
        assert max(image.size) == 1024
    with pytest.raises(ValueError):
        validate_reference({"kind": "image", "data": "not-base64"})


def test_reference_capabilities_and_audio_target_are_independent():
    ref = image_reference(Image.new("RGB", (10, 10)), "frame", "frame", 2.0)
    settings = {"idea": "Describe", "profile": {"kind": "audio"}}
    assert client.messages(settings, {}, [])[1]["content"]
    # Legacy unchecked capability flags must not silently exclude or block media.
    msg = client.messages(settings, {"vision": False}, [ref])
    assert msg[1]["content"][-1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "2.00s" in msg[1]["content"][-2]["text"]
    with pytest.raises(ValueError, match="at most 8"):
        client.messages(settings, {"vision": True}, [ref] * 9)


def test_small_image_batches_send_every_image_and_video_sampling_is_explicit():
    images = torch.rand(5, 20, 24, 3)
    refs = tensor_references(images, frame_count=2)
    assert len(refs) == 5
    assert all(r["kind"] == "image" and r["timestamp"] is None for r in refs)
    sampled = tensor_references(images, frame_count=2, fps=2, batch_mode="video")
    assert [r["timestamp"] for r in sampled] == [0, 2]
    large = tensor_references(torch.zeros(1, 1100, 1500, 3))[0]
    with Image.open(io.BytesIO(base64.b64decode(large["data"]))) as image:
        assert image.size == (1024, 751)


def test_native_video_samples_short_trimmed_views_without_full_decode(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy_api.latest._input_impl.video_types", SimpleNamespace(
        VideoFromComponents=type("VideoFromComponents", (), {}), VideoFromList=type("VideoFromList", (), {})))
    windows = []

    class Video:
        def get_duration(self): return 10.0
        def get_frame_rate(self): return 25
        def get_components(self): raise AssertionError("Must not decode the full video")
        def as_trimmed(self, start, duration, strict_duration):
            windows.append((start, duration, strict_duration))
            return SimpleNamespace(get_components=lambda: SimpleNamespace(images=torch.zeros(1, 24, 32, 3)))

    refs = video_references(Video(), 4)
    assert len(refs) == 4
    assert all(duration == 1 / 25 and not strict for _, duration, strict in windows)
    assert [r["timestamp"] for r in refs] == [1.25, 3.75, 6.25, 8.75]


def test_connected_generation_sends_text_and_all_images_in_one_http_request(tmp_path, monkeypatch):
    async def run():
        received = []
        async def chat(request):
            received.append(await request.json())
            return web.json_response(completion('{"prompt":"Five colored objects","negative_prompt":""}'))
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        async with TestServer(app) as server:
            store, config = configured(tmp_path, base_url=str(server.make_url("/v1")))
            monkeypatch.setattr(prompt_studio, "workspace", lambda: store)
            mm = SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None)
            monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
            monkeypatch.setattr(sys.modules["comfy"], "model_management", mm, raising=False)
            config["_studio_job"] = "test-job"
            result = await AppleSiliconPromptStudio().execute("generate_on_queue", json.dumps(config), "", "",
                                                            idea="Use all five references", images=torch.rand(5, 24, 32, 3))
            assert result["result"][0] == "Five colored objects"
            assert result["ui"]["studio_job"] == ["test-job"]
            assert result["ui"]["studio_result"][0]["visual_count"] == 5
            parts = received[0]["messages"][-1]["content"]
            assert json.loads(parts[0]["text"])["idea"] == "Use all five references"
            assert len([p for p in parts if p["type"] == "image_url"]) == 5
    # Provide the parent module when tests run without a ComfyUI installation.
    if "comfy" not in sys.modules:
        monkeypatch.setitem(sys.modules, "comfy", SimpleNamespace())
    asyncio.run(run())


def test_real_http_transport_and_error_redaction(tmp_path):
    async def run():
        received = []
        async def chat(request):
            data = await request.json()
            received.append((dict(request.headers), data))
            return web.json_response(completion('{"prompt":"Moonlit observatory","negative_prompt":""}'))
        async def bad(request):
            return web.Response(status=400, text="bad private-test-key")
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        app.router.add_get("/v1/bad", bad)
        async with TestServer(app) as server:
            store, settings = configured(tmp_path, base_url=str(server.make_url("/v1")))
            c = next(c for c in store.read()["connections"] if c["id"] == "omlx")
            update(store, "connections", c, api_key="private-test-key")
            settings.update(seed=42, negative="watermark")
            result = await client.generate(store, settings)
            assert result["prompt"] == "Moonlit observatory"
            assert result["negative_prompt"] == "watermark"
            assert received[0][0]["Authorization"] == "Bearer private-test-key"
            assert received[0][1]["seed"] == 42
            assert "response_format" not in received[0][1]
            with pytest.raises(ValueError) as error:
                await client.request_json(c, "private-test-key", "/bad")
            assert "private-test-key" not in str(error.value)
            assert "[redacted]" in str(error.value)
    asyncio.run(run())


def test_saved_node_does_not_call_llm_and_keeps_exact_text():
    node = AppleSiliconPromptStudio()
    result = asyncio.run(node.execute("use_saved", "not needed", "  Exact\ntext  ", "negative"))
    assert result["result"] == ("  Exact\ntext  ", "negative")
    assert node.IS_CHANGED("use_saved") == "saved"
    assert math.isnan(node.IS_CHANGED("generate_on_queue"))
    with pytest.raises(ValueError, match="generate or type"):
        asyncio.run(node.execute("use_saved", "{}", " ", ""))


def test_missing_reference_and_bad_backup_are_actionable(tmp_path):
    store, settings = configured(tmp_path)
    settings["references"] = ["missing"]
    with pytest.raises(ValueError, match="missing"):
        asyncio.run(client.generate(store, settings))
    with pytest.raises(ValueError, match="supported"):
        validate_workspace({"version": 400})


def test_routes_restore_and_cancel_are_scoped_and_release_requests(tmp_path, monkeypatch):
    store, _ = configured(tmp_path)
    monkeypatch.setattr(prompt_studio, "workspace", lambda: store)
    async def run():
        entered = asyncio.Event()
        async def pending(*args):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(prompt_studio, "generate", pending)
        server = SimpleNamespace(routes=web.RouteTableDef(), user_manager=SimpleNamespace(get_request_user_id=lambda r: r.headers.get("comfy-user", "default")))
        install_routes(server)
        install_routes(server)  # Idempotent registration.
        app = web.Application(); app.add_routes(server.routes)
        async with TestClient(TestServer(app)) as http:
            backup = await (await http.get("/asfp8/prompt-studio/backup")).json()
            response = await http.post("/asfp8/prompt-studio/restore", json={"workspace": backup, "revision": backup["revision"]})
            assert response.status == 200
            response = await http.post("/asfp8/prompt-studio/restore", json={"workspace": {"version": 2}, "revision": backup["revision"]})
            assert response.status == 400
            job = str(uuid.uuid4())
            request = asyncio.create_task(http.post("/asfp8/prompt-studio/generate", json={"job_id": job, "settings": {}}))
            await asyncio.wait_for(entered.wait(), 2)
            other = await http.post("/asfp8/prompt-studio/cancel", json={"job_id": job}, headers={"comfy-user": "other"})
            assert (await other.json())["cancelled"] is False
            own = await http.post("/asfp8/prompt-studio/cancel", json={"job_id": job})
            assert (await own.json())["cancelled"] is True
            response = await asyncio.wait_for(request, 2)
            assert response.status == 409
            assert "cancelled" in (await response.json())["error"]
    asyncio.run(run())
