"""Photo → configured vision model cleans it → Image to 3D from those bytes → checks against the cleaned image."""

import asyncio
import base64
import json
from types import SimpleNamespace

import anyio
import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.services import vision
from app.services.generation_prompt import meshy_prompt, meshy_texture_prompt
from app.services.meshy_service import MeshyError, MeshyService

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64


@pytest.mark.parametrize(("category", "kind", "expected"), [
    ("table", "", "table"),
    ("furniture", "double bed", "double bed"),
    ("wardrobe", "", "wardrobe"),
    ("furniture", "corner sofa", "corner sofa"),
    ("furniture", "bookcase", "bookcase"),
    ("tv", "", "TV unit"),
])
def test_the_cleaning_prompt_removes_loose_objects_for_any_type(category, kind, expected):
    p = meshy_prompt(category, kind or None)
    assert p.startswith(f"Create a clean, isolated product reference image of the {expected} shown.")
    assert f"Remove all objects placed on, inside, or around the {expected}" in p
    assert "Reconstruct any surfaces previously hidden by removed objects." in p
    assert "Flat wooden panels are perfectly flat." in p and "three-quarter product view" in p
    assert "Glass stays transparent and mirrors stay reflective." in p
    assert "Colors, patterns, and textures stay exactly as they are" in p
    assert "Do not add decorative elements" in p


def test_removal_is_not_asked_of_the_texture_prompt():
    assert "Remove" not in meshy_texture_prompt("furniture", kind="double bed")


def _service(sent: list[httpx.Request]) -> MeshyService:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(202, json={"result": "img-1"})

    settings = Settings(meshy_api_key=SecretStr("msy-secret"), meshy_image_model="nano-banana-2")
    api = httpx.AsyncClient(base_url=settings.meshy_base_url, transport=httpx.MockTransport(handler))
    return MeshyService(settings, api, api)


def test_image_to_image_payload_is_one_cleaned_view():
    sent: list[httpx.Request] = []
    anyio.run(lambda: _service(sent).create_image_to_image_task("clean it", image=PNG))
    assert sent[0].url.path == "/openapi/v1/image-to-image"
    body = json.loads(sent[0].content)
    assert body["ai_model"] == "nano-banana-2" and body["prompt"] == "clean it"
    assert body["generate_multi_view"] is False and body["remove_background"] is True
    assert len(body["reference_image_urls"]) == 1 and body["reference_image_urls"][0].startswith("data:image/png;base64,")


class FakeMeshy:
    def __init__(self, image_task=None, refuse_image=False):
        self.calls: dict = {}
        self.image_task = image_task
        self.refuse_image = refuse_image

    async def create_image_to_image_task(self, prompt, **kwargs):
        if self.refuse_image:
            raise MeshyError(400, "nope")
        self.calls["clean"] = {"prompt": prompt, **kwargs}
        return "img-1"

    async def get_image_to_image_task(self, _task_id):
        return self.image_task

    async def download_bytes(self, url):
        self.calls["downloaded"] = url
        return b"cleaned-png"

    async def create_image_to_3d_task(self, options, **kwargs):
        self.calls["mesh"] = kwargs
        return "mesh-1"


def _generate(tmp_path, monkeypatch, meshy: FakeMeshy, **settings):
    from fastapi.testclient import TestClient

    from app.api import vendor
    from app.main import app

    async def one_try(*_args):
        return 4

    async def nothing(*_args, **_kwargs):
        return None

    async def admin_chose_meshy(*_args, **_kwargs):
        return "meshy"

    seen: dict = {}

    async def clean_image(prompt, image, mime, settings, client, model):
        seen.update(prompt=prompt, image=image, model=model)
        return b"cleaned-png"

    async def chosen(*_args, **_kwargs):
        return "gemini-3.1-pro-preview"

    monkeypatch.setattr(vendor, "take_try", one_try)
    monkeypatch.setattr(vendor, "extract_product_details", nothing)
    monkeypatch.setattr(vendor, "analysed", nothing)
    monkeypatch.setattr(vendor.vision, "current_model", chosen)
    monkeypatch.setattr(vendor.vision, "clean_image", clean_image)
    monkeypatch.setattr(vendor.generator_setting, "current", admin_chose_meshy)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: meshy,
                                     vendor.get_settings: lambda: Settings(meshy_api_key="msy-test", gemini_api_key="gem-test", storage_dir=tmp_path, **settings)})
    try:
        with TestClient(app) as client:
            body = client.post("/api/vendor/generate", files={"image": ("t.png", PNG, "image/png")}, data={"category": "table", "kind": "desk"}).json()
            pending = dict(app.state.pending_cdn.get(body.get("task_id"), {}))
    finally:
        app.dependency_overrides.clear()
    return body, pending, seen


def test_generate_cleans_the_photo_first(tmp_path, monkeypatch):
    meshy = FakeMeshy()
    body, pending, seen = _generate(tmp_path, monkeypatch, meshy)
    assert pending["stage"] == "image" and pending["mesh_task_id"] is None and body["task_id"]
    assert seen["image"] == PNG and seen["model"] == "gemini-3.1-pro-preview"
    assert "product reference image of the table (desk)" in seen["prompt"]
    assert "clean" not in meshy.calls and "mesh" not in meshy.calls  # Meshy waits until the vision model returns the image
    assert pending["clean_prompt"] == seen["prompt"]


def test_a_refused_cleaning_step_does_not_start_3d(tmp_path, monkeypatch):
    from app.api import vendor

    async def quiet(*_args, **_kwargs):
        return None

    monkeypatch.setattr(vendor.bus, "publish", quiet)
    meshy = FakeMeshy(refuse_image=True)
    from fastapi.testclient import TestClient

    from app.main import app

    async def one_try(*_args):
        return 4

    async def nothing(*_args, **_kwargs):
        return None

    async def admin_chose_meshy(*_args, **_kwargs):
        return "meshy"

    async def chosen(*_args, **_kwargs):
        return "gemini-3.1-pro-preview"

    monkeypatch.setattr(vendor, "take_try", one_try)
    monkeypatch.setattr(vendor, "extract_product_details", nothing)
    monkeypatch.setattr(vendor, "analysed", nothing)
    monkeypatch.setattr(vendor.vision, "current_model", chosen)
    monkeypatch.setattr(vendor.generator_setting, "current", admin_chose_meshy)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: meshy,
                                     vendor.get_settings: lambda: Settings(meshy_api_key="msy-test", gemini_api_key="", openai_api_key="", plan_vision_model="gemini-3.1-pro-preview", storage_dir=tmp_path)})
    try:
        with TestClient(app) as client:
            res = client.post("/api/vendor/generate", files={"image": ("t.png", PNG, "image/png")}, data={"category": "table", "kind": "desk"})
    finally:
        app.dependency_overrides.clear()
    assert res.status_code == 400
    assert res.json()["detail"] == "Image to image failed, so the 3D model was not generated."
    assert "mesh" not in meshy.calls


def _request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(pending_cdn={})))


def _pending():
    return {"stage": "image", "mesh": {"textured": True, "texture_prompt": "t", "target_polycount": 30000}, "photo": PNG, "mime": "image/png"}


def test_the_cleaned_image_feeds_image_to_3d_and_becomes_the_reference():
    from app.api.vendor import conditioned_image

    async def go():
        async def cleaned():
            return b"cleaned-png"

        meshy = FakeMeshy()
        pending = _pending()
        pending["clean_job"] = asyncio.create_task(cleaned())
        await asyncio.sleep(0)
        out = await conditioned_image(_request(), meshy, "local-1", pending)
        return out, pending, meshy

    out, pending, meshy = anyio.run(go)
    assert out.status == "IN_PROGRESS" and pending["stage"] == "mesh" and pending["mesh_task_id"] == "mesh-1"
    assert meshy.calls["mesh"]["image"] == b"cleaned-png" and meshy.calls["mesh"].get("input_task_id") is None
    assert pending["reference"] == b"cleaned-png"
    assert [(s.id, s.status) for s in out.steps] == [("image", "SUCCEEDED"), ("mesh", "IN_PROGRESS")]


def test_a_failed_cleaning_task_does_not_start_3d():
    from app.api.vendor import conditioned_image

    async def go():
        async def refused():
            raise vision.VisionImageError("x")

        meshy = FakeMeshy()
        pending = _pending()
        pending["clean_job"] = asyncio.create_task(refused())
        await asyncio.sleep(0)
        return await conditioned_image(_request(), meshy, "local-1", pending), meshy

    out, meshy = anyio.run(go)
    assert "mesh" not in meshy.calls
    assert out.status == "FAILED" and out.steps[0].status == "FAILED" and out.steps[1].status == "PENDING"
    assert "not generated" in out.error and "x" in out.error


def test_gemini_cleans_the_photo_itself():
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        data = base64.b64encode(b"cleaned").decode()
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": data}}]}}]})

    settings = Settings(gemini_api_key=SecretStr("gem-test"), plan_vision_model="gemini-3.1-pro-preview")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await vision.clean_image("clean it", PNG, "image/png", settings, client, "gemini-3.1-pro-preview")

    assert anyio.run(go) == b"cleaned"
    body = json.loads(sent[0].content)
    assert sent[0].url.path.endswith("/models/gemini-3.1-pro-preview:generateContent")
    assert body["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]
    assert body["contents"][0]["parts"][0]["text"] == "clean it"
    assert body["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/png"
    assert "image-to-image" not in sent[0].url.path


def test_an_openai_vision_model_cleans_through_its_image_tool():
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"output": [{"type": "image_generation_call", "result": base64.b64encode(b"cleaned").decode()}]})

    settings = Settings(openai_api_key=SecretStr("sk-test"), plan_vision_model="gpt-6-astra")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await vision.clean_image("clean it", PNG, "image/png", settings, client, "gpt-6-astra")

    assert anyio.run(go) == b"cleaned"
    body = json.loads(sent[0].content)
    assert sent[0].url.path == "/v1/responses" and body["model"] == "gpt-6-astra"
    assert body["tools"][0]["type"] == "image_generation"
    assert "meshy" not in sent[0].url.path


def test_loose_objects_left_in_the_model_are_flagged():
    from app.api.vendor import loose_objects

    doc = {"components": [{"name": "Tabletop", "type": "top"}, {"name": "Left_Side_Panel", "type": "panel"},
                          {"name": "Stack of Books", "type": "decor"}, {"name": "Vase", "type": "other"}]}
    assert loose_objects(doc) == ["Stack of Books", "Vase"]
    assert loose_objects({"components": doc["components"][:2]}) == []
