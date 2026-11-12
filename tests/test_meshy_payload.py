"""The request bodies httpx actually sends to Meshy (captured at the transport, after serialisation)."""

import json
import logging

import anyio
import httpx
from pydantic import SecretStr

from app.core.config import Settings
from app.schemas.product_details import ProductDetails
from app.schemas.vendor import GenerationOptions
from app.services.generation_prompt import meshy_texture_prompt, split_prompt
from app.services.meshy_service import MeshyService
from tests.test_vision_service import SOFA


def _service(sent: list[httpx.Request]) -> MeshyService:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(202, json={"result": "task-123"})

    settings = Settings(meshy_api_key=SecretStr("msy-secret"))
    api = httpx.AsyncClient(base_url=settings.meshy_base_url, transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer msy-secret"})
    return MeshyService(settings, api, api)


def test_texture_prompt_and_split_prompt_are_in_the_outgoing_payloads(caplog):
    sent: list[httpx.Request] = []
    meshy = _service(sent)
    prompt = meshy_texture_prompt("furniture", "Oslo sofa", ProductDetails.model_validate(SOFA), "corner sofa")
    parts = split_prompt("furniture", "corner sofa")

    async def run():
        with caplog.at_level(logging.INFO):
            await meshy.create_image_to_3d_task(GenerationOptions(should_texture=True, enable_pbr=False), image=b"\x89PNG photo", texture_prompt=prompt)
            await meshy.create_split_task("task-123", parts, keep_texture=True)

    anyio.run(run)
    image_req, split_req = sent
    assert image_req.url.path == "/openapi/v1/image-to-3d"
    body = json.loads(image_req.content)
    assert body["texture_prompt"] == prompt and "seat cushions" in body["texture_prompt"]
    assert body["should_texture"] is True and body["ai_model"] == "meshy-7.1" and body["image_enhancement"] is False

    assert split_req.url.path == "/openapi/v1/print/split"
    assert json.loads(split_req.content) == {"input_task_id": "task-123", "mode": "by_parts", "prompt": parts, "keep_texture": True}
    assert parts == "frame, seat cushions, back cushions, left armrest, right armrest, throw pillows, legs"

    logs = caplog.text
    assert "texture_prompt" in logs and "task_id=task-123" in logs
    assert "msy-secret" not in logs and "base64," not in logs  # no key, no photo data


def test_untextured_requests_send_no_texture_prompt():
    sent: list[httpx.Request] = []
    anyio.run(lambda: _service(sent).create_image_to_3d_task(GenerationOptions(should_texture=False), image=b"x", texture_prompt="ignored"))
    assert "texture_prompt" not in json.loads(sent[0].content)


def test_meshy_is_asked_for_a_light_model_and_both_prompts_carry_the_size_rule():
    from app.services.generation_prompt import astra_prompt, target_polycount

    sent: list[httpx.Request] = []
    prompt = meshy_texture_prompt("furniture", "Oslo chair", None, "dining chair")
    anyio.run(lambda: _service(sent).create_image_to_3d_task(GenerationOptions(should_texture=True), image=b"x", texture_prompt=prompt, target_polycount=target_polycount("furniture", "dining chair")))
    body = json.loads(sent[0].content)
    # Remesh is left off: it replaces the generated surface. Textures stay 2k; glb_size shrinks the file afterwards.
    assert body["should_remesh"] is False and "target_polycount" not in body and body["texture_resolution"] == "2k"
    # File size is not texture: no size rule in texture_prompt (Meshy uses it for the texture only).
    assert "under 8 MB" not in body["texture_prompt"] and "dining chair" in body["texture_prompt"] and len(body["texture_prompt"]) <= 800
    assert target_polycount("furniture", "corner sofa") > target_polycount("furniture", "dining chair")
    astra = astra_prompt("furniture", "Oslo bed", None, "double bed")
    assert "under 8 MB" in astra and "headboard" in astra


def test_the_photo_goes_straight_to_meshy_image_to_3d(caplog):
    """Image to Image redraws a picture (nano-banana / GPT Image). The selected Meshy model is Image to 3D, and the photo is its input."""
    sent: list[httpx.Request] = []
    meshy = _service(sent)

    async def run():
        with caplog.at_level(logging.INFO):
            await meshy.create_image_to_3d_task(
                GenerationOptions(should_texture=True, enable_pbr=True),
                image=b"\x89PNG photo",
                mime_type="image/png",
                texture_prompt="A child wardrobe.",
            )

    anyio.run(run)
    assert sent[0].url.path == "/openapi/v1/image-to-3d"
    body = json.loads(sent[0].content)
    assert body["ai_model"] == "meshy-7.1" and body["image_enhancement"] is False
    assert body["image_url"].startswith("data:image/png;base64,") and "input_task_id" not in body and "prompt" not in body
    assert body["texture_prompt"] == "A child wardrobe."
    assert "base64," not in caplog.text and "msy-secret" not in caplog.text


def test_generate_sends_the_photo_to_meshy_image_to_3d(tmp_path, monkeypatch):
    """The admin's Meshy choice starts an Image to 3D task from the uploaded photo, even when a public URL is also sent."""
    from fastapi.testclient import TestClient

    from app.api import vendor
    from app.main import app

    made = {}

    class FakeMeshy:
        async def create_image_to_3d_task(self, options, **kwargs):
            made["options"] = options
            made.update(kwargs)
            return "mesh-1"

    async def one_try(*_args):
        return 4

    async def no_details(*_args, **_kwargs):
        return None

    async def admin_chose_meshy(*_args, **_kwargs):
        return "meshy"

    monkeypatch.setattr(vendor, "take_try", one_try)
    monkeypatch.setattr(vendor, "extract_product_details", no_details)
    monkeypatch.setattr(vendor.generator_setting, "current", admin_chose_meshy)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: FakeMeshy(),
                                     vendor.get_settings: lambda: Settings(meshy_api_key="msy-test", gemini_api_key="gem-test", storage_dir=tmp_path)})
    try:
        with TestClient(app) as client:
            png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
            body = client.post(
                "/api/vendor/generate",
                files={"image": ("p.png", png, "image/png")},
                data={"category": "furniture", "kind": "armchair", "provider": "sol", "image_url": "https://cdn.test/photo.png"},
            ).json()
            pending = app.state.pending_cdn["mesh-1"]
    finally:
        app.dependency_overrides.clear()
    assert body["provider"] == "meshy" and body["task_id"] == "mesh-1" and body["status"] == "PENDING"
    assert made["image"] == png and made["image_url"] is None and made.get("input_task_id") is None
    assert made["options"].should_texture is True and made["options"].enable_pbr is True
    assert "armchair" in made["texture_prompt"] and pending["stage"] == "mesh" and pending["mesh_task_id"] == "mesh-1"
