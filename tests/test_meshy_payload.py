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
    assert body["should_texture"] is True and body["ai_model"] == "latest"

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
    # The real size controls Meshy has (no file-size option exists): remesh to a triangle budget, 2k textures.
    assert body["should_remesh"] is True and body["target_polycount"] == 30000 and body["texture_resolution"] == "2k"
    # File size is not texture: no size rule in texture_prompt (Meshy uses it for the texture only).
    assert "under 8 MB" not in body["texture_prompt"] and "dining chair" in body["texture_prompt"] and len(body["texture_prompt"]) <= 800
    assert target_polycount("furniture", "corner sofa") > target_polycount("furniture", "dining chair")
    astra = astra_prompt("furniture", "Oslo bed", None, "double bed")
    assert "under 8 MB" in astra and "headboard" in astra


def test_the_furniture_type_goes_with_the_photo_in_one_image_to_image_request(caplog):
    """Meshy's Image to 3D has no prompt field (only texture_prompt, for the texture), so the furniture type goes in
    Image to Image's prompt, sent with the photo in the same request; Image to 3D then builds from that task."""
    from app.services.generation_prompt import meshy_prompt

    sent: list[httpx.Request] = []
    meshy = _service(sent)
    prompt = meshy_prompt("furniture", "child wardrobe")

    async def run():
        with caplog.at_level(logging.INFO):
            await meshy.create_image_to_image_task(prompt, image=b"\x89PNG photo", mime_type="image/png")
            await meshy.create_image_to_3d_task(GenerationOptions(should_texture=True), input_task_id="task-123", texture_prompt="A child wardrobe.")

    anyio.run(run)
    image_req, mesh_req = sent
    assert image_req.url.path == "/openapi/v1/image-to-image"  # not text-to-3d: the photo is the subject
    body = json.loads(image_req.content)
    assert body["prompt"].startswith("Convert this child wardrobe into a high-quality, realistic, modular 3D model. Preserve the reference image's geometry, colors, textures, and proportions.")
    assert body["reference_image_urls"][0].startswith("data:image/png;base64,") and len(body["reference_image_urls"]) == 1
    assert "generate_multi_view" not in body  # one image: what Image to 3D's input_task_id requires

    assert mesh_req.url.path == "/openapi/v1/image-to-3d"
    mesh = json.loads(mesh_req.content)
    assert mesh["input_task_id"] == "task-123" and "image_url" not in mesh and "prompt" not in mesh  # no unsupported field

    logs = caplog.text
    assert "child wardrobe" in logs and "reference_image_urls" in logs  # the payload is logged with the type…
    assert "base64," not in logs and "msy-secret" not in logs  # …without the photo's data or the key


def test_a_failed_image_step_still_builds_from_the_photo_and_says_so(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import vendor
    from app.main import app

    made = {}

    class FakeMeshy:
        async def create_image_to_image_task(self, _prompt, **_kwargs):
            return "img-1"

        async def get_image_to_image_task(self, _task_id):
            return {"status": "FAILED", "progress": 0, "task_error": {"message": "moderation"}}

        async def create_image_to_3d_task(self, _options, **kwargs):
            made.update(kwargs)
            return "mesh-1"

    async def one_try(*_args):
        return 4

    async def no_details(*_args, **_kwargs):
        return None

    monkeypatch.setattr(vendor, "take_try", one_try)
    monkeypatch.setattr(vendor, "extract_product_details", no_details)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: FakeMeshy(),
                                     vendor.get_settings: lambda: Settings(meshy_api_key="msy-test", gemini_api_key="gem-test", storage_dir=tmp_path)})
    try:
        with TestClient(app) as client:
            png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
            client.post("/api/vendor/generate", files={"image": ("p.png", png, "image/png")}, data={"category": "furniture", "kind": "armchair"})
            step = client.get("/api/vendor/tasks/img-1").json()
            pending = app.state.pending_cdn["img-1"]
    finally:
        app.dependency_overrides.clear()
    assert step["status"] == "IN_PROGRESS" and step["progress"] == 20
    assert made["input_task_id"] is None and made["image"] == png  # the photo itself, still image-to-3D
    assert "armchair" in made["texture_prompt"] and pending["mesh_task_id"] == "mesh-1"
    assert "could not shape this model" in pending["conditioning"]  # shown in the model's notes, not hidden
