"""Meshy texture_prompt: written by the vision model for each photo (services/texture_prompt.py) and present in the
OUTGOING Image to 3D request, whether the geometry comes from the cleaned Image to Image task or from the photo.
Paid APIs are mocked; Image to 3D goes through the real request builder to a mock transport (WireMeshy)."""

from __future__ import annotations

import json
import logging

import anyio
import httpx
import pytest

from app.api import vendor
from app.core.config import Settings
from app.services import texture_prompt as tp
from app.services.meshy_service import GenerationOptions, MeshyError
from tests.test_meshy_pipeline import BED_TEXTURE, PHOTO, TEXTURE_CALL, WireMeshy, _service, bed, run


def outgoing(meshy: WireMeshy) -> tuple[dict, httpx.Request]:
    (request,) = [q for q in meshy.sent if q.url.path.endswith("/image-to-3d")]
    return json.loads(request.content), request


def test_image_to_image_to_3d_sends_the_photos_own_texture_prompt_with_the_cleaned_task(tmp_path, monkeypatch, caplog):
    meshy = WireMeshy()
    with caplog.at_level(logging.INFO):
        r = run(tmp_path, monkeypatch, meshy, meshy_auto_split=False)
    body, request = outgoing(meshy)
    assert body["input_task_id"] == "img-1"  # geometry: the cleaned Image to Image task
    assert body["texture_prompt"] == BED_TEXTURE  # texture: this photo's description, in the real HTTP body
    assert "image_url" not in body and "texture_image_url" not in body and b"base64" not in request.content
    assert r.rec.vision == TEXTURE_CALL and not r.done["warnings"]
    # The outgoing payload is logged (fields and texture prompt), never the key or image data.
    assert "Meshy POST /image-to-3d" in caplog.text and "texture_prompt" in caplog.text and "warm medium-brown oak" in caplog.text
    assert "msy-secret-key" not in caplog.text and "base64," not in caplog.text


def test_direct_image_to_3d_sends_the_texture_prompt_and_no_texture_image(tmp_path, monkeypatch):
    meshy = WireMeshy()
    run(tmp_path, monkeypatch, meshy, meshy_clean_image=False, meshy_auto_split=False)
    body, _ = outgoing(meshy)
    assert body["image_url"].startswith("data:image/png;base64,")  # geometry: the photo
    assert body["texture_prompt"] == BED_TEXTURE and "texture_image_url" not in body  # never both (the prompt would win)
    assert "input_task_id" not in body


def test_the_prompt_is_written_for_each_photo_from_the_photo_itself(monkeypatch):
    """Not a fixed text: the vision model gets the photo and the furniture type, and its answer is what is sent."""
    seen: list[list[dict]] = []

    async def ask_json(content, _schema, purpose, *_a, **_k):
        seen.append(content)
        image = content[1]["image_url"]
        return json.dumps({"texture_prompt": f"Described from image {hash(image) % 997}: matte oak doors #A07850, vertical grain."})

    async def model(*_a):
        return "gpt-vision-test"

    monkeypatch.setattr(tp.vision, "ask_json", ask_json)
    monkeypatch.setattr(tp.vision, "current_model", model)
    s = Settings(openai_api_key="sk-test")
    state = type("S", (), {"auth_http": None, "openai_http": None})()
    one = anyio.run(lambda: tp.read_texture_prompt(PHOTO, "image/png", "furniture", "double bed", s, state))
    two = anyio.run(lambda: tp.read_texture_prompt(bed(white=True), "image/png", "wardrobe", None, s, state))
    assert one != two  # each photo its own description
    assert "double bed" in seen[0][0]["text"] and "wardrobe" in seen[1][0]["text"]
    assert "Never invent" in seen[0][0]["text"] and seen[0][1]["image_url"].startswith("data:image/png;base64,")


def test_when_the_description_cannot_be_written_the_model_says_so_and_textures_from_its_input(tmp_path, monkeypatch):
    async def down(*_a, **_k):
        raise tp.TexturePromptError("the vision model could not describe the photo (ConnectError)")

    monkeypatch.setattr(vendor, "read_texture_prompt", down)
    meshy = WireMeshy()
    r = run(tmp_path, monkeypatch, meshy, meshy_auto_split=False)
    body, _ = outgoing(meshy)
    assert "texture_prompt" not in body and "texture_image_url" not in body and body["input_task_id"] == "img-1"
    assert any("texture description could not be written" in w for w in r.done["warnings"])  # said, not silent
    assert r.done["components"]["meshyInput"]["textureSource"] == "input"


def test_meshy_n1_cannot_take_a_texture_prompt_and_is_refused_before_anything_is_paid():
    sent: list = []
    with pytest.raises(MeshyError) as err:
        anyio.run(lambda: _service(sent, meshy_texture_model="meshy-n1").create_image_to_3d_task(
            GenerationOptions(should_texture=True, enable_pbr=False), input_task_id="img-1", texture_prompt=BED_TEXTURE))
    assert err.value.status_code == 400 and "meshy-n1 does not take texture_prompt" in err.value.message and sent == []


def test_a_prompt_source_with_no_prompt_is_refused_not_sent_empty():
    sent: list = []
    with pytest.raises(MeshyError):
        anyio.run(lambda: _service(sent).create_image_to_3d_task(GenerationOptions(should_texture=True), input_task_id="img-1", texture_source="prompt"))
    assert sent == []


def test_the_description_fits_meshys_800_characters_at_a_sentence():
    long = " ".join(f"Door {i}: dusty pink #D9A5A8 lacquered MDF, satin, no visible grain." for i in range(30))
    out = tp.fit(long)
    assert len(out) <= 800 and out.endswith(".") and out.startswith("Door 0")


def test_an_explicit_override_is_honoured(tmp_path, monkeypatch):
    """MESHY_TEXTURE_SOURCE=image (an environment override) textures from the photo instead, and no prompt is written."""
    meshy = WireMeshy()
    r = run(tmp_path, monkeypatch, meshy, meshy_clean_image=False, meshy_auto_split=False, meshy_texture_source="image")
    body, _ = outgoing(meshy)
    assert body["texture_image_url"] == body["image_url"] and "texture_prompt" not in body and r.rec.vision == []
