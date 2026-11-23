"""Cleaning prompts and the image editors. The Meshy pipeline itself (Meshy Image to Image → Image to 3D) is in
tests/test_meshy_pipeline.py; vision.clean_image is no longer on the Meshy path (scripts/meshy_experiment.py uses it)."""

import base64
import json

import anyio
import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.services import vision
from app.services.generation_prompt import meshy_prompt, meshy_texture_prompt
from app.services.meshy_service import MeshyService

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
    assert p.startswith(f"Edit this photo into a clean product reference image of the {expected} shown. This is an edit of the photo, not a new picture.")
    assert f"Remove only the loose objects placed on, inside, or in front of the {expected}" in p
    assert "continue the furniture's own surface behind it" in p
    # The same camera and framing: a new viewpoint ("three-quarter product view") changed the furniture's proportions.
    assert "The same camera viewpoint, perspective, framing and image proportions" in p and "three-quarter" not in p
    assert "same material boundaries" in p and "Glass stays transparent and mirrors stay reflective." in p
    assert f"Do not redesign, simplify, straighten, restyle or relight the {expected}" in p


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
