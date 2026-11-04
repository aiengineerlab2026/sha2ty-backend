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
    assert parts == "frame, seat cushions, back pillows, armrests, legs"

    logs = caplog.text
    assert "texture_prompt" in logs and "task_id=task-123" in logs
    assert "msy-secret" not in logs and "base64," not in logs  # no key, no photo data


def test_untextured_requests_send_no_texture_prompt():
    sent: list[httpx.Request] = []
    anyio.run(lambda: _service(sent).create_image_to_3d_task(GenerationOptions(should_texture=False), image=b"x", texture_prompt="ignored"))
    assert "texture_prompt" not in json.loads(sent[0].content)
