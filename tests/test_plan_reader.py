import asyncio
import json
from pathlib import Path

import httpx

from app.core.config import Settings
from app.services.plan import image_processor
from app.services.plan.dimension_ocr import OcrBox
from app.services.plan.plan_reader import read_plan

FIXTURES = Path(__file__).parent / "fixtures"
TRUE_MPP = 0.3048 / 16.2
FT = 0.3048
# What GPT reads off the drawing (its printed sizes), names only reach the OpenCV pipeline.
READING = {
    "rooms": [
        {"name": "Bedroom", "width_m": 13 * FT, "length_m": 12 * FT, "printed": True},
        {"name": "Living room", "width_m": 16 * FT, "length_m": 13 * FT, "printed": True},
        {"name": "Dining", "width_m": 12.5 * FT, "length_m": 10 * FT, "printed": True},
        {"name": "Bedroom", "width_m": 14 * FT, "length_m": 10 * FT, "printed": True},
    ],
    "overall": None,
    "unit_on_plan": "imperial",
}


def names_only(_image):
    rows = json.loads((FIXTURES / "three_bedroom_apartment.ocr.json").read_text())
    return [OcrBox(**r) for r in rows if not any(ch.isdigit() for ch in r["text"])], []


def test_gpt_reading_sets_the_scale_when_the_plan_text_has_no_numbers(monkeypatch):
    monkeypatch.setattr(image_processor, "run_ocr", names_only)
    payload = image_processor.plan_from_image((FIXTURES / "three_bedroom_apartment.png").read_bytes(), reading=READING)
    scale = payload["scale"]
    assert "GPT reading" in scale["source"] and not scale["estimated"]
    assert abs(scale["meters_per_pixel"] / TRUE_MPP - 1) < 0.06
    assert payload["validation"]["ok"]
    # The correction view re-runs from the state: the GPT references travel with it.
    assert any(r["source"] == "gpt_printed" for r in payload["scale_refs"])


def test_failed_gpt_call_falls_back_to_opencv():
    transport = httpx.MockTransport(lambda request: httpx.Response(500, json={"error": "down"}))
    settings = Settings(openai_api_key="sk-test", openai_plan_model="gpt-test")
    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            return await read_plan(b"png", "image/png", settings, client)
    assert asyncio.run(run()) is None


def test_reading_is_parsed_from_structured_output():
    body = {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(READING)}]}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    settings = Settings(openai_api_key="sk-test", openai_plan_model="gpt-test")
    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            return await read_plan(b"png", "image/png", settings, client)
    assert len(asyncio.run(run())["rooms"]) == 4
