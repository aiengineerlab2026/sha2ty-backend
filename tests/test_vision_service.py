"""The shared vision service: both providers, the admin's model choice, and Workflow B (vendor photo →
ProductDetails JSON → Meshy prompt). Every HTTP call is mocked; no test pays for a model call."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import admin, vendor
from app.core.config import Settings
from app.main import app
from app.schemas.product_details import ProductDetails
from app.services import product_details, vision
from app.services.generation_prompt import astra_prompt, meshy_texture_prompt
from tests.conftest import REAL_CURRENT_MODEL

SOFA = {
    "object_type": "three-seat sofa", "matches_category": True, "description": "Grey fabric sofa",
    "bbox": [0.1, 0.2, 1.3, 0.9], "orientation": "three_quarter", "proportions": [2.2, 0.9, 0.85],
    "width_m": 2.2, "depth_m": 0.9, "height_m": 40.0, "dimension_source": "inferred",
    "materials": ["fabric", "wood"], "colors": ["grey", "walnut"],
    "components": [{"name": "seat cushions", "material": "fabric", "color": "grey", "bbox": [0.2, 0.4, 0.9, 0.6]},
                   {"name": "legs", "material": "wood", "color": "walnut"}],
    "hidden": "the back", "confidence": 0.8,
}


def settings(tmp_path, model="gemini-3.1-pro-preview") -> Settings:
    return Settings(gemini_api_key="gem-test", openai_api_key="sk-test", plan_vision_model=model, storage_dir=tmp_path)


def provider_reply(model: str, payload: dict) -> httpx.Response:
    text = json.dumps(payload)
    if vision.is_gemini(model):
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})
    return httpx.Response(200, json={"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]})


@pytest.mark.parametrize("model", list(vision.MODELS))
def test_product_details_same_json_from_both_providers(tmp_path, model):
    seen = []

    def handler(request):
        seen.append(request)
        return provider_reply(model, SOFA)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await product_details.extract_product_details(b"photo", "image/png", "furniture", "Oslo sofa", settings(tmp_path), client, model)

    details = asyncio.run(go())
    assert details.object_type == "three-seat sofa" and details.model == model
    assert details.bbox == [0.1, 0.2, 1.0, 0.9]  # clamped to the image
    assert details.height_m is None and any("height_m" in w for w in details.warnings)  # 40 m is not a sofa
    assert details.proportions == [1.0, 0.409, 0.386]
    host = seen[0].url.host
    assert host == ("generativelanguage.googleapis.com" if vision.is_gemini(model) else "api.openai.com")
    # cached: the same photo is read once per model
    asyncio.run(go())
    assert len(seen) == 1


def test_details_guide_the_generation_prompts_as_estimates():
    details = product_details.clean(ProductDetails.model_validate({**SOFA, "height_m": 0.85}))
    meshy = meshy_texture_prompt("furniture", "Oslo sofa", details)
    assert "three-seat sofa" in meshy and "grey fabric seat cushions" in meshy and len(meshy) <= 600
    astra = astra_prompt("furniture", "Oslo sofa", details)
    assert "roughly 2.20 m wide" in astra and "estimated" in astra and "does not show the back" in astra
    assert meshy_texture_prompt("furniture", "Oslo sofa") == meshy_texture_prompt("furniture", "Oslo sofa", None)  # no details: as before


def test_failed_vision_call_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "RETRY_DELAY", 0)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(503))) as client:
            return await product_details.extract_product_details(b"photo", "image/png", "table", "", settings(tmp_path), client, "gpt-6-astra")

    assert asyncio.run(go()) is None


def test_vendor_generate_reads_the_photo_before_meshy(tmp_path, monkeypatch):
    calls = {}

    async def fake_details(image, mime, category, title, _settings, _client, model, db=None):
        calls["details"] = (model, category)
        return product_details.clean(ProductDetails.model_validate(SOFA))

    class FakeMeshy:
        async def create_image_to_3d_task(self, _options, **kwargs):
            calls["prompt"] = kwargs["texture_prompt"]
            return "task-1"

    async def one_try(*_args):
        return 4

    monkeypatch.setattr(vendor, "extract_product_details", fake_details)
    monkeypatch.setattr(vendor, "take_try", one_try)
    app.dependency_overrides[vendor.current_user] = lambda: "vendor-1"
    app.dependency_overrides[vendor.get_meshy] = lambda: FakeMeshy()
    app.dependency_overrides[vendor.get_settings] = lambda: Settings(meshy_api_key="msy-test", gemini_api_key="gem-test", storage_dir=tmp_path)
    try:
        with TestClient(app) as client:
            png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
            response = client.post("/api/vendor/generate", files={"image": ("sofa.png", png, "image/png")}, data={"category": "furniture", "title": "Oslo sofa"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    body = response.json()
    assert calls["details"] == ("gemini-3.1-pro-preview", "furniture")  # the default vision model
    assert "three-seat sofa" in calls["prompt"]  # the reading guided Meshy
    assert body["details"]["object_type"] == "three-seat sofa" and body["details"]["dimension_source"] == "inferred"


def test_admin_choice_is_read_and_saved_in_the_database(monkeypatch):
    monkeypatch.setattr(vision, "current_model", REAL_CURRENT_MODEL)
    rows: dict[str, str] = {}

    def supabase(request):
        if request.method == "POST":
            body = json.loads(request.content)
            rows[body["key"]] = body["value"]
            assert "merge-duplicates" in request.headers["prefer"]  # upsert
            return httpx.Response(201)
        return httpx.Response(200, json=[{"value": rows["vision_model"]}] if "vision_model" in rows else [])

    s = Settings(supabase_url="https://db.test", supabase_service_role_key="service", plan_vision_model="gemini-3.1-pro-preview")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(supabase)) as client:
            first = await vision.current_model(s, client)  # no row yet: the default
            await vision.set_model(s, client, "gpt-6-astra")
            vision._cached = None  # as on another API host after the cache expires
            return first, await vision.current_model(s, client)

    assert asyncio.run(go()) == ("gemini-3.1-pro-preview", "gpt-6-astra")
    with pytest.raises(vision.VisionSettingError):
        asyncio.run(vision.set_model(s, httpx.AsyncClient(), "claude-vision"))


def test_unreadable_setting_falls_back_to_gemini(monkeypatch):
    monkeypatch.setattr(vision, "current_model", REAL_CURRENT_MODEL)
    s = Settings(supabase_url="https://db.test", supabase_service_role_key="service")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(404, json={"message": "relation does not exist"}))) as client:
            return await vision.current_model(s, client)

    assert asyncio.run(go()) == "gemini-3.1-pro-preview"  # table not created yet: the default, never an error


def test_admin_vision_endpoints(monkeypatch):
    saved = {}

    async def fake_current(_settings, _client):
        return saved.get("model", "gemini-3.1-pro-preview")

    async def fake_set(_settings, _client, model):
        saved["model"] = model
        return model

    monkeypatch.setattr(vision, "current_model", fake_current)
    monkeypatch.setattr(vision, "set_model", fake_set)
    monkeypatch.setattr(admin, "get_settings", lambda: Settings(admin_session_secret="t", gemini_api_key="gem", openai_api_key="sk-x"))
    client = TestClient(app)
    token = admin.issue_token(admin.get_settings())[0]
    assert client.get("/api/admin/vision").status_code == 401
    got = client.get("/api/admin/vision", headers={"X-Admin-Token": token}).json()
    assert got["model"] == "gemini-3.1-pro-preview" and {o["id"] for o in got["options"]} == {"gemini-3.1-pro-preview", "gpt-6-astra", "gpt-6.1-sol"}
    assert client.put("/api/admin/vision", json={"model": "gpt-6-astra"}, headers={"X-Admin-Token": token}).json() == {"model": "gpt-6-astra"}
    assert client.get("/api/admin/vision", headers={"X-Admin-Token": token}).json()["model"] == "gpt-6-astra"
    assert client.put("/api/admin/vision", json={"model": "other"}, headers={"X-Admin-Token": token}).status_code == 422


def fake_db():
    """A stand-in for the Supabase vision_cache table."""
    rows: dict[str, dict] = {}

    def handler(request):
        assert "vision_cache" in request.url.path and request.headers["apikey"] == "service"
        if request.method == "POST":
            body = json.loads(request.content)
            rows[body["key"]] = body
            return httpx.Response(201)
        key = request.url.params["key"].removeprefix("eq.")
        return httpx.Response(200, json=[{"result": rows[key]["result"]}] if key in rows else [])

    return rows, httpx.AsyncClient(transport=httpx.MockTransport(handler))


def db_settings(tmp_path, model="gemini-3.1-pro-preview") -> Settings:
    return Settings(gemini_api_key="gem-test", openai_api_key="sk-test", plan_vision_model=model, storage_dir=tmp_path,
                    supabase_url="https://db.test", supabase_service_role_key="service", plan_vision_cache=True)


@pytest.mark.parametrize("model", list(vision.MODELS))
def test_same_product_photo_is_read_once_from_the_database(tmp_path, model):
    calls = []
    rows, db = fake_db()

    def provider(request):
        calls.append(request)
        return provider_reply(model, SOFA)

    async def go(storage):
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
            return await product_details.extract_product_details(b"photo", "image/png", "furniture", "Oslo sofa", db_settings(storage, model), client, model, db=db)

    first = asyncio.run(go(tmp_path / "host-a"))
    second = asyncio.run(go(tmp_path / "host-b"))  # another API host: empty disk, same database
    assert len(calls) == 1 and second.object_type == first.object_type
    assert [r["kind"] for r in rows.values()] == ["product"] and next(iter(rows.values()))["model"] == model


def test_same_floor_plan_is_read_once_from_the_database(tmp_path):
    from app.services.plan.vision_analysis import analyze_plan

    answer = json.loads(open("tests/fixtures/open_plan_apartment.analysis.json").read())
    good = {k: answer[k] for k in ("rooms", "walls", "doors", "windows", "furniture", "scale_references", "warnings")}
    calls = []
    rows, db = fake_db()

    def provider(request):
        calls.append(request)
        return provider_reply("gemini-3.1-pro-preview", good)

    async def go(storage):
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
            return await analyze_plan(b"plan", "image/png", (100, 50), db_settings(storage), client, "gemini-3.1-pro-preview", db=db)

    first = asyncio.run(go(tmp_path / "host-a"))
    second = asyncio.run(go(tmp_path / "host-b"))
    assert len(calls) == 1 and not first.cached and second.cached and len(second.rooms) == len(first.rooms)
    assert [r["kind"] for r in rows.values()] == ["floor_plan"]
    # another model is another answer: the cache key includes it
    other = asyncio.run(analyze_plan(b"plan", "image/png", (100, 50), db_settings(tmp_path / "host-c", "gpt-6-astra"),
                                     httpx.AsyncClient(transport=httpx.MockTransport(lambda r: provider_reply("gpt-6-astra", good))), "gpt-6-astra", db=db))
    assert not other.cached and len(rows) == 2


def test_missing_cache_table_never_blocks_the_model(tmp_path):
    db = httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(404, json={"message": "relation does not exist"})))

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: provider_reply("gemini-3.1-pro-preview", SOFA))) as client:
            return await product_details.extract_product_details(b"photo", "image/png", "furniture", "", db_settings(tmp_path), client, "gemini-3.1-pro-preview", db=db)

    assert asyncio.run(go()).object_type == "three-seat sofa"


def test_floor_plans_call_the_model_on_every_upload_by_default(tmp_path):
    """PLAN_VISION_CACHE is off by default: the same plan uploaded twice is read twice, nothing is stored."""
    from app.services.plan.vision_analysis import analyze_plan

    answer = json.loads(open("tests/fixtures/open_plan_apartment.analysis.json").read())
    good = {k: answer[k] for k in ("rooms", "walls", "doors", "windows", "furniture", "scale_references", "warnings")}
    calls = []
    rows, db = fake_db()
    s = Settings(gemini_api_key="gem-test", storage_dir=tmp_path, supabase_url="https://db.test", supabase_service_role_key="service")
    assert not s.plan_vision_cache

    def provider(request):
        calls.append(request)
        return provider_reply("gemini-3.1-pro-preview", good)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
            return await analyze_plan(b"plan", "image/png", (100, 50), s, client, "gemini-3.1-pro-preview", db=db)

    assert not asyncio.run(go()).cached and not asyncio.run(go()).cached
    assert len(calls) == 2 and not rows and not (tmp_path / "plan-analysis").exists()
