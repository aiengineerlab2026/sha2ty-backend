"""The 3D generator is the admin's choice: admin-only, and only a generator with its API key on the server."""

import json

import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api import admin
from app.api.admin import issue_token
from app.core.config import get_settings
from app.main import app
from app.services import generator_setting


def test_admin_sets_the_generator(monkeypatch):
    saved: list[dict] = []

    def db(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            saved.append(json.loads(request.content))
            return httpx.Response(201)
        return httpx.Response(200, json=[{"value": saved[-1]["value"]}] if saved else [])

    settings = get_settings().model_copy(update={"meshy_api_key": SecretStr("m"), "openai_api_key": SecretStr("")})
    monkeypatch.setattr(admin, "get_settings", lambda: settings)
    monkeypatch.setattr(generator_setting, "_cached", {})
    test = TestClient(app)
    app.state.auth_http = httpx.AsyncClient(transport=httpx.MockTransport(db))
    assert test.put("/api/admin/generator", json={"provider": "meshy"}).status_code == 401

    headers = {"X-Admin-Token": issue_token(settings)[0]}
    assert test.put("/api/admin/generator", headers=headers, json={"provider": "astra"}).status_code == 409  # no OpenAI key
    assert test.put("/api/admin/generator", headers=headers, json={"provider": "sol"}).status_code == 409
    assert test.put("/api/admin/generator", headers=headers, json={"provider": "meshy"}).json() == {"provider": "meshy"}
    assert saved == [{"key": "model_generator", "value": "meshy"}]
    body = test.get("/api/admin/generator", headers=headers).json()
    assert body["provider"] == "meshy" and [o["configured"] for o in body["options"]] == [True, False, False]


def test_match_model_defaults_to_sol_and_only_offers_gpts(monkeypatch):
    saved: list[dict] = []

    def db(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            saved.append(json.loads(request.content))
            return httpx.Response(201)
        return httpx.Response(200, json=[{"value": saved[-1]["value"]}] if saved else [])

    settings = get_settings().model_copy(update={"openai_api_key": SecretStr("k")})
    monkeypatch.setattr(admin, "get_settings", lambda: settings)
    monkeypatch.setattr(generator_setting, "_cached", {})
    test = TestClient(app)
    app.state.auth_http = httpx.AsyncClient(transport=httpx.MockTransport(db))
    headers = {"X-Admin-Token": issue_token(settings)[0]}
    body = test.get("/api/admin/match-model", headers=headers).json()
    assert body["provider"] == "sol" and [o["id"] for o in body["options"]] == ["astra", "sol"]
    assert test.put("/api/admin/match-model", headers=headers, json={"provider": "meshy"}).status_code == 422
    assert test.put("/api/admin/match-model", headers=headers, json={"provider": "astra"}).json() == {"provider": "astra"}
    assert saved == [{"key": "match_model", "value": "astra"}]
