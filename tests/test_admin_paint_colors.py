"""Admin paint palette: admin-only, colours checked."""

import json

import httpx
from fastapi.testclient import TestClient

from app.api.admin import issue_token
from app.core.config import get_settings
from app.main import app

COLOR = "22222222-2222-2222-2222-222222222222"


def test_paint_colors_are_admin_only_and_validated():
    calls: list[tuple[str, dict]] = []

    def db(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        calls.append((request.method, body))
        return httpx.Response(200, json=[{"id": COLOR, **body}])

    test = TestClient(app)
    app.state.auth_http = httpx.AsyncClient(transport=httpx.MockTransport(db))
    assert test.get("/api/admin/paint-colors").status_code == 401
    assert test.post("/api/admin/paint-colors", json={"name": "Navy", "hex": "#1f2f4d"}).status_code == 401

    admin = {"X-Admin-Token": issue_token(get_settings())[0]}
    added = test.post("/api/admin/paint-colors", headers=admin, json={"name": " Navy ", "hex": "#1F2F4D"})
    assert added.status_code == 200, added.text
    assert calls[-1] == ("POST", {"name": "Navy", "hex": "#1f2f4d"})
    assert test.patch(f"/api/admin/paint-colors/{COLOR}", headers=admin, json={"name": "Navy", "hex": "navy"}).status_code == 422
    assert test.delete(f"/api/admin/paint-colors/{COLOR}", headers=admin).status_code == 200
