"""Admin portal woods: admin-only, written to the Sha2ty vendor's wood rows only, inputs checked."""

import json

import httpx
from fastapi.testclient import TestClient

from app.api import admin_woods
from app.api.admin import issue_token
from app.core.config import get_settings
from app.main import app

PLATFORM = "99999999-9999-9999-9999-999999999999"
WOOD = "11111111-1111-1111-1111-111111111111"


class FakeDb:
    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.method, str(request.url), body))
        if request.method == "GET" and request.url.path.endswith("/vendor_woods"):
            return httpx.Response(200, json=[{"wood_id": WOOD}, {"wood_id": WOOD}])
        if request.method == "GET":
            return httpx.Response(200, json=[{"id": WOOD, "name": "Oak", "wood_type": "oak"}])
        return httpx.Response(200, json=[{"id": WOOD, **body}])


class FakeCdn:
    def upload(self, key, data, content_type=None):
        return f"https://cdn.test/{key}"


def client(db: FakeDb, monkeypatch) -> TestClient:
    async def platform(_request):
        return {"id": PLATFORM}

    monkeypatch.setattr(admin_woods, "_platform", platform)
    test = TestClient(app)
    app.state.auth_http = httpx.AsyncClient(transport=httpx.MockTransport(db))
    app.state.cdn = FakeCdn()
    return test


def test_woods_are_admin_only():
    test = TestClient(app)
    assert test.get("/api/admin/woods").status_code == 401
    assert test.patch(f"/api/admin/woods/{WOOD}", data={"name": "x", "wood_type": "oak", "price_unit": "m3"}).status_code == 401


def test_admin_lists_edits_and_adds_woods_only_on_the_platform_vendor(monkeypatch):
    db = FakeDb()
    test = client(db, monkeypatch)
    admin = {"X-Admin-Token": issue_token(get_settings())[0]}

    listed = test.get("/api/admin/woods", headers=admin).json()["woods"]
    assert listed[0]["vendors"] == 2  # two vendors work with it

    form = {"name": "European oak", "wood_type": "oak", "price_unit": "m3", "price": "95000", "thicknesses": "20, 40", "roughness": "0.5"}
    saved = test.patch(f"/api/admin/woods/{WOOD}", headers=admin, data=form)
    assert saved.status_code == 200, saved.text
    method, url, body = db.calls[-1]
    assert method == "PATCH" and f"vendor_id=eq.{PLATFORM}" in url and "kind=eq.wood" in url  # never another vendor's row
    assert body["price"] == 95000 and body["thicknesses_mm"] == [20, 40] and "texture_url" not in body  # photo kept

    added = test.post("/api/admin/woods", headers=admin, data={**form, "price_unit": "sheet", "sheet_w": "1.22", "sheet_h": "2.44"}, files={"image": ("oak.jpg", b"jpg", "image/jpeg")})
    assert added.status_code == 200, added.text
    _, _, body = db.calls[-1]
    assert body["vendor_id"] == PLATFORM and body["kind"] == "wood" and body["texture_url"].startswith(f"https://cdn.test/{PLATFORM}/woods/")

    assert test.patch(f"/api/admin/woods/{WOOD}", headers=admin, data={**form, "wood_type": "plastic"}).status_code == 422
    assert test.post("/api/admin/woods", headers=admin, data={**form, "price_unit": "sheet"}, files={"image": ("oak.jpg", b"jpg", "image/jpeg")}).status_code == 422  # no sheet size
    assert test.post("/api/admin/woods", headers=admin, data=form, files={"image": ("x.gif", b"gif", "image/gif")}).status_code == 422
