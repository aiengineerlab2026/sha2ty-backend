from fastapi.testclient import TestClient

from app.api import admin
from app.core.config import Settings
from app.main import app


def settings() -> Settings:
    return Settings(admin_session_secret="test-secret", admin_session_hours=1)


def test_tokens_are_signed_and_expire():
    s = settings()
    token, expires = admin.issue_token(s, now=1000)
    assert admin.token_valid(token, s, now=1000)
    assert not admin.token_valid(token, s, now=expires + 1)  # expired
    assert not admin.token_valid(f"{expires + 9999}.{token.split('.')[1]}", s, now=1000)  # extended by hand
    assert not admin.token_valid(token, Settings(admin_session_secret="other"), now=1000)  # other secret
    assert not admin.token_valid("garbage", s)


def test_login_and_protected_routes(monkeypatch):
    monkeypatch.setattr(admin, "get_settings", settings)
    client = TestClient(app)
    assert client.post("/api/admin/login", json={"username": "admin", "password": "nope"}).status_code == 401
    ok = client.post("/api/admin/login", json={"username": "admin", "password": "admin"})
    assert ok.status_code == 200 and ok.json()["token"]
    assert client.get("/api/admin/vendors").status_code == 401  # no token
    assert client.post("/api/admin/vendors/not-a-uuid/status", json={"status": "approved"}, headers={"X-Admin-Token": ok.json()["token"]}).status_code == 422


def test_vendor_signup_needs_company_and_location():
    client = TestClient(app)
    body = {"email": "a@b.co", "password": "abcdefg1", "kind": "vendor", "name": "", "location": ""}
    assert client.post("/api/vendor/register", json=body).status_code == 422
