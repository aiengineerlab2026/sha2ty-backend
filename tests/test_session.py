"""Cookie sessions on the backend (app/api/session.py): login sets the cookies, the middleware turns them into
the bearer token every route reads, refreshes an expiring one and writes the rotated cookies back."""

import base64
import json
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api import session
from app.core.config import Settings
from app.main import app


def jwt(exp: float) -> str:
    part = base64.urlsafe_b64encode(json.dumps({"exp": int(exp)}).encode()).decode().rstrip("=")
    return f"h.{part}.s"


FRESH, OLD = jwt(time.time() + 3600), jwt(time.time() - 10)
calls: list[str] = []


def supabase(request: httpx.Request) -> httpx.Response:
    calls.append(request.url.path + ("?" + request.url.query.decode() if request.url.query else ""))
    if request.url.path.endswith("/token"):
        body = json.loads(request.content)
        if body.get("refresh_token") == "bad" or body.get("password") == "wrong":
            return httpx.Response(400, json={"error_description": "Invalid login credentials"})
        return httpx.Response(200, json={"access_token": FRESH, "refresh_token": "r2", "expires_in": 3600, "user": {"id": "u1"}})
    return httpx.Response(200, json={"id": "u1"})


echo = APIRouter()


@echo.get("/api/test-echo")
async def show(request: Request) -> dict:
    return {"authorization": request.headers.get("authorization"), "admin": request.headers.get("x-admin-token")}


app.include_router(echo)


def client(monkeypatch) -> TestClient:
    settings = Settings(supabase_url="https://sb.test", supabase_anon_key=SecretStr("anon"))
    monkeypatch.setattr(session, "get_settings", lambda: settings)
    c = TestClient(app)
    c.__enter__()
    app.state.auth_http = httpx.AsyncClient(transport=httpx.MockTransport(supabase))
    calls.clear()
    return c


def test_login_sets_httponly_cookies_and_they_become_the_bearer_token(monkeypatch):
    c = client(monkeypatch)
    assert c.post("/api/auth/login", json={"email": "a@b.c", "password": "wrong"}).status_code == 401
    res = c.post("/api/auth/login", json={"email": "a@b.c", "password": "pw"})
    assert res.status_code == 200 and "httponly" in res.headers["set-cookie"].lower()
    assert c.get("/api/test-echo").json()["authorization"] == f"Bearer {FRESH}"
    assert c.post("/api/auth/logout").json() == {"ok": True}
    assert c.get("/api/test-echo").json()["authorization"] is None


def test_an_expired_token_is_refreshed_and_a_refused_refresh_signs_out(monkeypatch):
    c = client(monkeypatch)
    c.cookies.set(session.ACCESS_COOKIE, OLD)
    c.cookies.set(session.REFRESH_COOKIE, "r1")
    res = c.get("/api/test-echo")
    assert res.json()["authorization"] == f"Bearer {FRESH}" and "sha2ty-refresh=r2" in res.headers["set-cookie"]
    c.cookies.clear()
    c.cookies.set(session.REFRESH_COOKIE, "bad")
    res = c.get("/api/test-echo")
    assert res.json()["authorization"] is None and 'sha2ty-refresh=""' in res.headers["set-cookie"]


def test_an_explicit_header_wins_and_the_admin_cookie_becomes_the_admin_header(monkeypatch):
    c = client(monkeypatch)
    c.cookies.set(session.ACCESS_COOKIE, FRESH)
    c.cookies.set(session.ADMIN_COOKIE, "adm")
    out = c.get("/api/test-echo", headers={"Authorization": "Bearer app"}).json()
    assert out == {"authorization": "Bearer app", "admin": "adm"}
