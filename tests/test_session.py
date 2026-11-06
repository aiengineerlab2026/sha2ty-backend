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


def test_the_public_anon_key_is_replaced_by_the_signed_in_users_token(monkeypatch):
    c = client(monkeypatch)
    c.cookies.set(session.ACCESS_COOKIE, FRESH)
    assert c.get("/api/test-echo", headers={"Authorization": "Bearer anon"}).json()["authorization"] == f"Bearer {FRESH}"


def test_the_database_passthrough_sends_the_users_token_and_a_readable_body(monkeypatch):
    import gzip

    from app.api import sb_proxy

    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["cookie"] = request.headers.get("cookie")
        return httpx.Response(200, content=gzip.compress(b'[{"id":"m1"}]'), headers={"content-encoding": "gzip", "content-type": "application/json"})

    c = client(monkeypatch)
    monkeypatch.setattr(sb_proxy, "get_settings", session.get_settings)
    app.state.auth_http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    c.cookies.set(session.ACCESS_COOKIE, FRESH)
    res = c.patch("/api/sb/rest/v1/models?id=eq.m1", headers={"Authorization": "Bearer anon", "Accept-Encoding": "br, zstd"}, json={"status": "ready"})
    assert res.json() == [{"id": "m1"}] and "content-encoding" not in res.headers
    assert seen == {"authorization": f"Bearer {FRESH}", "cookie": None}  # the user's token, never the browser's cookies


def test_logout_clears_every_session_cookie_even_with_an_expired_token(monkeypatch):
    c = client(monkeypatch)
    c.cookies.set(session.ACCESS_COOKIE, OLD)  # expired: the middleware would refresh it on any other route
    c.cookies.set(session.REFRESH_COOKIE, "r1")
    c.cookies.set(session.ADMIN_COOKIE, "adm")
    res = c.post("/api/auth/logout")
    cookies = res.headers.get_list("set-cookie")
    assert all("Max-Age=0" in v or 'expires=' in v.lower() for v in cookies) and len(cookies) == 3
    assert not any("r2" in v for v in cookies)  # no refreshed token written back


def test_a_site_on_another_domain_gets_cookies_it_can_send_cross_site(monkeypatch):
    c = client(monkeypatch)
    cross = {"Origin": "https://www.sha2ty.net", "Host": "sha2ty-1.europe-west1.run.app", "X-Forwarded-Proto": "https"}
    for cookie in c.post("/api/auth/login", json={"email": "a@b.c", "password": "pw"}, headers=cross).headers.get_list("set-cookie"):
        assert "SameSite=none" in cookie and "Secure" in cookie and "Partitioned" in cookie and "HttpOnly" in cookie
    same = c.post("/api/auth/login", json={"email": "a@b.c", "password": "pw"}, headers={"Origin": "http://localhost:3000"})
    assert all("SameSite=lax" in v and "Partitioned" not in v for v in same.headers.get_list("set-cookie"))


def test_only_a_valid_admin_session_counts_and_signing_in_as_a_user_ends_it(monkeypatch):
    from app.api import admin

    c = client(monkeypatch)
    monkeypatch.setattr(admin, "token_valid", lambda token, settings: token == "good")
    c.cookies.set(session.ADMIN_COOKIE, "forged")
    res = c.get("/api/auth/admin-me")
    assert res.json() == {"admin": False} and "sha2ty-admin=" in res.headers["set-cookie"]  # cleared
    c.cookies.set(session.ADMIN_COOKIE, "good")
    assert c.get("/api/auth/admin-me").json() == {"admin": True}
    login = c.post("/api/auth/login", json={"email": "a@b.c", "password": "pw"})
    assert any(v.startswith("sha2ty-admin=") and "Max-Age=0" in v for v in login.headers.get_list("set-cookie"))


def test_an_admin_cookie_beside_another_signed_in_account_is_not_an_admin(monkeypatch):
    from app.api import admin
    from app.services import platform

    c = client(monkeypatch)
    monkeypatch.setattr(admin, "token_valid", lambda token, settings: True)
    c.cookies.set(session.ADMIN_COOKIE, "good")
    c.cookies.set(session.ACCESS_COOKIE, FRESH)
    monkeypatch.setattr(platform, "is_platform", lambda *a: _false())
    res = c.get("/api/auth/admin-me")
    assert res.json() == {"admin": False} and "sha2ty-admin=" in res.headers["set-cookie"]
    monkeypatch.setattr(platform, "is_platform", lambda *a: _true())
    assert c.get("/api/auth/admin-me").json() == {"admin": True}  # the Sha2ty account an admin opened


async def _false():
    return False


async def _true():
    return True
