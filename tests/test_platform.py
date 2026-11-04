"""The Sha2ty catalogue vendor: created once, approved with its logo, and an admin-only way to sign in as it."""

import json

import anyio
import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings
from app.main import app
from app.services import platform

SB = "https://sb.test"
USER = "99999999-9999-9999-9999-999999999999"


class FakeAuth:
    def __init__(self):
        self.vendors: list[dict] = []
        self.users_created = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content else {}
        if path == "/rest/v1/vendors" and method == "GET":
            wanted = request.url.params.get("platform") == "eq.true"
            return httpx.Response(200, json=[v for v in self.vendors if not wanted or v.get("platform")])
        if path == "/rest/v1/vendors" and method == "POST":
            self.vendors.append({**body, "logo_url": None})
            return httpx.Response(201, json=[self.vendors[-1]])
        if path == "/rest/v1/vendors" and method == "PATCH":
            self.vendors[0].update(body)
            return httpx.Response(200, json=[self.vendors[0]])
        if path == "/auth/v1/admin/users":
            self.users_created += 1
            return httpx.Response(200, json={"id": USER})
        if path == "/auth/v1/admin/generate_link":
            assert body == {"type": "magiclink", "email": "studio@sha2ty.app"}
            return httpx.Response(200, json={"id": USER, "hashed_token": "hash-1"})
        if path == "/auth/v1/verify":
            assert body == {"type": "magiclink", "token_hash": "hash-1"}
            return httpx.Response(200, json={"access_token": "at", "refresh_token": "rt", "expires_in": 3600})
        raise AssertionError(f"{method} {path}")


class FakeCdn:
    def __init__(self):
        self.files: dict[str, bytes] = {}

    def upload(self, key, data, content_type=None):
        self.files[key] = data
        return f"{SB}/storage/v1/object/public/cads/{key}"


def test_the_platform_vendor_is_made_once_and_admins_get_a_session_for_it():
    auth, cdn = FakeAuth(), FakeCdn()
    settings = Settings(supabase_url=SB, supabase_service_role_key=SecretStr("srv"))
    client = httpx.AsyncClient(transport=httpx.MockTransport(auth))

    first = anyio.run(platform.platform_session, settings, client, cdn)
    assert first["access_token"] == "at" and first["refresh_token"] == "rt"
    vendor = auth.vendors[0]
    assert vendor["id"] == USER and vendor["name"] == "Sha2ty" and vendor["status"] == "approved" and vendor["platform"] is True
    assert vendor["logo_url"].endswith(f"{USER}/brand/sha2ty-logo.svg") and b"<svg" in cdn.files[f"{USER}/brand/sha2ty-logo.svg"]

    anyio.run(platform.platform_session, settings, client, cdn)
    assert auth.users_created == 1 and len(auth.vendors) == 1  # reused, not created again


def test_studio_routes_are_admin_only():
    client = TestClient(app)
    assert client.post("/api/admin/studio-session").status_code == 401
    assert client.get("/api/admin/studio").status_code == 401
