"""The admin's list of every model with its owner: vendors', Sha2ty's and customers' private ones."""

import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api import admin
from app.core.config import Settings
from app.main import app
from app.api.admin import issue_token

SB = "https://sb.example"


def test_admin_sees_every_model_and_its_owner(monkeypatch):
    settings = Settings(supabase_url=SB, supabase_service_role_key=SecretStr("srv"), admin_password=SecretStr("pw"), admin_session_secret=SecretStr("s" * 32))
    monkeypatch.setattr(admin, "get_settings", lambda: settings)

    def db(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json=[
                {"id": "m1", "title": "Oslo sofa", "status": "ready", "vendor_id": "v1", "created_at": "2026-11-02", "vendors": {"name": "Oslo Studio", "email": "o@studio.example", "platform": False, "status": "approved"}},
                {"id": "m2", "title": "Sha2ty bed", "status": "ready", "vendor_id": "p1", "created_at": "2026-11-01", "vendors": {"name": "Sha2ty", "email": "", "platform": True, "status": "approved"}},
            ])
        if path.endswith("/user_models"):
            return httpx.Response(200, json=[{"id": "u1", "title": "Grandma's chair", "status": "ready", "owner_id": "c1", "created_at": "2026-11-03", "glb_url": "x.glb"}])
        if path.endswith("/auth/v1/admin/users"):
            return httpx.Response(200, json={"users": [{"id": "c1", "email": "sara@example.com", "user_metadata": {"full_name": "Sara A"}}]})
        return httpx.Response(404)

    with TestClient(app) as client:
        app.state.auth_http = httpx.AsyncClient(transport=httpx.MockTransport(db))
        assert client.get("/api/admin/models").status_code == 401
        rows = client.get("/api/admin/models", headers={"X-Admin-Token": issue_token(settings)[0]}).json()["models"]
    assert [(r["title"], r["kind"], r["owner"]["name"]) for r in rows] == [("Grandma's chair", "customer", "Sara A"), ("Oslo sofa", "vendor", "Oslo Studio"), ("Sha2ty bed", "platform", "Sha2ty")]
    assert rows[0]["owner"]["email"] == "sara@example.com" and "owner_id" not in rows[0]
