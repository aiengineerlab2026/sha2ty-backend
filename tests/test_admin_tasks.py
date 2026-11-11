"""The admin opens one request and sees it from both sides: the customer's, and every vendor's."""

import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api import admin
from app.api.admin import issue_token
from app.core.config import Settings
from app.main import app

SB = "https://sb.example"
TASK = "11111111-1111-1111-1111-111111111111"


def test_admin_sees_a_request_from_the_customer_and_every_vendor(monkeypatch):
    settings = Settings(supabase_url=SB, supabase_service_role_key=SecretStr("srv"), admin_password=SecretStr("pw"), admin_session_secret=SecretStr("s" * 32))
    monkeypatch.setattr(admin, "get_settings", lambda: settings)
    monkeypatch.setattr(admin, "proposal_spec", lambda task, p: {"title": task["title"], "from": p["id"]})
    users = {"c1": ("Sara A", "sara@example.com"), "v1": ("", "oak@studio.example"), "v2": ("", "pine@studio.example")}

    def db(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        assert request.headers["apikey"] == "srv"  # the service key: the admin sees every row
        if path.endswith("/freelance_tasks"):
            return httpx.Response(200, json=[{"id": TASK, "title": "Oak bed", "customer_id": "c1", "status": "awarded", "awarded_bid_id": "b2", "spec": {}, "description": "King size"}])
        if path.endswith("/task_bids"):
            return httpx.Response(200, json=[{"id": "b1", "vendor_id": "v1", "price": 9000, "days": 20, "status": "rejected"}, {"id": "b2", "vendor_id": "v2", "price": 8000, "days": 15, "status": "accepted"}])
        if path.endswith("/task_proposals"):
            return httpx.Response(200, json=[{"id": "p1", "vendor_id": "v1", "note": "Thinner legs", "status": "pending", "data": {}}])
        if path.endswith("/task_messages"):
            return httpx.Response(200, json=[
                {"id": "m1", "vendor_id": "v2", "sender_id": "v2", "body": "Which oak?", "created_at": "2026-11-01"},
                {"id": "m2", "vendor_id": "v2", "sender_id": "c1", "body": "Light oak", "created_at": "2026-11-02"},
                {"id": "m3", "vendor_id": "v3", "sender_id": "v3", "body": "Still open?", "created_at": "2026-11-03"},
            ])
        if path.endswith("/vendors"):
            return httpx.Response(200, json=[{"id": "v1", "name": "Oak Studio", "location": "Cairo"}, {"id": "v2", "name": "Pine Works", "location": "Giza"}, {"id": "v3", "name": "Elm", "location": ""}])
        if "/auth/v1/admin/users/" in path:
            uid = path.rsplit("/", 1)[1]
            if uid not in users:
                return httpx.Response(404)
            name, email = users[uid]
            return httpx.Response(200, json={"id": uid, "email": email, "user_metadata": {"full_name": name}})
        return httpx.Response(404)

    with TestClient(app) as client:
        app.state.auth_http = httpx.AsyncClient(transport=httpx.MockTransport(db))
        assert client.get(f"/api/admin/tasks/{TASK}").status_code == 401  # admins only
        body = client.get(f"/api/admin/tasks/{TASK}", headers={"X-Admin-Token": issue_token(settings)[0]}).json()

    assert body["task"]["title"] == "Oak bed" and body["customer"] == {"id": "c1", "name": "Sara A", "email": "sara@example.com", "phone": ""}
    names = [th["vendor"]["name"] for th in body["vendors"]]
    assert names == ["Pine Works", "Oak Studio", "Elm"]  # hired first, then other offers, then questions only
    hired, other, asking = body["vendors"]
    assert hired["hired"] and hired["bid"]["price"] == 8000 and hired["vendor"]["email"] == "pine@studio.example"
    assert [(m["from"], m["body"]) for m in hired["messages"]] == [("vendor", "Which oak?"), ("customer", "Light oak")]
    assert not other["hired"] and other["proposals"][0]["note"] == "Thinner legs" and other["proposals"][0]["spec"] == {"title": "Oak bed", "from": "p1"}
    assert asking["bid"] is None and asking["messages"][0]["body"] == "Still open?"


def test_an_unknown_request_is_not_found(monkeypatch):
    settings = Settings(supabase_url=SB, supabase_service_role_key=SecretStr("srv"), admin_session_secret=SecretStr("s" * 32))
    monkeypatch.setattr(admin, "get_settings", lambda: settings)
    with TestClient(app) as client:
        app.state.auth_http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
        res = client.get(f"/api/admin/tasks/{TASK}", headers={"X-Admin-Token": issue_token(settings)[0]})
    assert res.status_code == 404
