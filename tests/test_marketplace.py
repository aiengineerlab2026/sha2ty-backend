"""Implementation requests and vendor profiles through the real routes (in-memory PostgREST from test_designs)."""

import uuid

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api import designs as designs_api
from app.api import marketplace
from app.core.config import Settings
from app.main import app
from app.services.design_store import DesignStore
from app.services.plan_store import PlanStore
from tests.test_designs import ALICE, BOB, FILES, MODEL, SB, VENDOR, FakeSupabase


class Fake(FakeSupabase):
    def __init__(self):
        super().__init__()
        self.tables.update(freelance_tasks=[], task_bids=[], vendor_reviews=[], vendors=[
            {"id": VENDOR, "name": "Oslo Studio", "location": "Cairo", "logo_url": None, "created_at": "2026-01-01", "status": "approved",
             "email": "owner@studio.example", "full_name": "Private Owner"},
        ])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        status_in = params.get("status", "")
        if request.method == "GET" and status_in.startswith("in.("):
            wanted = status_in[4:-1].split(",")
            table = request.url.path.removeprefix("/rest/v1/")
            rows = [r for r in self.tables[table] if r.get("status") in wanted and all(
                str(r.get(k)) == v.removeprefix("eq.") for k, v in params.items() if v.startswith("eq."))]
            return httpx.Response(200, json=rows)
        response = super().__call__(request)
        if request.method == "GET" and "task:freelance_tasks" in params.get("select", ""):
            tasks = {t["id"]: t for t in self.tables["freelance_tasks"]}
            return httpx.Response(200, json=[{**b, "task": tasks.get(b["task_id"])} for b in response.json()])
        return response


@pytest.fixture
def env(monkeypatch):
    fake = Fake()
    settings = Settings(supabase_url=SB, supabase_service_role_key=SecretStr("srv"))
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    monkeypatch.setattr(designs_api, "get_settings", lambda: settings)
    monkeypatch.setattr(marketplace, "get_settings", lambda: settings)
    app.dependency_overrides[designs_api.designs] = lambda: DesignStore(settings, http)
    app.dependency_overrides[designs_api.plans] = lambda: PlanStore(settings, http)

    def user(request: Request) -> str:
        return request.headers.get("x-test-user", ALICE)

    app.dependency_overrides[designs_api.current_user] = user
    with TestClient(app) as client:
        app.state.auth_http = http
        yield fake, client
    app.dependency_overrides.clear()


def _bed(client):
    item = {"dimensions": {"width": 2.1, "height": 1.1, "depth": 2.1},
            "components": {"components": [{"id": "headboard", "name": "Headboard"}, {"id": "frame", "name": "Frame"}]},
            "material_edits": {"headboard": {"mapUrl": f"{FILES}/{ALICE}/t.png", "color": "#ffffff", "tile": 0.25, "mapping": "box"}}}
    return client.post("/api/designs/from-model", json={"model_id": MODEL, "title": "My Modern Bed", "item": item, "glb_url": f"{FILES}/{ALICE}/d/a.glb"}).json()


def test_request_is_built_from_the_saved_version_and_not_duplicated(env):
    fake, client = env
    bed = _bed(client)
    body = {"description": "Walnut legs please", "location": "Cairo", "budget_max": 30000, "deadline": "2026-12-20"}
    made = client.post(f"/api/designs/{bed['id']}/request", json=body)
    assert made.status_code == 201
    task = made.json()
    spec = task["spec"]["design"]
    assert task["customer_id"] == ALICE and task["status"] == "published" and task["scope"] == "item"
    assert task["design_version_id"] == bed["current"]["id"]
    assert spec["dimensions"] == {"width": 2.1, "height": 1.1, "depth": 2.1}
    assert spec["materials"] == [{"part": "Headboard", "part_id": "headboard", "mapUrl": f"{FILES}/{ALICE}/t.png", "color": "#ffffff", "tile": 0.25, "mapping": "box"}]
    assert spec["glb_url"].endswith("a.glb") and spec["source"]["model_id"] == MODEL
    # Asking again for the same version returns the open request.
    assert client.post(f"/api/designs/{bed['id']}/request", json=body).json()["id"] == task["id"]
    assert len(fake.tables["freelance_tasks"]) == 1
    # A new saved version may be requested separately; the first request keeps its own version.
    client.post(f"/api/designs/{bed['id']}/versions", json={"item": {"dimensions": {"width": 2.3, "height": 1.1, "depth": 2.1}}})
    second = client.post(f"/api/designs/{bed['id']}/request", json=body).json()
    assert second["id"] != task["id"] and second["spec"]["design"]["dimensions"]["width"] == 2.3
    assert fake.tables["freelance_tasks"][0]["spec"]["design"]["dimensions"]["width"] == 2.1


def test_only_the_owner_can_request_and_only_items(env):
    fake, client = env
    bed = _bed(client)
    assert client.post(f"/api/designs/{bed['id']}/request", json={}, headers={"x-test-user": BOB}).status_code == 404
    plan = {"id": str(uuid.uuid4()), "owner_id": ALICE, "name": "Flat", "geometry": {"rooms": [{}]}, "items": []}
    fake.tables["floor_plans"].append(plan)
    room = client.post("/api/designs/from-plan", json={"plan_id": plan["id"]}).json()
    assert client.post(f"/api/designs/{room['id']}/request", json={}).status_code == 422


def test_vendor_profile_is_public_without_private_details(env):
    fake, client = env
    bed = _bed(client)
    task = client.post(f"/api/designs/{bed['id']}/request", json={}).json()
    task_row = fake.tables["freelance_tasks"][0]
    task_row["status"] = "completed"
    fake.tables["task_bids"].append({"id": str(uuid.uuid4()), "task_id": task["id"], "vendor_id": VENDOR, "price": 25000, "days": 20, "status": "accepted"})
    fake.tables["vendor_reviews"] += [{"id": "r1", "task_id": task["id"], "vendor_id": VENDOR, "rating": 5, "comment": "Great", "created_at": "2026-11-01"},
                                      {"id": "r2", "task_id": "t2", "vendor_id": VENDOR, "rating": 4, "comment": "", "created_at": "2026-10-01"}]
    profile = client.get(f"/api/vendors/{VENDOR}/profile").json()
    assert profile["vendor"] == {"id": VENDOR, "name": "Oslo Studio", "location": "Cairo", "logo_url": None, "created_at": "2026-01-01"}
    assert "email" not in str(profile) and "Private Owner" not in str(profile)
    assert profile["stats"] == {"completed_projects": 1, "active_projects": 0, "bids": 1, "rating": 4.5, "reviews": 2}
    assert profile["projects"][0]["title"] == "My Modern Bed" and [m["title"] for m in profile["listings"]] == ["Oslo bed"]
    fake.tables["vendors"][0]["status"] = "pending"
    assert client.get(f"/api/vendors/{VENDOR}/profile").status_code == 404


def test_request_can_name_the_parts_to_make(env):
    fake, client = env
    bed = _bed(client)
    only = client.post(f"/api/designs/{bed['id']}/request", json={"part_ids": ["headboard"]}).json()
    parts = only["spec"]["design"]["parts"]
    assert parts == [{"id": "headboard", "name": "Headboard", "requested": True}, {"id": "frame", "name": "Frame", "requested": False}]
    assert only["spec"]["design"]["whole_model"] is False
    # A different choice for the same version is a separate request; the same choice returns the same one.
    whole = client.post(f"/api/designs/{bed['id']}/request", json={}).json()
    assert whole["id"] != only["id"] and whole["spec"]["design"]["whole_model"] is True
    assert client.post(f"/api/designs/{bed['id']}/request", json={"part_ids": ["headboard"]}).json()["id"] == only["id"]
    assert client.post(f"/api/designs/{bed['id']}/request", json={"part_ids": []}).status_code == 422
    assert client.post(f"/api/designs/{bed['id']}/request", json={"part_ids": ["roof"]}).status_code == 422
    three = client.post(f"/api/designs/{bed['id']}/request", json={"quantity": 3}).json()
    assert three["spec"]["quantity"] == 3 and three["id"] != whole["id"]  # vendors see how many pieces
    assert whole["spec"]["quantity"] == 1
    assert client.post(f"/api/designs/{bed['id']}/request", json={"quantity": 0}).status_code == 422


PLATFORM = str(uuid.UUID(int=77))
SHA2TY_MODEL = str(uuid.UUID(int=78))


def _with_platform_model(fake):
    fake.tables["vendors"].append({"id": PLATFORM, "name": "Sha2ty", "location": "Egypt", "status": "approved", "platform": True, "created_at": "2026-01-01"})
    fake.tables["models"].append({"id": SHA2TY_MODEL, "title": "Sha2ty sofa", "vendor_id": PLATFORM, "status": "ready", "glb_url": f"{FILES}/{PLATFORM}/sofa.glb",
                                  "width_m": 2.0, "height_m": 0.9, "depth_m": 0.95, "vendors": {"status": "approved", "platform": True}})


def test_a_vendor_model_is_requested_from_that_vendor_only(env):
    fake, client = env
    bed = _bed(client)
    assert bed["current"]["data"]["item"]["source"]["source_type"] == "vendor"
    target = client.get(f"/api/designs/{bed['id']}/request-target").json()
    assert target == {"audience": "vendor_exclusive", "vendor": {"id": VENDOR, "name": "Oslo Studio"}, "available": True}
    task = client.post(f"/api/designs/{bed['id']}/request", json={}).json()
    assert task["audience"] == "vendor_exclusive" and task["exclusive_vendor_id"] == VENDOR
    assert task["spec"]["audience"] == "vendor_exclusive" and task["spec"]["vendor"]["name"] == "Oslo Studio"


def test_a_sha2ty_model_gets_an_open_request(env):
    fake, client = env
    _with_platform_model(fake)
    sofa = client.post("/api/designs/from-model", json={"model_id": SHA2TY_MODEL, "title": "My sofa", "item": {"dimensions": {"width": 2.2, "height": 0.9, "depth": 0.95}}}).json()
    assert sofa["current"]["data"]["item"]["source"]["source_type"] == "platform"
    assert client.get(f"/api/designs/{sofa['id']}/request-target").json() == {"audience": "open", "vendor": None, "available": True}
    task = client.post(f"/api/designs/{sofa['id']}/request", json={}).json()
    assert task["audience"] == "open" and task["exclusive_vendor_id"] is None


def test_a_copy_keeps_the_original_vendor_and_a_suspended_vendor_is_not_bypassed(env):
    fake, client = env
    bed = _bed(client)
    copy = client.post(f"/api/designs/{bed['id']}/duplicate", json={}).json()
    source = copy["current"]["data"]["item"]["source"]
    assert source["vendor_id"] == VENDOR and source["source_type"] == "vendor"
    # A save cannot change the origin either (the source is not part of what a save sends).
    sneaky = {"item": {"dimensions": {"width": 2, "height": 1, "depth": 2}, "source": {"model_id": SHA2TY_MODEL, "title": "x", "glb_url": "u", "source_type": "platform"}}}
    assert client.post(f"/api/designs/{copy['id']}/versions", json=sneaky).status_code == 422

    fake.tables["vendors"][0]["status"] = "rejected"
    assert client.get(f"/api/designs/{copy['id']}/request-target").json()["available"] is False
    refused = client.post(f"/api/designs/{copy['id']}/request", json={})
    assert refused.status_code == 409 and "Oslo Studio" in refused.json()["detail"]
    assert fake.tables["freelance_tasks"] == []  # not published to anyone else


def test_removed_parts_are_not_sent_to_vendors(env):
    fake, client = env
    bed = _bed(client)
    edits = {**bed["current"]["data"]["item"], "removed_parts": ["headboard"]}
    edits = {k: v for k, v in edits.items() if k not in ("source", "scale")}
    client.post(f"/api/designs/{bed['id']}/versions", json={"item": edits})
    spec = client.post(f"/api/designs/{bed['id']}/request", json={}).json()["spec"]["design"]
    assert [p["id"] for p in spec["parts"]] == ["frame"] and spec["materials"] == []
    assert client.post(f"/api/designs/{bed['id']}/request", json={"part_ids": ["headboard"]}).status_code == 422


def test_spec_carries_each_parts_wood_and_fabric_and_their_estimated_total() -> None:
    from app.api.marketplace import spec_from_version
    from app.schemas.design import ItemEdits

    oak = {"id": "w1", "kind": "wood", "name": "Oak", "wood_type": "oak", "thickness_mm": 25, "unit": "m3", "price": 40000,
           "finish": {"id": "v1", "name": "Walnut veneer", "wood_type": "veneer", "unit": "m2", "price": 300}, "quantity": 0.02, "finish_quantity": 1.2, "total": 1160}
    linen = {"id": "f1", "kind": "fabric", "name": "Linen", "unit": "m", "price": 250, "quantity": 1.6, "total": 400}
    edits = {"legs": {"mapUrl": "https://x/oak.jpg", "material": oak}, "headboard": {"mapUrl": "https://x/linen.jpg", "material": linen}}
    ItemEdits.model_validate({"material_edits": edits})  # the saved shape is accepted
    item = {"components": {"components": [{"id": "legs", "name": "Legs"}, {"id": "headboard", "name": "Headboard"}]}, "material_edits": edits}
    spec = spec_from_version({"id": "d"}, {"id": "v", "number": 1, "data": {"item": item}})["design"]
    assert {m["part_id"]: m["material"]["name"] for m in spec["materials"]} == {"legs": "Oak", "headboard": "Linen"}
    assert spec["materials_total"] == 1560
    # Only the parts asked for count; a part priced on request makes the total unknown.
    assert spec_from_version({"id": "d"}, {"id": "v", "number": 1, "data": {"item": item}}, ["legs"])["design"]["materials_total"] == 1160
    edits["legs"]["material"] = {**oak, "total": None}
    assert spec_from_version({"id": "d"}, {"id": "v", "number": 1, "data": {"item": item}})["design"]["materials_total"] is None
