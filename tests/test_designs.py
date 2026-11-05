"""Designs + versions through the real FastAPI routes, against an in-memory PostgREST (the queries the store sends).
`add_design_version` here follows supabase/designs.sql (locked numbering, owner check, idempotent token), which is
tested on Postgres itself separately."""

import json
import uuid
from datetime import datetime, timezone
from urllib.parse import unquote

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api import designs as api
from app.core.config import Settings
from app.main import app
from app.schemas.design import DesignData
from app.services.design_store import DesignStore
from app.services.plan_store import PlanStore

SB = "https://sb.test"
FILES = f"{SB}/storage/v1/object/public/cads"
ALICE, BOB, VENDOR = (str(uuid.UUID(int=i)) for i in (1, 2, 3))
MODEL, HIDDEN = str(uuid.UUID(int=10)), str(uuid.UUID(int=11))


class FakeSupabase:
    def __init__(self):
        self.tables = {"designs": [], "design_versions": [], "floor_plans": [], "freelance_tasks": [], "models": [
            {"id": MODEL, "title": "Oslo bed", "vendor_id": VENDOR, "status": "ready", "glb_url": f"{FILES}/{VENDOR}/bed.glb",
             "width_m": 1.8, "height_m": 1.1, "depth_m": 2.1, "components": {"separated": True, "components": [{"id": "headboard", "name": "Headboard", "editable": True}]},
             "vendors": {"status": "approved"}},
            {"id": HIDDEN, "title": "Draft", "vendor_id": VENDOR, "status": "draft", "glb_url": f"{FILES}/{VENDOR}/x.glb", "vendors": {"status": "approved"}},
        ]}

    @staticmethod
    def _match(row, filters):
        for key, cond in filters:
            op, _, value = cond.partition(".")
            if op == "eq" and str(row.get(key)) != value:
                return False
            if op == "is" and value == "null" and row.get(key) is not None:
                return False
            if op == "not" and value.startswith("in.(") and str(row.get(key)) in value[4:-1].split(","):
                return False
        return True

    def _rpc(self, body):
        designs = [d for d in self.tables["designs"] if d["id"] == body["p_design"]]
        if not designs or designs[0]["owner_id"] != body["p_owner"] or designs[0]["deleted_at"]:
            return httpx.Response(400, json={"code": "P0002", "message": "Design not found"})
        d = designs[0]
        versions = self.tables["design_versions"]
        if body["p_client_token"]:
            same = [v for v in versions if v["design_id"] == d["id"] and v["client_token"] == body["p_client_token"]]
            if same:
                return httpx.Response(200, json=same[0])
        v = {"id": str(uuid.uuid4()), "design_id": d["id"], "number": d["version_count"] + 1, "schema": body["p_data"].get("schema"),
             "data": body["p_data"], "glb_url": body["p_glb_url"], "thumbnail_url": body["p_thumbnail_url"], "note": body["p_note"] or "",
             "restored_from": body["p_restored_from"], "client_token": body["p_client_token"], "created_by": body["p_owner"],
             "created_at": datetime.now(timezone.utc).isoformat()}
        versions.append(v)
        d.update(current_version_id=v["id"], version_count=v["number"], thumbnail_url=body["p_thumbnail_url"] or d["thumbnail_url"])
        return httpx.Response(200, json=v)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer srv"  # always the service key, never the user's token
        path = request.url.path.removeprefix("/rest/v1/")
        if path == "rpc/add_design_version":
            return self._rpc(json.loads(request.content))
        table = self.tables[path]
        filters = [(k, unquote(v)) for k, v in request.url.params.multi_items() if k not in {"select", "order", "limit", "offset"}]
        rows = [r for r in table if self._match(r, filters)]
        if request.method == "GET":
            if "current:design_versions" in request.url.params.get("select", ""):  # embedded latest version
                by_id = {v["id"]: v for v in self.tables["design_versions"]}
                rows = [{**r, "current": by_id.get(r["current_version_id"])} for r in rows]
            return httpx.Response(200, json=sorted(rows, key=lambda r: -r.get("number", 0)))
        if request.method == "POST":
            row = {"id": str(uuid.uuid4()), "deleted_at": None, "version_count": 0, "current_version_id": None, "thumbnail_url": None,
                   "source_model_id": None, "plan_id": None, "created_at": "now", "updated_at": "now", **json.loads(request.content)}
            table.append(row)
            return httpx.Response(201, json=[row])
        if request.method == "PATCH":
            for r in rows:
                r.update(json.loads(request.content))
            return httpx.Response(200, json=rows)
        if request.method == "DELETE":
            for r in rows:
                table.remove(r)
            return httpx.Response(200, json=rows)
        raise AssertionError(request.method)


@pytest.fixture
def env(monkeypatch):
    fake = FakeSupabase()
    settings = Settings(supabase_url=SB, supabase_service_role_key=SecretStr("srv"))
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    app.dependency_overrides[api.designs] = lambda: DesignStore(settings, http)
    app.dependency_overrides[api.plans] = lambda: PlanStore(settings, http)
    def user(request: Request) -> str:  # who is calling, per request (the real one reads the Supabase token)
        return request.headers.get("x-test-user", ALICE)

    app.dependency_overrides[api.current_user] = user
    client = TestClient(app)
    yield fake, client
    app.dependency_overrides.clear()


def _as(user):
    return {"x-test-user": user}


def test_item_design_keeps_the_catalogue_original_and_versions_its_edits(env):
    fake, client = env
    made = client.post("/api/designs/from-model", json={"model_id": MODEL}).json()
    assert made["kind"] == "item" and made["version_count"] == 1
    v1 = made["current"]["data"]["item"]
    assert v1["source"]["glb_url"].endswith(f"{VENDOR}/bed.glb") and v1["dimensions"] == {"width": 1.8, "height": 1.1, "depth": 2.1}

    edits = {
        "dimensions": {"width": 2.0, "height": 1.1, "depth": 2.1},
        "components": v1["components"],
        "material_edits": {"headboard": {"mapUrl": f"{FILES}/{VENDOR}/paints/pattern.png", "color": "#ffffff", "tile": 0.25, "mapping": "box"}},
    }
    token = str(uuid.uuid4())
    body = {"item": edits, "glb_url": f"{FILES}/{ALICE}/designs/{made['id']}/model/a.glb", "thumbnail_url": f"{FILES}/{ALICE}/designs/{made['id']}/thumb.webp", "note": "Wider", "client_token": token}
    v2 = client.post(f"/api/designs/{made['id']}/versions", json=body)
    assert v2.status_code == 201 and v2.json()["number"] == 2
    # A retried save (lost connection) returns the same version, not a third.
    assert client.post(f"/api/designs/{made['id']}/versions", json=body).json()["id"] == v2.json()["id"]
    # The vendor's listing is untouched, and the source cannot be swapped by a save.
    assert fake.tables["models"][0]["glb_url"].endswith("bed.glb")
    sneaky = {**body, "client_token": None, "item": {**edits, "source": {"model_id": HIDDEN, "title": "x", "glb_url": "https://evil/x.glb"}}}
    assert client.post(f"/api/designs/{made['id']}/versions", json=sneaky).status_code == 422
    saved = client.get(f"/api/designs/{made['id']}").json()
    assert saved["current"]["number"] == 2 and saved["current"]["data"]["item"]["source"]["model_id"] == MODEL
    assert saved["current"]["data"]["item"]["material_edits"]["headboard"]["tile"] == 0.25
    assert saved["thumbnail_url"].endswith("thumb.webp")
    assert [v["number"] for v in client.get(f"/api/designs/{made['id']}/versions").json()["versions"]] == [2]  # one version: the latest


def test_older_item_versions_are_gone_and_duplicate_is_independent(env):
    fake, client = env
    made = client.post("/api/designs/from-model", json={"model_id": MODEL, "title": "My bed"}).json()
    client.post(f"/api/designs/{made['id']}/versions", json={"item": {"dimensions": {"width": 2.2, "height": 1, "depth": 2}}, "note": "v2"})
    assert client.post(f"/api/designs/{made['id']}/restore", json={"number": 1}).status_code == 404  # replaced by the save

    copy = client.post(f"/api/designs/{made['id']}/duplicate", json={}).json()
    assert copy["id"] != made["id"] and copy["title"] == "My bed (copy)" and copy["version_count"] == 1
    assert copy["current"]["data"]["item"]["dimensions"]["width"] == 2.2
    client.post(f"/api/designs/{copy['id']}/versions", json={"item": {"dimensions": {"width": 3, "height": 1, "depth": 2}}})
    assert client.get(f"/api/designs/{made['id']}").json()["current"]["data"]["item"]["dimensions"]["width"] == 2.2  # the original did not move


def test_room_and_apartment_designs_snapshot_the_plan(env):
    fake, client = env
    one = {"id": str(uuid.uuid4()), "owner_id": ALICE, "name": "Bedroom", "source": "dimensions", "geometry": {"rooms": [{"id": "r1"}]}, "items": [{"instanceId": "bed"}], "camera": None, "warnings": []}
    two = {**one, "id": str(uuid.uuid4()), "name": "Flat", "geometry": {"rooms": [{"id": "a"}, {"id": "b"}]}}
    fake.tables["floor_plans"] += [one, two]
    room = client.post("/api/designs/from-plan", json={"plan_id": one["id"]}).json()
    flat = client.post("/api/designs/from-plan", json={"plan_id": two["id"]}).json()
    assert room["kind"] == "room" and flat["kind"] == "apartment"
    assert client.post("/api/designs/from-plan", json={"plan_id": two["id"], "kind": "room"}).status_code == 422

    one["items"] = [{"instanceId": "bed"}, {"instanceId": "desk"}]  # edited in the planner
    v2 = client.post(f"/api/designs/{room['id']}/versions", json={"note": "Desk"}).json()
    assert len(v2["data"]["plan"]["items"]) == 2
    # Restoring v1 puts it back into the planner's working plan as well.
    client.post(f"/api/designs/{room['id']}/restore", json={"number": 1})
    assert one["items"] == [{"instanceId": "bed"}]
    # A duplicated room has its own working plan.
    copy = client.post(f"/api/designs/{room['id']}/duplicate", json={}).json()
    assert copy["plan_id"] not in (one["id"], None)


def test_other_users_cannot_read_write_or_borrow(env):
    fake, client = env
    made = client.post("/api/designs/from-model", json={"model_id": MODEL}).json()
    for method, path, body in [
        ("GET", f"/api/designs/{made['id']}", None),
        ("GET", f"/api/designs/{made['id']}/versions", None),
        ("POST", f"/api/designs/{made['id']}/versions", {"item": {}}),
        ("POST", f"/api/designs/{made['id']}/restore", {"number": 1}),
        ("POST", f"/api/designs/{made['id']}/duplicate", {}),
        ("PATCH", f"/api/designs/{made['id']}", {"title": "mine now"}),
        ("DELETE", f"/api/designs/{made['id']}", None),
    ]:
        assert client.request(method, path, json=body, headers=_as(BOB)).status_code == 404, (method, path)
    assert client.get("/api/designs", headers=_as(BOB)).json()["designs"] == []
    # Files must be the saver's own uploads; textures must be uploaded (no blob: URLs).
    bad_glb = {"item": {}, "glb_url": f"{FILES}/{BOB}/x.glb"}
    assert client.post(f"/api/designs/{made['id']}/versions", json=bad_glb).status_code == 422
    blob = {"item": {"material_edits": {"headboard": {"mapUrl": "blob:http://localhost/abc"}}}}
    assert client.post(f"/api/designs/{made['id']}/versions", json=blob).status_code == 422
    # Unpublished listings cannot be customised; another user's plan cannot be saved as a design.
    assert client.post("/api/designs/from-model", json={"model_id": HIDDEN}).status_code == 404
    plan = {"id": str(uuid.uuid4()), "owner_id": ALICE, "name": "P", "geometry": {}, "items": []}
    fake.tables["floor_plans"].append(plan)
    assert client.post("/api/designs/from-plan", json={"plan_id": plan["id"]}, headers=_as(BOB)).status_code == 404


def test_delete_is_soft_and_ids_are_validated(env):
    fake, client = env
    made = client.post("/api/designs/from-model", json={"model_id": MODEL}).json()
    assert client.delete(f"/api/designs/{made['id']}").json() == {"status": "deleted"}
    assert client.get(f"/api/designs/{made['id']}").status_code == 404
    assert fake.tables["design_versions"]  # versions kept for what references them
    assert client.get("/api/designs/not-a-uuid").status_code == 422
    assert client.get("/api/designs?kind=garage").status_code == 422


def test_schema_rules():
    item = {"schema": "sha2ty.design/1", "kind": "item", "item": {"source": {"model_id": MODEL, "title": "t", "glb_url": "u"}}}
    assert DesignData.model_validate(item).dump()["units"] == "m"
    with pytest.raises(ValueError):
        DesignData.model_validate({**item, "plan": {"geometry": {}}})  # an item design has no plan
    with pytest.raises(ValueError):
        DesignData.model_validate({**item, "item": {**item["item"], "dimensions": {"width": 0, "height": 1, "depth": 1}}})
    with pytest.raises(ValueError):
        DesignData.model_validate({**item, "item": {**item["item"], "material_edits": {"p": {"color": "red"}}}})
    with pytest.raises(ValueError):
        DesignData.model_validate({**item, "schema": "sha2ty.design/9"})


def test_first_save_makes_the_design_with_its_edits_once_and_later_saves_add_versions(env):
    fake, client = env
    token = str(uuid.uuid4())
    first = {
        "model_id": MODEL, "title": "My Modern Bed", "client_token": token,
        "item": {"dimensions": {"width": 2.1, "height": 1.1, "depth": 2.1}, "resize_mode": "independent", "scale": [9, 9, 9],
                 "material_edits": {"headboard": {"mapUrl": f"{FILES}/{ALICE}/designs/t.png", "color": "#ffffff", "mapping": "box"}}},
        "glb_url": f"{FILES}/{ALICE}/designs/new/model/a.glb", "thumbnail_url": f"{FILES}/{ALICE}/designs/new/thumb.webp",
    }
    made = client.post("/api/designs/from-model", json=first)
    assert made.status_code == 201
    design = made.json()
    assert design["title"] == "My Modern Bed" and design["version_count"] == 1
    v1 = design["current"]["data"]["item"]
    assert v1["dimensions"]["width"] == 2.1 and v1["material_edits"]["headboard"]["mapping"] == "box"
    assert v1["scale"] == [round(2.1 / 1.8, 6), 1.0, 1.0]  # from the size, not the client's [9, 9, 9]
    assert design["current"]["glb_url"].endswith("a.glb") and design["thumbnail_url"].endswith("thumb.webp")
    # Double click / retry: the same design, not a second one.
    again = client.post("/api/designs/from-model", json=first).json()
    assert again["id"] == design["id"] and len(fake.tables["designs"]) == 1

    edits = {**first["item"], "material_edits": {"headboard": {"color": "#222222"}}}
    v2 = client.post(f"/api/designs/{design['id']}/versions", json={"item": edits, "client_token": str(uuid.uuid4())}).json()
    assert v2["number"] == 2
    # Saving the same state again adds nothing.
    same = client.post(f"/api/designs/{design['id']}/versions", json={"item": edits, "client_token": str(uuid.uuid4())}).json()
    assert same["unchanged"] is True and same["number"] == 2
    assert client.get(f"/api/designs/{design['id']}").json()["title"] == "My Modern Bed"  # name and id kept


def test_library_lists_own_customised_models_with_their_latest_version(env):
    fake, client = env
    bed = client.post("/api/designs/from-model", json={"model_id": MODEL, "title": "My Modern Bed", "item": {"dimensions": {"width": 2.1, "height": 1.1, "depth": 2.1}}}).json()
    client.post(f"/api/designs/{bed['id']}/versions", json={"item": {"dimensions": {"width": 2.2, "height": 1.1, "depth": 2.1}}})
    plan = {"id": str(uuid.uuid4()), "owner_id": ALICE, "name": "Flat", "geometry": {"rooms": [{}]}, "items": []}
    fake.tables["floor_plans"].append(plan)
    client.post("/api/designs/from-plan", json={"plan_id": plan["id"]})  # rooms are not furniture
    models = client.get("/api/designs/library").json()["models"]
    assert [m["title"] for m in models] == ["My Modern Bed"]
    assert models[0]["current"]["number"] == 2 and models[0]["current"]["data"]["item"]["dimensions"]["width"] == 2.2
    assert client.get("/api/designs/library", headers=_as(BOB)).json()["models"] == []


def test_a_customised_model_keeps_one_version_and_the_ones_requests_use(env):
    fake, client = env
    made = client.post("/api/designs/from-model", json={"model_id": MODEL, "title": "Bed"}).json()
    first = made["current"]["id"]
    for width in (2.0, 2.1, 2.2):
        client.post(f"/api/designs/{made['id']}/versions", json={"item": {"dimensions": {"width": width, "height": 1.1, "depth": 2.1}}})
    versions = [v for v in fake.tables["design_versions"] if v["design_id"] == made["id"]]
    assert len(versions) == 1 and versions[0]["data"]["item"]["dimensions"]["width"] == 2.2
    assert client.get(f"/api/designs/{made['id']}").json()["current"]["id"] == versions[0]["id"]

    # A version a request was made from stays, so the request still shows what was sent.
    fake.tables.setdefault("freelance_tasks", []).append({"id": "t1", "design_id": made["id"], "design_version_id": versions[0]["id"]})
    client.post(f"/api/designs/{made['id']}/versions", json={"item": {"dimensions": {"width": 2.3, "height": 1.1, "depth": 2.1}}})
    ids = {v["id"] for v in fake.tables["design_versions"] if v["design_id"] == made["id"]}
    assert versions[0]["id"] in ids and len(ids) == 2 and first not in ids


def test_a_customer_can_build_their_design_on_their_own_3d_file(env):
    fake, client = env
    made = client.post("/api/designs/from-model", json={"model_id": MODEL}).json()
    mine = f"{FILES}/{ALICE}/designs/{made['id']}/base/new.glb"
    saved = client.post(f"/api/designs/{made['id']}/versions", json={"item": {"base_glb_url": mine}})
    assert saved.status_code == 201 and saved.json()["data"]["item"]["base_glb_url"] == mine
    assert saved.json()["data"]["item"]["source"]["glb_url"].endswith("bed.glb")  # the listing it came from stays
    # Someone else's file (or any other URL) is refused.
    for url in (f"{FILES}/{BOB}/x.glb", "https://evil/x.glb"):
        assert client.post(f"/api/designs/{made['id']}/versions", json={"item": {"base_glb_url": url}}).status_code == 422
