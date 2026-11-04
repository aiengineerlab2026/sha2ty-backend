"""Vendors' proposed changes: only the vendor a request is open to proposes; only its customer answers; accepting
makes the proposal the design's newest version and the request's specification."""

from app.api import proposals
from tests.test_designs import ALICE, BOB, FILES, VENDOR
from tests.test_marketplace import _bed, env  # noqa: F401 (fixture)


def test_vendor_proposes_and_customer_accepts(env, monkeypatch):  # noqa: F811
    fake, client = env
    from app.api import designs as designs_api
    monkeypatch.setattr(proposals, "get_settings", designs_api.get_settings)
    fake.tables["task_proposals"] = []
    bed = _bed(client)
    task = client.post(f"/api/designs/{bed['id']}/request", json={"quantity": 2}).json()
    as_vendor = {"x-test-user": VENDOR}

    base = client.get(f"/api/tasks/{task['id']}/proposal-base", headers=as_vendor)
    assert base.status_code == 200, base.text
    item = {**base.json()["item"], "dimensions": {"width": 1.8, "height": 1.1, "depth": 2.0}}
    item.pop("source"), item.pop("scale", None)
    body = {"item": item, "glb_url": f"{FILES}/{VENDOR}/p/model.glb", "note": "1.8 m fits your room better"}
    assert client.get(f"/api/tasks/{task['id']}/proposal-base", headers={"x-test-user": BOB}).status_code == 404  # not a vendor
    assert client.post(f"/api/tasks/{task['id']}/proposals", headers=as_vendor, json={**body, "glb_url": f"{FILES}/{ALICE}/x.glb"}).status_code == 422
    sent = client.post(f"/api/tasks/{task['id']}/proposals", headers=as_vendor, json=body)
    assert sent.status_code == 201, sent.text
    proposal = sent.json()
    assert proposal["spec"]["dimensions"]["width"] == 1.8 and proposal["status"] == "pending"

    listed = client.get(f"/api/tasks/{task['id']}/proposals").json()
    assert [p["id"] for p in listed] == [proposal["id"]]
    assert client.post(f"/api/proposals/{proposal['id']}/decision", headers=as_vendor, json={"accept": True}).status_code == 404

    accepted = client.post(f"/api/proposals/{proposal['id']}/decision", json={"accept": True})
    assert accepted.status_code == 200, accepted.text
    row = next(t for t in fake.tables["freelance_tasks"] if t["id"] == task["id"])
    assert row["spec"]["design"]["dimensions"]["width"] == 1.8 and row["spec"]["quantity"] == 2
    assert row["design_version_id"] != task["design_version_id"]
    assert client.post(f"/api/proposals/{proposal['id']}/decision", json={"accept": False}).status_code == 409
