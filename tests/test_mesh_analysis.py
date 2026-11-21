"""Vision mesh analysis: validated component JSON → the Meshy prompt (each part where Meshy accepts it) → the
generated meshes matched to the components, geometry untouched."""

from __future__ import annotations

import asyncio
import copy
import json

from fastapi.testclient import TestClient

from app.api import vendor
from app.core import config
from app.core.config import Settings
from app.main import app
from app.services import mesh_analysis as ma

WARDROBE = {
    "objectType": "wardrobe",
    "summary": "A two-door wardrobe on a plinth with two drawers below.",
    "components": [
        {"id": "Body", "name": "Wardrobe body", "nameAr": "هيكل الدولاب", "shape": "rectangular cabinet", "position": "main structure",
         "box": [0, 0, 1, 1], "color": "f5f5f0", "material": "painted wood", "texture": "matte", "role": "structural",
         "visibility": "visible", "separateMesh": True, "relations": []},
        {"id": "door_left", "name": "Left door", "nameAr": "الباب الأيسر", "shape": "flat panel", "position": "front left",
         "box": [0.02, 0.05, 0.5, 0.7], "color": "#FFFFFF", "material": "painted wood", "texture": "", "role": "functional",
         "visibility": "visible", "separateMesh": True, "relations": [{"type": "attached_to", "target": "body"}, {"type": "on", "target": "ghost"}]},
        {"id": "door_right", "name": "Right door", "nameAr": "الباب الأيمن", "shape": "flat panel", "position": "front right",
         "box": [0.5, 0.05, 0.98, 0.7], "color": "#FFFFFF", "material": "painted wood", "texture": "", "role": "functional",
         "visibility": "visible", "separateMesh": True, "relations": [{"type": "attached_to", "target": "body"}]},
        {"id": "door_right", "name": "Back panel", "nameAr": "اللوح الخلفي", "shape": "flat panel", "position": "back",
         "box": None, "color": "not a colour", "material": "hdf", "texture": "", "role": "structural",
         "visibility": "inferred", "separateMesh": False, "relations": []},
        {"id": "x", "name": "", "nameAr": "", "shape": "", "position": "", "box": None, "color": None, "material": "", "texture": "",
         "role": "decorative", "visibility": "visible", "separateMesh": False, "relations": []},
    ],
}


def mesh(id_, name, lo, hi):
    return {"id": id_, "name": name, "mesh": 0, "primitive": 0, "material": 0, "node": 0, "box": [lo, hi], "editable": True}


def test_the_answer_is_validated_and_cleaned():
    a = ma.parse(json.dumps(WARDROBE))
    assert [c.id for c in a.components] == ["body", "door_left", "door_right", "door_right_2"]  # unique; the unnamed one dropped
    assert a.components[0].color == "#F5F5F0" and a.components[3].color is None  # hex normalised, nonsense dropped
    assert [r.target for r in a.components[1].relations] == ["body"]  # relations only to listed parts
    assert a.components[3].visibility == "inferred"
    assert ma.parse("not json") is None and ma.parse(json.dumps({"objectType": "x", "components": []})) is None


def test_the_prompt_goes_where_meshy_accepts_it():
    a = ma.parse(json.dumps(WARDROBE))
    prompt = ma.meshy_prompt(a)
    assert prompt.startswith("Generate a realistic modular 3D wardrobe.") and "left door attached to wardrobe body" in prompt
    assert "Model these as independent meshes: wardrobe body, left door, right door." in prompt  # not the inferred back
    texture = ma.texture_prompt(a, "Base brief. " * 200)
    assert len(texture) <= ma.TEXTURE_PROMPT_MAX and "left door: #FFFFFF painted wood" in texture
    from app.services.generation_prompt import meshy_texture_prompt
    kept = ma.texture_prompt(a, meshy_texture_prompt("wardrobe") + (" Extra colour note." * 80))
    assert kept.startswith("Carpenter-ready modular wardrobe only.") and "reflective mirrors exactly as in the reference image." in kept
    assert len(kept) <= ma.TEXTURE_PROMPT_MAX
    assert ma.split_prompt(a) == "wardrobe body, left door, right door"
    many = copy.deepcopy(WARDROBE)
    many["components"] = [{**WARDROBE["components"][1], "id": f"drawer_{i}", "name": f"Drawer {i}"} for i in range(14)] + WARDROBE["components"][:1]
    split = ma.split_prompt(ma.parse(json.dumps(many)))
    assert split == "drawers, wardrobe body" and len(split.split(", ")) <= ma.SPLIT_MAX_PARTS  # Meshy reads 1–10 names
    record = ma.conditioning(prompt, texture, "a, b", split_on=False)
    assert record["prompt"] == prompt and "auto_split.prompt" not in record["sent"]
    assert any("no prompt for geometry" in n for n in record["notSent"]) and any("Auto Split is turned off" in n for n in record["notSent"])


def test_meshes_take_the_components_names_and_keep_their_geometry():
    a = ma.parse(json.dumps(WARDROBE))
    meshes = [
        mesh("part_1", "Part 1", [0, 0, 0], [2, 2, 0.6]),  # the whole cabinet
        mesh("part_2", "Part 2", [0.04, 0.6, 0.6], [1, 1.9, 0.62]),  # left door (y up: high in the model = top of the photo)
        mesh("part_3", "Part 3", [1, 0.6, 0.6], [1.96, 1.9, 0.62]),
    ]
    doc = {"separated": True, "components": copy.deepcopy(meshes)}
    out = ma.match(doc, a)
    assert [(c["name"], c["nameAr"], c["analysisId"]) for c in out["components"]] == [
        ("Wardrobe body", "هيكل الدولاب", "body"), ("Left door", "الباب الأيسر", "door_left"), ("Right door", "الباب الأيمن", "door_right")]
    for before, after in zip(meshes, out["components"]):  # ids, boxes, mesh / material indexes untouched
        assert {k: after[k] for k in before if k != "name"} == {k: v for k, v in before.items() if k != "name"}
    assert out["modularity"]["level"] == "full" and out["analysis"]["objectType"] == "wardrobe"


def test_a_model_missing_parts_is_partially_modular_and_a_fused_one_is_not_modular():
    a = ma.parse(json.dumps(WARDROBE))
    doc = {"separated": True, "components": [mesh("p1", "Part 1", [0, 0, 0], [2, 2, 0.6]), mesh("p2", "Part 2", [0.9, 0, 0], [1.1, 0.1, 0.6])]}
    out = ma.match(doc, a)
    assert out["modularity"]["level"] == "partial" and set(out["modularity"]["missing"]) == {"door_left", "door_right"}
    assert out["modularity"]["unmatchedMeshes"] == ["p2"] and out["components"][1]["name"] == "Part 2"  # never forced
    fused = ma.match({"separated": False, "components": [mesh("whole", "Whole model", [0, 0, 0], [2, 2, 0.6])]}, a)
    assert fused["modularity"]["level"] == "none" and fused["components"][0]["name"] == "Whole model"
    assert ma.match(doc, None) is doc  # no analysis: the document as it was


def test_auto_split_part_names_match_by_name():
    a = ma.parse(json.dumps(WARDROBE))
    doc = {"separated": True, "components": [mesh("a", "right door", [5, 5, 5], [6, 6, 6]), mesh("b", "Left Door", [7, 7, 7], [8, 8, 8]), mesh("c", "wardrobe_body", [0, 0, 0], [9, 9, 9])]}
    out = ma.match(doc, a)
    assert [c["analysisId"] for c in out["components"]] == ["door_right", "door_left", "body"] and out["modularity"]["level"] == "full"


def test_analyze_asks_with_the_schema_and_survives_a_failure():
    seen = {}

    async def ask(prompt, schema):
        seen["prompt"], seen["schema"] = prompt, schema
        return json.dumps(WARDROBE)

    a = asyncio.run(ma.analyze(ask, "wardrobe (kids)", "kids wardrobe"))
    assert a.objectType == "wardrobe" and "wardrobe (kids)" in seen["prompt"] and seen["schema"] is ma.SCHEMA

    async def broken(_prompt, _schema):
        raise RuntimeError("vision down")

    assert asyncio.run(ma.analyze(broken, "wardrobe")) is None


def test_generate_sends_the_analysis_to_meshy_and_stores_it_with_the_model(tmp_path, monkeypatch):
    calls = {}
    parts = [mesh("p1", "Part 1", [0, 0, 0], [2, 2, 0.6]), mesh("p2", "Part 2", [0.04, 0.6, 0.6], [1, 1.9, 0.62]), mesh("p3", "Part 3", [1, 0.6, 0.6], [1.96, 1.9, 0.62])]

    async def analysed(*_args, **_kwargs):
        return ma.parse(json.dumps(WARDROBE))

    class FakeMeshy:
        async def create_image_to_3d_task(self, _options, **kwargs):
            calls["texture_prompt"] = kwargs["texture_prompt"]
            return "task-1"

        async def get_image_to_3d_task(self, _task_id):
            return {"status": "SUCCEEDED", "progress": 100, "model_urls": {"glb": "https://meshy.test/m.glb"}}, None

        async def create_split_task(self, _id, prompt, _keep):
            calls["split"] = prompt
            return "split-1"

        async def get_split_task(self, _id):
            return {"status": "SUCCEEDED", "model_urls": {"glb": "https://meshy.test/s.glb"}}

        async def download_file(self, _url, dest):
            dest.write_bytes(b"glTF")

    class FakeCdn:
        def upload(self, path, _data, _type):
            return f"https://cdn.test/{path}"

        def public_url(self, path):  # the original's address is known before its upload (after the response)
            return f"https://cdn.test/{path}"

    async def one_try(*_args):
        return 4

    async def passthrough(_request, glb, doc, *_args):
        return glb, doc

    async def unchanged(glb, source, doc):
        return glb, source, doc

    monkeypatch.setattr(vendor, "analysed", analysed)
    monkeypatch.setattr(vendor, "read_photo", lambda *a, **k: asyncio.sleep(0, None))
    monkeypatch.setattr(vendor, "take_try", one_try)
    monkeypatch.setattr(vendor, "inspect_glb", lambda *_a: {"separated": True, "components": copy.deepcopy(parts)})
    monkeypatch.setattr(vendor, "named", passthrough)
    monkeypatch.setattr(vendor, "sized", unchanged)
    monkeypatch.setattr(vendor, "validated", lambda glb, doc, *_a: doc)
    s = Settings(meshy_api_key="msy-test", meshy_clean_image=False, storage_dir=tmp_path)
    monkeypatch.setattr(vendor, "get_settings", lambda: s)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: FakeMeshy(), vendor.get_cdn: lambda: FakeCdn(), config.get_settings: lambda: s})
    try:
        with TestClient(app) as client:
            png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
            first = client.post("/api/vendor/generate", files={"image": ("w.png", png, "image/png")}, data={"category": "furniture", "kind": "sofa"}).json()
            done = client.get(f"/api/vendor/tasks/{first['task_id']}").json()
    finally:
        app.dependency_overrides.clear()
    assert "left door: #FFFFFF painted wood" in calls["texture_prompt"] and calls["split"] == "wardrobe body, left door, right door"
    doc = done["components"]
    assert done["status"] == "SUCCEEDED" and doc["method"] == "meshy_auto_split" and doc["modularity"]["level"] == "full"
    assert [c["nameAr"] for c in doc["components"]] == ["هيكل الدولاب", "الباب الأيسر", "الباب الأيمن"]
    assert doc["analysis"]["components"][0]["id"] == "body" and doc["conditioning"]["prompt"].startswith("Generate a realistic modular 3D wardrobe")
    assert done["source_glb_url"].endswith(".source.glb")  # the original Meshy GLB kept as the fallback
