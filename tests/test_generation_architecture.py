"""Photo → 1 vision call (spec) → 1 image-to-3D call (visual) + deterministic structural model → editor.

Paid calls are mocked: the vision model answers with a recorded spec, Meshy / Astra return a small textured GLB.
What still needs real provider output (does the visual model really look like the photo?) is not judged here.
"""

import asyncio
import copy
import io
import json

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from app.api import vendor
from app.core import config
from app.core.config import Settings
from app.main import app
from app.services import fidelity, furniture_spec, material_spec, structural
from app.services.glb_size import read_glb, write_glb, _views

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64


def part(id_, name, box, cls="solid_wood", color="#8B5A2B", role="structural", **extra):
    return {"id": id_, "name": name, "nameAr": "", "shape": "", "position": "", "box": box, "dimensions": extra.pop("dimensions", None),
            "color": color, "material": extra.pop("material", cls.replace("_", " ")), "materialClass": cls, "finish": extra.pop("finish", "matte"),
            "texture": extra.pop("texture", ""), "role": role, "visibility": extra.pop("visibility", "visible"), "separateMesh": True,
            "relations": [], "motion": {"type": "none", "confidence": 0}, "confidence": extra.pop("confidence", 0.8), **extra}


def answer(object_type, components, size, loose=(), category="furniture", features=()):
    w, h, d = size
    return {
        "object_type": object_type, "matches_category": True, "category": category, "name": "Test", "description": "Test",
        "bbox": [0.1, 0.1, 0.9, 0.9], "orientation": "front", "proportions": None,
        "width_m": w, "depth_m": d, "height_m": h, "dimension_source": "inferred",
        "materials": [], "colors": [], "hidden": None, "confidence": 0.8, "warnings": [],
        "loose_objects": list(loose), "design_features": list(features),
        "structure": {"objectType": object_type, "summary": "", "components": components, "movable": False, "bodyId": components[0]["id"],
                      "interior": {"hasInterior": False, "confidence": 0, "items": []}},
    }


WARDROBE = answer("mirrored wardrobe", [
    part("body", "Wardrobe body", [0, 0, 1, 1], texture="oak wood grain", material="oak"),
    part("door_left", "Left mirrored door", [0.02, 0.03, 0.5, 0.7], cls="mirror", color="#E6ECEE"),
    part("door_right", "Right mirrored door", [0.5, 0.03, 0.98, 0.7], cls="mirror", color="#E6ECEE"),
    part("drawer_1", "Drawer 1", [0.02, 0.72, 0.98, 0.84], texture="oak wood grain", material="oak"),
    part("drawer_2", "Drawer 2", [0.02, 0.85, 0.98, 0.97], texture="oak wood grain", material="oak"),
    part("handle_1", "Left handle", [0.46, 0.35, 0.48, 0.42], cls="metal", color="#C9A227", finish="brushed"),
], (1.6, 2.1, 0.6), category="wardrobe", features=["mirrored doors", "oak grain"])

BED = answer("upholstered bed", [
    part("frame", "Bed frame", [0, 0, 1, 1], cls="solid_wood"),
    part("headboard", "Curved upholstered headboard", [0, 0, 1, 0.55], cls="fabric", color="#B9A48A", texture="velvet", shape="curved arch"),
    part("mattress", "Mattress", [0.03, 0.45, 0.97, 0.7], cls="fabric", color="#FFFFFF"),
    part("leg_1", "Front left leg", [0.02, 0.9, 0.06, 1], cls="solid_wood"),
    part("leg_2", "Front right leg", [0.94, 0.9, 0.98, 1], cls="solid_wood"),
], (1.7, 1.2, 2.1), features=["curved headboard"])

TABLE = answer("dining table", [
    part("top", "Tabletop", [0, 0, 1, 0.06], texture="walnut grain", material="walnut"),
    part("leg_1", "Left leg", [0.03, 0.06, 0.08, 1]),
    part("leg_2", "Right leg", [0.92, 0.06, 0.97, 1]),
    part("vase", "Vase", [0.45, 0, 0.55, 0.05], cls="other", role="decorative"),
], (1.8, 0.75, 0.9), loose=["vase", "books", "phone"], category="table")

CABINET = answer("display cabinet", [
    part("body", "Cabinet body", [0, 0, 1, 1], cls="painted_wood", color="#F2EFE8", material="painted MDF"),
    part("door_glass", "Glass door", [0.03, 0.05, 0.97, 0.6], cls="glass", color="#DDE8EA"),
    part("shelf_1", "Shelf 1", [0.03, 0.3, 0.97, 0.32], cls="painted_wood", color="#F2EFE8", dimensions={"width": 0.94, "height": 0.02, "depth": 0.9}),
    part("drawer_1", "Drawer", [0.03, 0.65, 0.97, 0.95], cls="painted_wood", color="#F2EFE8"),
], (0.9, 1.8, 0.45), loose=["vases", "plates", "books on the shelves"], category="furniture")

MIXED = answer("sideboard", [
    part("top", "Marble top", [0, 0, 1, 0.05], cls="stone", color="#EDEDED"),
    part("door_1", "Fabric door", [0.02, 0.07, 0.49, 0.8], cls="fabric", color="#556B2F"),
    part("door_2", "Glass door", [0.51, 0.07, 0.98, 0.8], cls="glass", color="#DDE8EA"),
    part("leg_1", "Metal leg", [0.02, 0.82, 0.05, 1], cls="metal", color="#222222"),
    part("leg_2", "Metal leg right", [0.95, 0.82, 0.98, 1], cls="metal", color="#222222"),
], (1.6, 0.8, 0.45))


def built(spec_json, category="furniture", kind=None):
    spec = furniture_spec.parse(json.dumps(spec_json), "en")
    return structural.build(spec.details.model_dump(mode="json"), spec.analysis.model_dump(mode="json"), category, kind)


def names(report):
    return [c["name"] for c in report["components"]]


# ---- 1. the one vision call ----


def test_one_vision_answer_gives_details_components_and_loose_objects():
    spec = furniture_spec.parse(json.dumps(TABLE), "en")
    assert spec.details.object_type == "dining table" and spec.details.width_m == 1.8
    assert [c.id for c in spec.analysis.components] == ["top", "leg_1", "leg_2", "vase"]
    assert spec.needs_cleaning and spec.loose_objects == ["vase", "books", "phone"]
    assert [c.name for c in spec.details.components] == ["Tabletop", "Left leg", "Right leg"]  # decorative left out
    assert not furniture_spec.parse(json.dumps(WARDROBE), "en").needs_cleaning
    assert furniture_spec.parse("not json") is None


def test_the_spec_is_read_once_and_cached(tmp_path, monkeypatch):
    calls = []

    async def ask(content, schema, name, *_args):
        calls.append(name)
        assert "structure" in schema["properties"] and "loose_objects" in schema["properties"]
        return json.dumps(WARDROBE)

    monkeypatch.setattr(furniture_spec.vision, "ask_json", ask)
    s = Settings(gemini_api_key="gem-test", storage_dir=tmp_path)

    async def twice():
        a = await furniture_spec.read(b"photo", "image/png", "wardrobe", "", s, None, "gemini-3.1-pro-preview", lang="en")
        b = await furniture_spec.read(b"photo", "image/png", "wardrobe", "", s, None, "gemini-3.1-pro-preview", lang="en")
        return a, b

    a, b = asyncio.run(twice())
    assert calls == ["furniture_spec"] and a.model_dump() == b.model_dump()


# ---- 2. structural models from the spec, per furniture type ----


def test_wardrobe_with_mirrored_doors_wood_grain_and_drawers():
    glb, report = built(WARDROBE, "wardrobe")
    assert report["built"] and glb
    n = names(report)
    for want in ("Left mirrored door", "Right mirrored door", "Drawer 1", "Drawer 2", "Left handle", "Left Side Panel", "Right Side Panel", "Top Panel", "Back Panel"):
        assert want in n
    assert "Wardrobe body" not in n  # the whole body is its panels, not one block
    by = {c["name"]: c for c in report["components"]}
    assert by["Left mirrored door"]["materialClass"] == "mirror" and by["Left Side Panel"]["materialClass"] == "solid_wood"
    assert by["Left Side Panel"]["source"] == "inferred" and by["Left Side Panel"]["sizeCm"]["height"] == 210.0  # full height
    assert by["Drawer 1"]["specId"] == "drawer_1" and by["Left Side Panel"]["specId"] == "body"  # linked to the spec
    assert report["materials"]["drawer_1"]["woodType"] == "oak"
    assert report["checks"]["closed"] and not [w for w in report["checks"]["warnings"] if "same space" in w]
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    assert len(scene.geometry) == len(report["components"]) and all(g.is_watertight for g in scene.geometry.values())
    doc, _ = read_glb(glb)
    mirror = next(m for m in doc["materials"] if m["name"] == "Left mirrored door")
    assert mirror["pbrMetallicRoughness"]["metallicFactor"] == 1.0 and mirror["pbrMetallicRoughness"]["roughnessFactor"] < 0.1


def test_bed_keeps_its_curved_upholstered_headboard_and_drops_the_mattress():
    _, report = built(BED, "furniture", "bed")
    n = names(report)
    assert "Curved upholstered headboard" in n and "Mattress" not in n
    assert any(e["name"] == "Mattress" for e in report["excluded"])
    head = next(c for c in report["components"] if c["name"] == "Curved upholstered headboard")
    assert head["materialClass"] == "fabric" and head["positionCm"]["z"] < 10  # at the back
    assert {"Left Side Panel", "Right Side Panel", "Footboard"} <= set(n)  # the frame as its rails, inferred


def test_table_with_objects_on_it_builds_only_the_table():
    _, report = built(TABLE, "table")
    n = names(report)
    assert set(n) == {"Tabletop", "Left leg", "Right leg"} and report["excludedLoose"] == ["Vase"]
    top = next(c for c in report["components"] if c["name"] == "Tabletop")
    assert top["sizeCm"]["width"] == 180.0 and top["materialClass"] == "solid_wood"


def test_cabinet_with_accessories_keeps_its_glass_door_shelf_and_drawer():
    _, report = built(CABINET, "furniture", "cabinet")
    by = {c["name"]: c for c in report["components"]}
    assert by["Glass door"]["materialClass"] == "glass" and by["Shelf 1"]["source"] == "observed"
    assert by["Drawer"]["materialClass"] == "painted_wood" and "Back Panel" in by


def test_multiple_materials_each_get_their_own_material():
    glb, report = built(MIXED)
    doc, _ = read_glb(glb)
    classes = {m["extras"]["materialClass"] for m in doc["materials"] if "extras" in m}
    assert {"stone", "fabric", "glass", "metal"} <= classes
    glass = next(m for m in doc["materials"] if m["name"] == "Glass door")
    assert glass["alphaMode"] == "BLEND" and "KHR_materials_transmission" in doc["extensionsUsed"]
    assert all(c.get("estimated") for c in report["components"])  # never photographic: estimated


def test_too_little_structure_is_flagged_not_guessed():
    one = answer("lamp", [part("shade", "Shade", [0, 0, 1, 0.5], cls="fabric")], (0.4, 1.5, 0.4))
    glb, report = built(one)
    assert glb is None and not report["built"] and any("too little structure" in u for u in report["uncertain"])
    no_size = copy.deepcopy(WARDROBE)
    no_size.update(width_m=None, height_m=None, depth_m=None, proportions=None)
    glb, report = built(no_size, "wardrobe")
    assert glb is None and any("could not be read" in u for u in report["uncertain"])
    by_shape = copy.deepcopy(WARDROBE)
    by_shape.update(width_m=None, height_m=None, depth_m=None, proportions=[0.76, 0.29, 1.0])
    glb, report = built(by_shape, "wardrobe")
    assert glb and not report["scaleKnown"] and any("Scale unknown" in u for u in report["uncertain"])


def test_material_classes_from_words_when_the_reading_gave_none():
    assert material_spec.material_class("brushed brass") == "metal"
    assert material_spec.material_class("زجاج") == "glass"
    assert material_spec.entry({"material": "walnut veneer", "materialClass": "other"})["woodType"] == "walnut"


# ---- 3. fidelity at each stage ----


def textured_glb(colour=(200, 40, 40)) -> bytes:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    image = Image.new("RGB", (64, 64), colour)
    mesh = trimesh.Trimesh(verts, faces, visual=trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=image)))
    return trimesh.Scene(mesh).export(file_type="glb")


def strip_textures(glb: bytes) -> bytes:
    doc, binary = read_glb(glb)
    views = _views(doc, binary)
    for m in doc["materials"]:
        m["pbrMetallicRoughness"].pop("baseColorTexture", None)
    return write_glb(doc, views)


def big_photo(colour) -> bytes:
    """A usable cleaned image (Image to Image output is at least 256 px): the product in its colour on Meshy's neutral
    gray backdrop, same 1:1 framing."""
    buf = io.BytesIO()
    img = Image.new("RGB", (400, 400), (128, 128, 128))
    img.paste(colour, (60, 40, 340, 380))
    img.save(buf, "PNG")
    return buf.getvalue()


def photo(colour) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), colour).save(buf, "PNG")
    return buf.getvalue()


def test_processing_that_keeps_everything_passes_and_lost_textures_are_found():
    raw = textured_glb()
    same = fidelity.compare_glbs(raw, raw)
    assert all(s["ok"] for s in same)
    lost = {s["stage"]: s for s in fidelity.compare_glbs(raw, strip_textures(raw))}
    assert lost["materials"]["ok"] is False and any("baseColor map" in i for i in lost["materials"]["issues"])


def test_cleaning_that_recolours_the_product_is_where_fidelity_was_lost():
    red, blue = photo((200, 40, 40)), photo((40, 40, 200))
    assert fidelity.compare_images(red, red, [0.1, 0.1, 0.9, 0.9])["ok"]
    stage = fidelity.compare_images(red, blue, [0.1, 0.1, 0.9, 0.9])
    assert stage["ok"] is False
    result = fidelity.report([stage, fidelity.provider(textured_glb(), red, None)])
    assert result["quality"] == "low" and result["lostAt"] == "cleaning"


def test_provider_model_in_the_wrong_colour_is_flagged():
    assert fidelity.provider(textured_glb((200, 40, 40)), photo((200, 40, 40)), None)["ok"]
    assert fidelity.provider(textured_glb((40, 200, 40)), photo((200, 40, 40)), None)["ok"] is False


# ---- 4. the pipeline: one vision call, one provider call, the structural model beside it ----


class FakeCdn:
    def __init__(self):
        self.uploads = {}

    def upload(self, path, data, _type):
        self.uploads[path] = data
        return f"https://cdn.test/{path}"

    def public_url(self, path):
        return f"https://cdn.test/{path}"


class FakeMeshy:
    """Meshy with every paid call mocked: Image to Image returns `cleaned`, Image to 3D the GLB."""

    def __init__(self, glb, cleaned: bytes | None = None):
        self.glb, self.calls, self.cleaned = glb, [], cleaned

    async def create_image_to_image_task(self, prompt, **kwargs):
        self.calls.append(("image-to-image", {"prompt": prompt, **kwargs}))
        return "img-1"

    async def get_image_to_image_task(self, _id):
        return {"status": "SUCCEEDED", "progress": 100, "image_urls": ["https://assets.meshy.test/clean.png?Signature=secret"]}

    async def download_bytes(self, _url):
        return self.cleaned

    async def create_image_to_3d_task(self, _options, **kwargs):
        self.calls.append(("image-to-3d", kwargs))
        return "mesh-1"

    async def get_image_to_3d_task(self, _id):
        return {"status": "SUCCEEDED", "progress": 100, "model_urls": {"glb": "https://meshy.test/m.glb"}}, None

    async def create_split_task(self, *_a):
        raise AssertionError("Auto Split is off in this test")

    async def download_file(self, _url, dest):
        dest.write_bytes(self.glb)
        return len(self.glb)


def run_meshy(tmp_path, monkeypatch, spec_json, **settings):
    vision_calls, cleaned = [], []
    raw = textured_glb((139, 90, 43))

    async def ask(content, schema, name, *_args):
        vision_calls.append(name)
        return json.dumps(spec_json)

    async def clean(prompt, *_args):
        cleaned.append(prompt)
        return photo((139, 90, 43))

    async def one_try(*_args):
        return 4

    async def quiet(*_args, **_kwargs):
        return None

    monkeypatch.setattr(vendor, "read_furniture_spec", furniture_spec.read)  # the real reader, its model mocked
    monkeypatch.setattr(vendor.vision, "ask_json", ask)
    monkeypatch.setattr(vendor.vision, "clean_image", clean)
    monkeypatch.setattr(vendor, "take_try", one_try)
    monkeypatch.setattr(vendor.bus, "publish", quiet)
    meshy, cdn = FakeMeshy(raw, cleaned=big_photo((139, 90, 43))), FakeCdn()
    s = Settings(meshy_api_key="msy-test", gemini_api_key="gem-test", storage_dir=tmp_path, meshy_auto_split=False, **settings)  # the minimum-cost path
    monkeypatch.setattr(vendor, "get_settings", lambda: s)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: meshy, vendor.get_cdn: lambda: cdn, config.get_settings: lambda: s})
    try:
        with TestClient(app) as client:
            first = client.post("/api/vendor/generate", files={"image": ("p.png", photo((139, 90, 43)), "image/png")}, data={"category": spec_json["category"], "lang": "en"}).json()
            done = first
            for _ in range(8):
                done = client.get(f"/api/vendor/tasks/{first['task_id']}").json()
                if done["status"] in ("SUCCEEDED", "FAILED"):
                    break
    finally:
        app.dependency_overrides.clear()
    return done, vision_calls, cleaned, meshy, cdn, raw


def test_meshy_makes_no_vision_call_and_cleans_with_meshy(tmp_path, monkeypatch):
    done, vision_calls, cleaned, meshy, cdn, raw = run_meshy(tmp_path, monkeypatch, WARDROBE)
    assert done["status"] == "SUCCEEDED" and done["phase"] == "completed"
    assert vision_calls == [] and cleaned == []  # no Gemini / OpenAI call anywhere on the Meshy path
    assert [c[0] for c in meshy.calls] == ["image-to-image", "image-to-3d"]
    assert meshy.calls[1][1]["input_task_id"] == "img-1"  # Image to 3D from the cleaned image's task
    doc = done["components"]
    assert doc["cleaning"]["used"] == "cleaned" and doc["cleaning"]["provider"] == "meshy-image-to-image"
    assert not doc["structural"]["built"] and "No structural specification" in doc["structural"]["uncertain"][0] and "carpenter" not in doc
    assert cdn.uploads["vendor-1/" + doc["raw_glb_url"].rsplit("/", 1)[1]] == raw  # kept byte for byte


def test_the_optional_analysis_reads_once_and_still_cleans_with_meshy(tmp_path, monkeypatch):
    done, vision_calls, cleaned, meshy, cdn, raw = run_meshy(tmp_path, monkeypatch, WARDROBE, meshy_vision_analysis=True)
    assert done["status"] == "SUCCEEDED"
    assert vision_calls == ["furniture_spec"] and cleaned == []  # read once; the vision model never cleans for Meshy
    assert [c[0] for c in meshy.calls] == ["image-to-image", "image-to-3d"]
    doc = done["components"]
    assert doc["structural"]["built"] and doc["carpenter"]["url"].endswith(".carpenter.glb")
    assert cdn.uploads["vendor-1/" + doc["raw_glb_url"].rsplit("/", 1)[1]] == raw  # kept byte for byte
    assert doc["materials"]["parts"]["door_left"]["class"] == "mirror"
    stages = {s["stage"]: s for s in doc["fidelity"]["stages"]}
    assert set(stages) == {"cleaning", "provider", "processing", "materials", "structural"} and stages["materials"]["ok"]


def test_with_the_optional_analysis_the_objects_it_saw_go_into_the_meshy_prompt(tmp_path, monkeypatch):
    done, vision_calls, cleaned, meshy, _cdn, _raw = run_meshy(tmp_path, monkeypatch, TABLE, meshy_vision_analysis=True)
    assert done["status"] == "SUCCEEDED" and vision_calls == ["furniture_spec"] and cleaned == []
    assert meshy.calls[0][0] == "image-to-image" and "vase, books, phone" in meshy.calls[0][1]["prompt"]
    assert meshy.calls[1][1]["input_task_id"] == "img-1"  # Image to 3D got the cleaned image
    assert "cleaning" in {s["stage"] for s in done["components"]["fidelity"]["stages"]}


def test_a_single_photo_never_falls_back_to_the_procedural_builder():
    assert not vendor.builds_wardrobe("astra", "wardrobe", "", False)
    assert not vendor.builds_wardrobe("meshy", "wardrobe", "", False)
    assert vendor.builds_wardrobe("astra", "wardrobe", "", True)  # open + closed photos: the states builder


def test_the_structural_model_is_rebuilt_from_an_edited_spec_without_any_ai(monkeypatch):
    async def no_vision(*_args, **_kwargs):
        raise AssertionError("no vision call")

    monkeypatch.setattr(vendor.vision, "ask_json", no_vision)
    cdn = FakeCdn()
    spec = furniture_spec.parse(json.dumps(TABLE), "en")
    edited = spec.analysis.model_dump(mode="json")
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_cdn: lambda: cdn})
    try:
        with TestClient(app) as client:
            body = {"model_id": "m1", "details": {"width_m": 2.2, "height_m": 0.75, "depth_m": 1.0}, "analysis": edited, "category": "table"}
            res = client.post("/api/vendor/structural", json=body).json()
            bad = client.post("/api/vendor/structural", json={**body, "analysis": {"components": []}})
    finally:
        app.dependency_overrides.clear()
    top = next(c for c in res["carpenter"]["components"] if c["name"] == "Tabletop")
    assert top["sizeCm"]["width"] == 220.0 and res["carpenter"]["url"].startswith("https://cdn.test/vendor-1/m1.carpenter.")
    assert bad.status_code == 422


@pytest.mark.parametrize("spec_json", [WARDROBE, BED, TABLE, CABINET, MIXED])
def test_every_structural_part_is_independently_editable(spec_json):
    glb, report = built(spec_json, spec_json["category"], spec_json["object_type"].split()[-1])
    doc, _ = read_glb(glb)
    ids = [n["extras"]["partId"] for n in doc["nodes"] if "extras" in n]
    assert len(ids) == len(set(ids)) == len(report["components"])  # one mesh, one id, per part
    assert all(c["sizeCm"]["width"] >= 1 and c["sizeCm"]["height"] >= 1 for c in report["components"])  # no slivers
    assert report["fabricationReady"] is False
