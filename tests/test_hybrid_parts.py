"""Phase 2–5 of the hybrid engine on real GLB bytes: solid decoration geometry (solids), photo processing
(photo_tools), validation and repair of the built file (model_validation), and complex components generated one by one
(api/hybrid_parts) through the real endpoint with Meshy mocked. Paid APIs are mocked; every model is a real GLB."""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from app.api import vendor
from app.core import config
from app.core.config import Settings
from app.main import app
from app.services import furniture_spec, hybrid, model_validation, photo_tools, solids
from app.services.glb_size import _views, read_glb, write_glb
from tests import hybrid_fixtures as F


def gltf(glb: bytes) -> dict:
    n = struct.unpack_from("<I", glb, 12)[0]
    return json.loads(glb[20:20 + n])


def nodes_by_part(glb: bytes) -> dict[str, dict]:
    doc = gltf(glb)
    root = doc["scenes"][0]["nodes"][0]
    linked = set(doc["nodes"][root].get("children") or [])
    return {n["extras"]["partId"]: n for i, n in enumerate(doc["nodes"]) if i in linked and (n.get("extras") or {}).get("partId")}


# ---- solids: decoration is geometry, every part a closed solid ----


@pytest.mark.parametrize(("decoration", "more"), [("grooved", False), ("fluted", True), ("recessed_panel", False), ("raised_panel", True), ("arched", False)])
def test_every_decoration_is_a_closed_solid_cut_into_or_added_onto_the_board(decoration, more):
    lo, hi = [0, 0, 0], [0.4, 1.8, 0.018]
    plain = solids.box(lo, hi).volume()
    solid, made = solids.decorated(lo, hi, decoration)
    mesh = solid.to_mesh()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts))
    assert tm.is_watertight and made["method"] == "geometry"
    chamfered = solids.chamfer(lo, hi).volume()
    assert (solid.volume() > chamfered) == more and abs(solid.volume() - chamfered) > 1e-6  # material really added / removed
    assert solid.volume() < plain * 2


def test_counts_come_from_the_photo_when_it_shows_them():
    _, made = solids.decorated([0, 0, 0], [0.9, 0.2, 0.018], "fluted", {"count": 16, "axis": "vertical"})
    assert made["count"] == 16 and made["fromPhoto"]


def test_a_turned_leg_is_revolved_and_closed():
    leg = solids.revolve([(0, 0), (0.02, 0), (0.025, 0.1), (0.015, 0.3), (0.02, 0.4), (0, 0.4)])
    assert leg.status().name == "NoError" and leg.volume() > 0


# ---- photo tools: the original's pixels, never invented ----


def photo_rgb():
    return photo_tools.to_array(Image.open(io.BytesIO(F.photo())))


def test_the_silhouette_keeps_the_furniture_and_drops_the_wall():
    rgb = photo_rgb()
    wall = np.abs(rgb.astype(int) - [120, 130, 140]).sum(-1) < 10
    mask = photo_tools.silhouette(rgb, F.OBJ) > 0
    assert (mask & ~wall).sum() / (~wall).sum() > 0.97 and (mask & wall).sum() / wall.sum() < 0.02


def test_repeated_lines_are_counted_from_the_photo_and_a_plain_door_has_none():
    rgb = photo_rgb()
    H, W = rgb.shape[:2]
    X = lambda f: int((F.OBJ[0] + f * (F.OBJ[2] - F.OBJ[0])) * W)  # noqa: E731
    Y = lambda f: int((F.OBJ[1] + f * (F.OBJ[3] - F.OBJ[1])) * H)  # noqa: E731
    assert photo_tools.pattern(rgb[Y(0.84):Y(0.98), X(0.02):X(0.98)]) == {"axis": "vertical", "count": 16, "regularity": pytest.approx(0.98, abs=0.03)}
    assert photo_tools.pattern(rgb[Y(0.03):Y(0.82), X(0.02):X(0.255)]) is None


def test_a_part_photographed_at_an_angle_is_straightened():
    import cv2

    img = np.full((400, 400, 3), 90, np.uint8)
    cv2.fillPoly(img, [np.array([[60, 40], [330, 80], [320, 370], [70, 350]], np.int32)], (217, 165, 168))
    out, fixed = photo_tools.rectified(img, (60, 40, 330, 370))
    assert fixed and (np.abs(out.astype(int) - [217, 165, 168]).sum(-1) < 40).mean() > 0.8  # mostly the panel, not the wall


# ---- validation and repair on the real file ----


@pytest.fixture()
def built():
    glb, doc = hybrid.build(F.photo(), F.spec(), "wardrobe", None)
    doc.pop("_pendingImages")
    return glb, doc


def test_a_clean_model_passes_with_no_repairs(built):
    glb, doc = model_validation.validate_and_repair(*built)
    assert doc["validation"]["ok"] and not doc["validation"]["repairs"] and not doc["validation"]["errors"]


def test_broken_geometry_is_found_and_repaired_without_changing_the_design(built):
    glb, doc = built
    d, b = read_glb(glb)
    views = _views(d, b)

    def prim(name):
        node = next(n for n in d["nodes"] if n.get("name") == name)
        return d["meshes"][node["mesh"]]["primitives"][0]

    p = prim("Door 1")  # inside out
    model_validation._write(d, views, p["indices"], model_validation._read(d, views, p["indices"]).reshape(-1, 3)[:, ::-1].ravel().astype(np.uint32))
    p = prim("Door 2")  # normals zeroed
    model_validation._write(d, views, p["attributes"]["NORMAL"], np.zeros_like(model_validation._read(d, views, p["attributes"]["NORMAL"])))
    p = prim("Door 3")  # a broken texture coordinate
    uv = model_validation._read(d, views, p["attributes"]["TEXCOORD_0"])
    uv[0] = np.nan
    model_validation._write(d, views, p["attributes"]["TEXCOORD_0"], uv)
    p = prim("Top Panel")  # bent: one corner pushed 1 cm off the board
    pos = model_validation._read(d, views, p["attributes"]["POSITION"])
    pos[0, 1] += 0.01
    model_validation._write(d, views, p["attributes"]["POSITION"], pos)
    copy = {**next(n for n in d["nodes"] if n.get("name") == "Door 4")}
    copy["extras"] = {**copy["extras"], "partId": "door_4_copy"}
    d["nodes"].append(copy)
    d["nodes"][d["scenes"][0]["nodes"][0]]["children"].append(len(d["nodes"]) - 1)
    doc["components"].append({**next(c for c in doc["components"] if c["name"] == "Door 4"), "id": "door_4_copy"})
    fixed, report = model_validation.validate_and_repair(write_glb(d, views), doc)
    repairs = " ".join(report["validation"]["repairs"])
    for what in ("Door 1: was inside out", "Door 2: had invalid normals", "Door 3: had invalid texture", "Top Panel: was 8.0 mm off flat; rebuilt", "copy was removed"):
        assert what in repairs, what
    assert report["validation"]["ok"], report["validation"]["errors"]
    assert "door_4_copy" not in nodes_by_part(fixed) and len(report["components"]) == 14
    again, second = model_validation.validate_and_repair(fixed, report)
    assert not second["validation"]["repairs"]  # repairs are idempotent
    top = trimesh.load(io.BytesIO(fixed), file_type="glb", force="scene").geometry
    assert any(abs(g.extents[1] - 0.018) < 1e-4 for g in top.values())  # the top is a flat board again


def test_a_fluted_front_is_not_flattened_by_the_repairs(built):
    glb, doc = model_validation.validate_and_repair(*built)
    drawer = next(n for pid, n in nodes_by_part(glb).items() if "drawer" in pid)
    mesh = gltf(glb)["meshes"][drawer["mesh"]]
    assert gltf(glb)["accessors"][mesh["primitives"][0]["attributes"]["POSITION"]]["count"] > 24  # its ribs are still there


# ---- complex components: generated one by one, set in, never a block ----


class Cdn:
    def __init__(self):
        self.files: dict[str, bytes] = {}

    def upload(self, path, data, _type):
        self.files[path] = data
        return f"https://cdn.test/{path}"

    def public_url(self, path):
        return f"https://cdn.test/{path}"

    def get(self, key):
        return {"Body": io.BytesIO(self.files[key])}


def ornament_glb() -> bytes:
    """What Meshy would return for a carved crest: a textured, wide, shallow shape."""
    mesh = trimesh.creation.capsule(height=1.0, radius=0.12, count=[16, 16])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))  # lying along x (a capsule stands on z)
    mesh.apply_scale([1.0, 0.6, 0.5])
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(1).random((len(mesh.vertices), 2)),
                                                material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (32, 32), (150, 110, 70))))
    return mesh.export(file_type="glb")


class Meshy:
    def __init__(self, fail=False):
        self.created, self.fail = [], fail

    async def create_image_to_3d_task(self, _options, **kwargs):
        self.created.append(kwargs)
        return f"comp-{len(self.created)}"

    async def get_image_to_3d_task(self, task_id):
        if self.fail:
            return {"status": "FAILED", "task_error": {"message": "could not model it"}}, None
        return {"status": "SUCCEEDED", "progress": 100, "model_urls": {"glb": f"https://assets.meshy.test/{task_id}.glb"}}, None

    async def download_file(self, _url, dest):
        dest.write_bytes(ornament_glb())
        return dest.stat().st_size


CREST = {"type": "attached_to", "target": "body"}


def crest(id_, name, box):
    return F.comp(id_, name, box, cls="solid_wood", deco="carved", complexity="complex", relations=[CREST])


def run(tmp_path, monkeypatch, spec_doc, meshy, cdn=None, cached=None, polls=6):
    cdn = cdn or Cdn()
    calls = {"tries": 0, "refunds": 0}

    async def read_spec(*_a, **_k):
        return furniture_spec.FurnitureSpec.model_validate(spec_doc)

    async def take(*_a):
        calls["tries"] += 1
        return 29

    async def publish(event, _state):
        calls["refunds"] += type(event).__name__ == "GenerationFailed"

    async def cache_get(_s, _db, key, table="vision_cache"):
        return (cached or {}).get(key)

    async def cache_put(*_a, **_k):
        return None

    monkeypatch.setattr(vendor, "read_spec", read_spec)
    monkeypatch.setattr(vendor, "take_try", take)
    monkeypatch.setattr(vendor.bus, "publish", publish)
    monkeypatch.setattr(vendor.vision, "cache_get", cache_get)
    monkeypatch.setattr(vendor.vision, "cache_put", cache_put)
    s = Settings(meshy_api_key="msy-test", storage_dir=tmp_path, supabase_url="", cdn_public_url="https://cdn.test")
    monkeypatch.setattr(vendor, "get_settings", lambda: s)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: meshy, vendor.get_cdn: lambda: cdn,
                                     config.get_settings: lambda: s})
    seen = []
    try:
        with TestClient(app) as client:
            first = client.post("/api/vendor/generate", files={"image": ("w.png", F.photo(), "image/png")},
                                data={"category": "wardrobe", "lang": "en", "mode": "hybrid"}).json()
            done = first
            if "glb_url" not in first:
                for _ in range(polls):
                    done = client.get(f"/api/vendor/tasks/{first['task_id']}").json()
                    seen.append(done)
                    if done["status"] in ("SUCCEEDED", "FAILED"):
                        break
    finally:
        app.dependency_overrides.clear()
    return first, done, calls, cdn, seen


def test_a_carved_crest_is_generated_on_its_own_and_set_onto_its_fixed_host(tmp_path, monkeypatch):
    meshy = Meshy()
    spec = F.spec(extra=[crest("crest", "Carved crest", [0.3, 0.0, 0.7, 0.03])])
    first, done, calls, cdn, seen = run(tmp_path, monkeypatch, spec, meshy)
    assert first["status"] == "PENDING" and done["status"] == "SUCCEEDED", done
    assert len(meshy.created) == 1 and meshy.created[0]["texture_source"] == "image"  # only that component, its own pixels
    sent = Image.open(io.BytesIO(meshy.created[0]["image"]))
    assert sent.mode == "RGBA" and sent.width < 400  # the crest's cutout, not the whole photo
    doc = done["components"]
    part = next(c for c in doc["components"] if c["id"] == "carved_crest_generated")
    assert part["generation"] == "meshy" and part["taskId"] == "comp-1" and part["attachedTo"]
    assert not any(a["component"] == part["attachedTo"] for a in doc["animation"]["animations"])  # never a door
    glb = cdn.files[next(k for k in cdn.files if k.endswith(f"{first['task_id'][:0]}.glb") and "components/" not in k and "carpenter" not in k)]
    node = nodes_by_part(glb)["carved_crest_generated"]
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    assert node["extras"]["generation"] == "meshy" and doc["cost"]["meshGeneratorCalls"] == 1
    lo, hi = np.array(part["box"][0]), np.array(part["box"][1])
    geo = scene.geometry[next(k for k in scene.geometry if "Carved" in k or "crest" in k.lower())]
    assert np.all(geo.bounds[0] >= lo - 1e-3) and np.all(geo.bounds[1] <= hi + 1e-3)  # fitted into its box
    assert doc["validation"]["ok"] and calls["tries"] == 1 and any(p["status"] == "done" for p in doc["pendingComponents"])
    assert seen[0]["phase"] in ("decorative_reconstruction", "completed")


def test_a_repeated_ornament_is_generated_once_and_set_twice(tmp_path, monkeypatch):
    meshy = Meshy()
    spec = F.spec(extra=[crest("c_l", "Left carved corbel", [0.02, 0.0, 0.32, 0.03]), crest("c_r", "Right carved corbel", [0.68, 0.0, 0.98, 0.03])])
    _, done, *_ = run(tmp_path, monkeypatch, spec, meshy)
    ids = {c["id"] for c in done["components"]["components"] if c["generation"] == "meshy"}
    assert len(meshy.created) == 1 and ids == {"left_carved_corbel_generated", "right_carved_corbel_generated"}  # independent parts


def test_a_failed_component_is_reported_and_left_out_never_a_block(tmp_path, monkeypatch):
    first, done, _, cdn, _ = run(tmp_path, monkeypatch, F.spec(extra=[crest("crest", "Carved crest", [0.3, 0.0, 0.7, 0.03])]), Meshy(fail=True))
    doc = done["components"]
    assert done["status"] == "SUCCEEDED"  # the rest of the model is usable
    fail = doc["componentFailures"][0]
    assert fail["id"] == "carved_crest_generated" and fail["regenerable"] and "could not model it" in fail["reason"]
    assert not any(c["id"] == "carved_crest_generated" for c in doc["components"])  # nothing stands in for it
    assert "regenerate it from the editor" in doc["limits"]


def test_a_cutout_generated_before_costs_no_new_task(tmp_path, monkeypatch):
    cdn, meshy = Cdn(), Meshy()
    spec = F.spec(extra=[crest("crest", "Carved crest", [0.3, 0.0, 0.7, 0.03])])
    run(tmp_path, monkeypatch, spec, meshy, cdn=cdn)
    key = next(k for k in cdn.files if k.startswith("vendor-1/components/") and k.endswith(".glb"))
    cached = {}

    class Remember(Meshy):
        pass

    # the cutout's cache key, as start() computed it the first time
    import hashlib

    image = next(v for k, v in cdn.files.items() if k.endswith(".png") and "/components/" in k)
    cached[hashlib.sha256(image + b"meshy-component-1").hexdigest()] = {"glb_url": f"https://cdn.test/{key}"}
    again = Remember()
    first, done, *_ = run(tmp_path, monkeypatch, spec, again, cdn=cdn, cached=cached)
    assert again.created == [] and "glb_url" in first  # set from the cache at once, no Meshy task
    assert any(c["id"] == "carved_crest_generated" and c.get("cached") for c in first["components"]["components"])


def test_one_component_is_regenerated_keeping_every_other_part(tmp_path, monkeypatch):
    cdn = Cdn()
    first, done, *_ = run(tmp_path, monkeypatch, F.spec(extra=[crest("crest", "Carved crest", [0.3, 0.0, 0.7, 0.03])]), Meshy(fail=True), cdn=cdn)
    doc, url = done["components"], done["glb_url"]
    before = {c["id"] for c in doc["components"]}
    meshy = Meshy()
    s = Settings(meshy_api_key="msy-test", storage_dir=tmp_path, supabase_url="", cdn_public_url="https://cdn.test")

    async def take(*_a):
        return 28

    async def nothing(*_a, **_k):
        return None

    monkeypatch.setattr(vendor, "take_try", take)
    monkeypatch.setattr(vendor.vision, "cache_get", nothing)
    monkeypatch.setattr(vendor.vision, "cache_put", nothing)
    monkeypatch.setattr(vendor, "get_settings", lambda: s)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: meshy, vendor.get_cdn: lambda: cdn, config.get_settings: lambda: s})
    try:
        with TestClient(app) as client:
            started = client.post("/api/vendor/hybrid/components/regenerate", json={"glb_url": url, "components": doc, "component_id": "carved_crest_generated",
                                                                                    "model_id": first["task_id"]}).json()
            assert started["status"] == "PENDING", started
            result = client.get(f"/api/vendor/tasks/{started['task_id']}").json()
            other = client.post("/api/vendor/hybrid/components/regenerate", json={"glb_url": url.replace("vendor-1", "someone-else"), "components": doc,
                                                                                  "component_id": "carved_crest_generated", "model_id": "x"})
    finally:
        app.dependency_overrides.clear()
    assert result["status"] == "SUCCEEDED" and result["glb_url"] != url  # a new file; the saved one stays valid
    after = {c["id"] for c in result["components"]["components"]}
    assert before | {"carved_crest_generated"} == after and len(meshy.created) == 1
    assert not result["components"]["componentFailures"]
    assert other.status_code == 403  # only the owner's model


# ---- editor: confirmed sizes, one part repaired ----


def editor_call(tmp_path, monkeypatch, cdn, path, payload):
    s = Settings(meshy_api_key="msy-test", storage_dir=tmp_path, supabase_url="", cdn_public_url="https://cdn.test")
    monkeypatch.setattr(vendor, "get_settings", lambda: s)

    async def nothing(*_a, **_k):
        return None

    monkeypatch.setattr(vendor.vision, "cache_get", nothing)
    monkeypatch.setattr(vendor.vision, "cache_put", nothing)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: Meshy(), vendor.get_cdn: lambda: cdn, config.get_settings: lambda: s})
    try:
        with TestClient(app) as client:
            return client.post(path, json=payload)
    finally:
        app.dependency_overrides.clear()


def test_confirmed_sizes_rebuild_the_model_with_no_vision_call_and_the_same_part_ids(tmp_path, monkeypatch):
    cdn = Cdn()
    first, *_ = run(tmp_path, monkeypatch, F.spec(), Meshy(), cdn=cdn)
    doc = first["components"]
    doc["components"][0]["material"]["confirmed"] = True  # the vendor confirmed one material
    cdn.files["vendor-1/photo.png"] = F.photo()

    async def no_vision(*_a, **_k):
        raise AssertionError("a rebuild must not read the photo again")

    monkeypatch.setattr(vendor, "read_spec", no_vision)
    res = editor_call(tmp_path, monkeypatch, cdn, "/api/vendor/hybrid/rebuild", {"model_id": "m1", "source_image_url": "https://cdn.test/vendor-1/photo.png",
                                                                                "components": doc, "category": "wardrobe", "width_m": 1.8, "height_m": 2.1, "depth_m": 0.6})
    body = res.json()
    assert res.status_code == 200 and body["status"] == "SUCCEEDED", body
    new = body["components"]
    assert new["sizeM"] == {"width": 1.8, "height": 2.1, "depth": 0.6} and new["sizeSource"] == "vendor"
    assert [c["id"] for c in new["components"]] == [c["id"] for c in doc["components"]]  # same ids: edits and links survive
    assert new["components"][0]["material"]["confirmed"]
    glb = cdn.files[body["glb_url"].split("https://cdn.test/", 1)[1]]
    assert abs(trimesh.load(io.BytesIO(glb), file_type="glb", force="scene").extents[0] - 1.8) < 0.01  # the real file at the new size


def test_repairing_one_part_touches_only_that_part(tmp_path, monkeypatch, built):
    glb, doc = model_validation.validate_and_repair(*built)
    d, b = read_glb(glb)
    views = _views(d, b)
    for name in ("Door 1", "Door 2"):  # both broken; only Door 1 is repaired
        node = next(n for n in d["nodes"] if n.get("name") == name)
        p = d["meshes"][node["mesh"]]["primitives"][0]
        model_validation._write(d, views, p["attributes"]["NORMAL"], np.zeros_like(model_validation._read(d, views, p["attributes"]["NORMAL"])))
    cdn = Cdn()
    cdn.files["vendor-1/m1.glb"] = write_glb(d, views)
    door1 = next(c["id"] for c in doc["components"] if c["name"] == "Door 1")
    res = editor_call(tmp_path, monkeypatch, cdn, "/api/vendor/hybrid/components/repair",
                      {"glb_url": "https://cdn.test/vendor-1/m1.glb", "components": doc, "component_id": door1, "model_id": "m1"}).json()
    assert res["repairs"] == ["Door 1: had invalid normals; recomputed from its faces."] and res["glb_url"] != "https://cdn.test/vendor-1/m1.glb"
    fixed = cdn.files[res["glb_url"].split("https://cdn.test/", 1)[1]]
    _, again = model_validation.validate_and_repair(fixed, res["components"])
    assert again["validation"]["repairs"] == ["Door 2: had invalid normals; recomputed from its faces."]  # Door 2 left as it was
