"""The hybrid reconstruction engine (services/hybrid.py): furniture rebuilt from the photo's spec, part by part, each
part textured with its own region of the photo. No mesh generator, no paid call besides the (mocked) photo spec."""

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
from app.services import furniture_spec, hybrid
from tests import hybrid_fixtures as F
from tests.test_generation_architecture import FakeCdn


def gltf(glb: bytes) -> dict:
    n = struct.unpack_from("<I", glb, 12)[0]
    return json.loads(glb[20:20 + n])


def texture_of(glb: bytes, node_name: str) -> np.ndarray:
    """The decoded base-colour texture of the named part."""
    doc = gltf(glb)
    n = struct.unpack_from("<I", glb, 12)[0]
    binary = glb[20 + n + 8:]
    node = next(nd for nd in doc["nodes"] if nd.get("name") == node_name)
    mat = doc["materials"][doc["meshes"][node["mesh"]]["primitives"][0]["material"]]
    image = doc["images"][doc["textures"][mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]]
    view = doc["bufferViews"][image["bufferView"]]
    return np.asarray(Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])).convert("RGB")).astype(int)


@pytest.fixture(scope="module")
def wardrobe():
    return hybrid.build(F.photo(), F.spec(), "wardrobe", None)


# ---- structure and modularity ----


def test_a_four_door_wardrobe_is_rebuilt_as_its_named_independent_parts(wardrobe):
    glb, doc = wardrobe
    names = [c["name"] for c in doc["components"]]
    assert doc["separated"] and doc["method"] == "hybrid"
    for n in range(1, 5):
        assert f"Door {n}" in names and f"Door {n} handle" in names
    assert {"Left Side Panel", "Right Side Panel", "Top Panel", "Back Panel", "Plinth Drawer"} <= {n.title() for n in names}
    assert not any(n.lower().startswith("mesh") for n in names)  # meaningful names only
    nodes = [nd for nd in gltf(glb)["nodes"] if "mesh" in nd]
    assert len(nodes) == len(doc["components"]) == len({nd["mesh"] for nd in nodes})  # one mesh per part, none shared
    assert [nd["extras"]["partId"] for nd in nodes] == [c["id"] for c in doc["components"]]  # stable ids in the file


def test_panels_are_flat_closed_boards_at_the_real_size(wardrobe):
    glb, doc = wardrobe
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    w, h, d = scene.extents
    assert abs(w - 1.6) < 0.01 and abs(h - 2.0) < 0.01 and abs(d - 0.58) < 0.06  # metres, handles proud of the front
    for c in doc["components"]:
        size = np.subtract(c["box"][1], c["box"][0])
        if "panel" in c["name"].lower() or (c["name"].startswith("Door") and "handle" not in c["name"]):
            assert abs(sorted(size)[0] - 0.018) < 1e-3, c["name"]  # one board thickness, flat
    for geom in scene.geometry.values():
        if len(geom.vertices) == 24:  # a plain board
            assert geom.is_volume or geom.is_watertight or len(geom.faces) == 12


def test_vendor_measurements_win_and_are_marked_measured():
    glb, doc = hybrid.build(F.photo(), F.spec(), "wardrobe", None, size=(2.4, 2.2, 0.6))
    assert doc["sizeM"] == {"width": 2.4, "height": 2.2, "depth": 0.6} and doc["sizeSource"] == "vendor"
    assert doc["validation"]["confidence"] == "measured"
    _, est = hybrid.build(F.photo(), F.spec(), "wardrobe", None)
    assert est["sizeSource"] == "estimated" and est["validation"]["confidence"] == "estimated"


# ---- appearance: the photo's own pixels ----


def test_each_door_is_textured_with_its_own_colour_from_the_photo_without_its_handle(wardrobe):
    glb, doc = wardrobe
    door1, door2 = texture_of(glb, "Door 1"), texture_of(glb, "Door 2")
    assert np.abs(np.median(door1.reshape(-1, 3), 0) - F.PINK).max() < 12  # the pink outer door stays pink
    assert np.abs(np.median(door2.reshape(-1, 3), 0) - F.WHITE).max() < 12  # the white inner door stays white
    brass = (np.abs(door1 - F.BRASS).sum(-1) < 40).mean()
    assert brass < 0.002  # its handle is its own part, painted out of the panel's texture
    assert all(c["material"]["fromPhoto"] for c in doc["components"] if c["name"].startswith("Door"))
    assert not any(c["material"]["confirmed"] for c in doc["components"])  # the vendor confirms materials


def test_a_mirror_door_keeps_a_mirror_material_not_a_photo_of_its_reflection():
    spec = F.spec()
    door = next(c for c in spec["analysis"]["components"] if c["id"] == "door_2")
    door.update(materialClass="mirror", name="Door 2 mirror")
    glb, doc = hybrid.build(F.photo(), spec, "wardrobe", None)
    part = next(c for c in doc["components"] if c.get("analysisId") == "door_2")
    mat = gltf(glb)["materials"][part["mesh"]]["pbrMetallicRoughness"]
    assert "baseColorTexture" not in mat and mat["metallicFactor"] == 1.0 and mat["roughnessFactor"] < 0.1


def test_a_fluted_front_gets_real_ribs_and_other_regular_decoration_stays_the_photos_pixels(wardrobe):
    glb, doc = wardrobe
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    drawer = next(g for name, g in scene.geometry.items() if "drawer" in name.lower())
    assert len(drawer.vertices) > 24 and drawer.bounds[1][2] > next(c for c in doc["components"] if "drawer" in c["name"].lower())["box"][1][2] - 1e-6
    made = next(d for d in doc["hybrid"]["decorations"] if d["decoration"] == "fluted")
    assert made["method"] == "geometry" and made["count"] == 16 and made["fromPhoto"]  # the 16 flutes the photo shows


# ---- movement ----


def test_every_door_opens_on_its_own_hinge_edge_carrying_only_its_own_handle(wardrobe):
    _, doc = wardrobe
    anims = {a["component"]: a for a in doc["animation"]["animations"]}
    doors = [c for c in doc["components"] if c["name"].startswith("Door") and "handle" not in c["name"]]
    for c, hinge in zip(doors, ("left", "right", "left", "right")):
        a = anims[c["id"]]
        assert a["enabled"] and a["type"] == "rotation" and a["hinge"] == hinge and not a["issues"]
        edge = c["box"][0][0] if hinge == "left" else c["box"][1][0]
        assert abs(a["pivot"][0] - edge) < 1e-6  # on the real hinge edge
        handle = next(h["id"] for h in doc["components"] if h["name"] == f"{c['name']} handle")
        assert a["follow"] == [handle]  # its own handle, no neighbour's
    drawer = next(a for a in anims.values() if a["type"] == "translation")
    assert drawer["enabled"] and drawer["distance"] > 0  # slides out the front
    assert doc["validation"]["ok"], doc["validation"]["errors"]


def test_a_relation_to_the_wrong_door_is_overruled_by_the_geometry():
    """The reading ties door 1's handle to door 2 (a handle by the line between two doors): it stays on door 1."""
    spec = F.spec()
    h1 = next(c for c in spec["analysis"]["components"] if c["id"] == "handle_1")
    h1["relations"] = [{"type": "attached_to", "target": "door_2"}]
    _, doc = hybrid.build(F.photo(), spec, "wardrobe", None)
    owner = doc["attachments"][next(c["id"] for c in doc["components"] if c.get("analysisId") == "handle_1")]
    assert owner == next(c["id"] for c in doc["components"] if c.get("analysisId") == "door_1")


def test_a_door_with_two_handles_keeps_both():
    spec = F.spec(extra=[F.comp("handle_1b", "Door 1 lower handle", [0.23, 0.6, 0.242, 0.7], cls="metal", role="functional")])
    _, doc = hybrid.build(F.photo(), spec, "wardrobe", None)
    door1 = next(a for a in doc["animation"]["animations"] if a["component"] == next(c["id"] for c in doc["components"] if c.get("analysisId") == "door_1"))
    assert len(door1["follow"]) == 2 and doc["validation"]["ok"]


# ---- what is not rebuilt from boards ----


def test_an_upholstered_sofa_is_handed_to_the_generator_with_the_reason():
    spec = F.spec()
    spec["analysis"]["components"] = [
        F.comp("frame", "Sofa frame", [0, 0.3, 1, 1], cls="solid_wood"),
        F.comp("seat", "Seat cushions", [0.05, 0.45, 0.95, 0.7], cls="fabric", deco="upholstered", complexity="complex"),
        F.comp("back", "Back cushions", [0.05, 0.05, 0.95, 0.45], cls="fabric", deco="upholstered", complexity="complex"),
        F.comp("arm_l", "Left armrest", [0, 0.2, 0.1, 0.8], cls="fabric", deco="upholstered"),
    ]
    glb, doc = hybrid.build(F.photo(), spec, "furniture", "sofa")
    assert glb is None and "upholstered" in doc["reason"]


def test_a_carved_part_is_left_for_the_mesh_provider_never_built_as_a_block():
    spec = F.spec(extra=[F.comp("crest", "Carved crest", [0.3, 0.0, 0.7, 0.03], cls="solid_wood", deco="carved", complexity="complex")])
    glb, doc = hybrid.build(F.photo(), spec, "wardrobe", None)
    images = doc.pop("_pendingImages")
    (part,) = doc["pendingComponents"]
    assert part["status"] == "pending" and part["name"] == "Carved crest" and len(images[part["group"]]) > 0
    assert not any("crest" in c["name"].lower() for c in doc["components"])  # no stand-in in the model


# ---- the endpoint ----


def run_generate(tmp_path, monkeypatch, spec_doc, **form):
    cdn, made = FakeCdn(), {"meshy": 0, "tries": 0, "refunds": 0}

    async def read_spec(*_a, **_k):
        return furniture_spec.FurnitureSpec.model_validate(spec_doc) if spec_doc else None

    async def take(*_a):
        made["tries"] += 1
        return 29

    async def publish(event, _state):
        made["refunds"] += type(event).__name__ == "GenerationFailed"

    class NoMeshy:
        async def create_image_to_image_task(self, *_a, **_k):
            made["meshy"] += 1
            return "img-1"

        async def create_image_to_3d_task(self, *_a, **_k):
            made["meshy"] += 1
            return "mesh-1"

    monkeypatch.setattr(vendor, "read_spec", read_spec)
    monkeypatch.setattr(vendor, "take_try", take)
    monkeypatch.setattr(vendor.bus, "publish", publish)
    s = Settings(meshy_api_key="msy-test", storage_dir=tmp_path, supabase_url="", meshy_texture_source="image", meshy_clean_image=False)
    monkeypatch.setattr(vendor, "get_settings", lambda: s)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: NoMeshy(), vendor.get_cdn: lambda: cdn,
                                     config.get_settings: lambda: s})
    try:
        with TestClient(app) as client:
            res = client.post("/api/vendor/generate", files={"image": ("w.png", F.photo(), "image/png")},
                              data={"category": "wardrobe", "lang": "en", "mode": "hybrid", **form})
    finally:
        app.dependency_overrides.clear()
    return res, made, cdn


def test_hybrid_generation_returns_the_rebuilt_model_with_no_mesh_generator_call(tmp_path, monkeypatch):
    res, made, cdn = run_generate(tmp_path, monkeypatch, F.spec(), width_m="2.4", height_m="2.2", depth_m="0.6")
    body = res.json()
    assert res.status_code == 200 and body["mode"] == "hybrid" and body["glb_url"].endswith(".glb")
    assert made == {"meshy": 0, "tries": 1, "refunds": 0}  # one try, no Meshy task
    doc = body["components"]
    assert doc["method"] == "hybrid" and doc["validation"]["ok"] and doc["sizeSource"] == "vendor"
    assert doc["carpenter"]["url"].endswith(".carpenter.glb") and doc["cost"]["meshGeneratorCalls"] == 0
    assert any(k.endswith(".glb") and not k.endswith(".carpenter.glb") for k in cdn.uploads)


def test_when_the_photo_cannot_be_read_the_generator_makes_it_and_says_why(tmp_path, monkeypatch):
    res, made, _ = run_generate(tmp_path, monkeypatch, None)
    assert res.status_code == 200 and made["meshy"] == 1 and made["refunds"] == 1 and made["tries"] == 2  # the hybrid try went back
