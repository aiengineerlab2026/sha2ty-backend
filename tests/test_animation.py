"""Movable parts: the animation configuration (rigid moves of separate meshes only, everything doubtful flagged),
the estimated interior added as its own parts without touching the generated geometry."""

from __future__ import annotations

import io
import json

import numpy as np
import trimesh

from app.services import animation, wardrobe
from app.services.glb_parts import _gltf_json, append_boxes, inspect_glb
from app.services.mesh_analysis import parse


def cabinet_glb() -> bytes:
    scene = trimesh.Scene()
    for name, extents, centre in (("body", (1.0, 2.0, 0.6), (0, 1, 0)), ("door_left", (0.48, 1.4, 0.02), (-0.25, 1.2, 0.31)),
                                  ("door_right", (0.48, 1.4, 0.02), (0.25, 1.2, 0.31)), ("drawer_1", (0.96, 0.3, 0.02), (0, 0.3, 0.31))):
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(centre)
        scene.add_geometry(box, node_name=name, geom_name=name)
    return scene.export(file_type="glb")


ANALYSIS = {
    "objectType": "wardrobe", "summary": "", "movable": True,
    "interior": {"hasInterior": True, "confidence": 0.8, "items": [{"kind": "shelf", "count": 2, "behind": ["door_left"]}, {"kind": "rail", "count": 1, "behind": ["door_right"]}, {"kind": "drawer_box", "count": 1, "behind": []}]},
    "components": [
        {"id": "body", "name": "Body", "nameAr": "هيكل", "color": "#E8E0D0", "motion": {"type": "none"}},
        {"id": "door_left", "name": "Left door", "nameAr": "الباب الأيسر", "motion": {"type": "rotate", "hinge": "left", "limit": 100, "confidence": 0.9, "revealsInterior": True}},
        {"id": "door_right", "name": "Right door", "nameAr": "الباب الأيمن", "motion": {"type": "rotate", "hinge": "none", "limit": 400, "confidence": 0.9, "revealsInterior": True}},
        {"id": "drawer_1", "name": "Drawer", "nameAr": "درج", "motion": {"type": "slide", "direction": "front", "travel": 0.5, "confidence": 0.85}},
        {"id": "flap", "name": "Flap", "nameAr": "غطاء", "motion": {"type": "fold", "hinge": "bottom", "confidence": 0.9}},
    ],
}


def doc_for(glb: bytes) -> dict:
    doc = inspect_glb(glb, "wardrobe")
    for c in doc["components"]:
        c["analysisId"] = c["id"]
    return {**doc, "analysis": parse(json.dumps(ANALYSIS)).model_dump(mode="json")}


def test_doors_drawers_and_folds_become_rigid_moves_of_their_own_meshes():
    glb = cabinet_glb()
    _, doc = animation.configure(glb, doc_for(glb))
    anim = {a["component"]: a for a in doc["animation"]["animations"]}
    left, right, drawer = anim["door_left"], anim["door_right"], anim["drawer_1"]
    assert left["type"] == "rotation" and left["axis"] == "y" and left["to"] == -100 and left["enabled"]  # swings out to the front
    assert left["pivot"] == [-0.49, 1.2, 0.32]  # the door's own left edge, at its front face
    assert right["hinge"] == "left" and right["to"] == -180 and len(right["issues"]) == 2  # assumed hinge, limit capped: flagged
    assert drawer["type"] == "translation" and drawer["axis"] == "z" and drawer["distance"] == 0.31  # half the depth, forwards
    assert doc["animation"]["movable"] and doc["animation"]["source"] == "vision"
    flagged = {f["label"]: f["reason"] for f in doc["animation"]["flagged"]}
    assert "not a separate mesh" in flagged["Flap"] and "Right door" in flagged  # never forced onto another mesh
    assert "body" not in anim  # static parts stay static


def test_estimated_interior_is_added_as_separate_parts_and_the_original_bytes_are_kept():
    glb = cabinet_glb()
    out, doc = animation.configure(glb, doc_for(glb))
    estimated = [c for c in doc["components"] if c.get("estimated")]
    assert [c["id"] for c in estimated] == ["estimated_shelf_1", "estimated_shelf_2", "estimated_hanging_rail_1", "estimated_drawer_box_1"]
    assert estimated[0]["nameAr"] == "رف 1 (تقديري)" and doc["animation"]["interior"]["added"] == [c["id"] for c in estimated]
    drawer = next(a for a in doc["animation"]["animations"] if a["component"] == "drawer_1")
    assert drawer["follow"] == ["estimated_drawer_box_1"]  # the box slides with its front
    shelf_lo, shelf_hi = estimated[0]["box"]
    assert -0.5 < shelf_lo[0] and shelf_hi[0] <= 0.0 + 1e-6  # behind the left door only, inside the body
    before = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    after = trimesh.load(io.BytesIO(out), file_type="glb", force="scene")
    for name, geom in before.geometry.items():  # every generated mesh exactly as it was
        assert np.array_equal(after.geometry[name].vertices, geom.vertices) and np.array_equal(after.geometry[name].faces, geom.faces)
    assert len(after.geometry) == len(before.geometry) + 4
    parts = inspect_glb(out)["components"]
    assert [p["name"] for p in parts][-4:] == [c["name"] for c in estimated]  # in the order the editor maps by


def test_a_fused_model_and_a_static_object_are_never_animated():
    whole = trimesh.creation.box(extents=(1, 2, 0.6)).export(file_type="glb")
    doc = {**inspect_glb(whole), "analysis": parse(json.dumps(ANALYSIS)).model_dump(mode="json")}
    out, doc = animation.configure(whole, doc)
    assert out == whole and doc["animation"]["animations"] == [] and not doc["animation"]["movable"]
    assert all("fused" in f["reason"] or "not a separate mesh" in f["reason"] for f in doc["animation"]["flagged"])
    table = {"objectType": "table", "components": [{"id": "top", "name": "Top"}, {"id": "leg", "name": "Leg"}]}
    glb = cabinet_glb()
    _, doc = animation.configure(glb, {**inspect_glb(glb), "analysis": parse(json.dumps(table)).model_dump(mode="json")})
    assert doc["animation"]["animations"] == [] and doc["animation"]["flagged"] == [] and "interior" not in doc["animation"]


def test_without_an_analysis_doors_and_drawers_are_found_by_name():
    from tests.test_wardrobe import _built

    glb, names = _built()
    _, doc = animation.configure(glb, wardrobe.components_doc(glb, names))
    kinds = {a["component"]: (a["type"], a.get("hinge"), a["enabled"]) for a in doc["animation"]["animations"]}
    assert kinds["door_left"] == ("rotation", "left", True) and kinds["door_right"] == ("rotation", "right", True)
    assert any(k.startswith("drawer") and v[0] == "translation" for k, v in kinds.items())
    assert doc["animation"]["source"] == "names" and not any(c.get("estimated") for c in doc["components"])


def test_append_boxes_writes_a_valid_glb():
    glb = cabinet_glb()
    out, added = append_boxes(glb, [{"name": "Shelf", "min": [0, 0, 0], "max": [1, 0.02, 0.5], "color": "#336699"}])
    doc = _gltf_json(out)
    assert added == [{"mesh": len(doc["meshes"]) - 1, "node": len(doc["nodes"]) - 1, "material": len(doc["materials"]) - 1, "primitive": 0}]
    assert doc["buffers"][0]["byteLength"] == int.from_bytes(out[20 + int.from_bytes(out[12:16], "little") :][:4], "little")
    assert int.from_bytes(out[8:12], "little") == len(out)
    shelf = trimesh.load(io.BytesIO(out), file_type="glb", force="scene").geometry["Shelf"]
    assert np.allclose(shelf.bounds, [[0, 0, 0], [1, 0.02, 0.5]])
