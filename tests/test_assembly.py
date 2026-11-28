"""The constraint-based assembly engine (services/assembly.py + hybrid.py + materials.py + model_validation.py +
visual_check.py) on a real regression case: a two-door sliding wardrobe (model 7970bf39) whose earlier reconstruction
z-fought, overlapped and pasted photographs on its parts. Its photo and the reading the configured vision model gave
(spec v3) are fixtures; every check runs on the real GLB that is produced."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from app.services import assembly, hybrid, materials, model_validation, solids, visual_check
from app.services.glb_size import read_glb

FIX = Path(__file__).parent / "fixtures"
PHOTO = (FIX / "mirror_wardrobe.png").read_bytes()
SPEC = json.loads((FIX / "mirror_wardrobe.spec.json").read_text())


@pytest.fixture(scope="module")
def model():
    glb, doc = hybrid.build(PHOTO, json.loads(json.dumps(SPEC)), "wardrobe", None)
    doc.pop("_pendingImages")
    return model_validation.validate_and_repair(glb, doc)


def parts(doc):
    return {c["id"]: c for c in doc["components"]}


def size(c):
    return np.subtract(c["box"][1], c["box"][0])


# ---- the regression case ----


def test_the_old_reconstruction_is_rejected_for_what_was_wrong_with_it():
    old = json.loads((FIX / "mirror_wardrobe.before.json").read_text())
    _, doc = model_validation.validate_and_repair((FIX / "mirror_wardrobe.before.glb").read_bytes(), old)
    v = doc["validation"]
    assert v["status"] == "needs_repair" and not v["publishable"]
    assert any("flicker" in e for e in v["errors"]) and any("pass through each other" in e for e in v["errors"])


def test_the_wardrobe_has_exactly_its_physical_parts_each_once(model):
    _, doc = model
    got = {c["kind"] for c in doc["components"]}
    assert {"side_panel", "top_panel", "crown_trim", "plinth", "back_panel", "base_panel", "sliding_door", "door_frame", "mirror"} == got
    assert len([c for c in doc["components"] if c["kind"] == "side_panel"]) == 2  # left and right, inferred one by one
    assert not any(c["id"] in ("door_right", "body") for c in doc["components"])  # logical assemblies are not solids
    door = next(a for a in doc["assemblies"] if a["id"] == "door_right")
    assert sorted(door["children"]) == ["door_right_frame", "door_right_mirror"]
    estimated = {c["id"] for c in doc["components"] if c.get("estimated")}
    assert {"side_left", "back_panel", "base_panel"} <= estimated and "side_right" not in estimated


def test_physical_sizes_come_from_roles_not_photographed_rectangles(model):
    _, doc = model
    p = parts(doc)
    for side in ("side_left", "side_right"):  # a side seen at an angle looks ~4 cm wide in the photo; it is a board
        assert abs(sorted(size(p[side]))[0] - assembly.BOARD) < 1e-6
    assert abs(size(p["back_panel"])[2] - assembly.BACK) < 1e-6
    w = [size(p[d])[0] for d in ("door_left", "door_right_frame")]
    assert abs(w[0] - w[1]) < 1e-6  # sliding doors: the same width
    assert p["door_left"]["box"][1][2] != p["door_right_frame"]["box"][1][2]  # on two tracks
    overlap = p["door_left"]["box"][1][0] - p["door_right_frame"]["box"][0][0]
    assert abs(overlap - assembly.OVERLAP) < 1e-6


def test_vendor_sizes_do_not_stretch_board_thickness():
    glb, doc = hybrid.build(PHOTO, json.loads(json.dumps(SPEC)), "wardrobe", None, size=(2.6, 2.4, 0.6))
    doc.pop("_pendingImages")
    p = parts(doc)
    assert doc["sizeM"] == {"width": 2.6, "height": 2.4, "depth": 0.6}
    assert abs(sorted(size(p["side_right"]))[0] - assembly.BOARD) < 1e-6 and abs(size(p["door_left"])[2] - assembly.SLIDING) < 1e-6
    ext = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene").extents
    assert abs(ext[0] - 2.6) < 1e-3 and abs(ext[1] - 2.4) < 1e-3


def test_the_mirror_door_is_a_hollow_frame_with_its_pane_in_the_opening(model):
    glb, doc = model
    p = parts(doc)
    frame, pane = p["door_right_frame"], p["door_right_mirror"]
    lo, hi = np.array(frame["box"])
    plo, phi = np.array(pane["box"])
    assert np.all(plo >= lo) and np.all(phi <= hi) and plo[2] > lo[2] and phi[2] < hi[2]  # inside the frame, mid-thickness
    m = doc["hybrid"]["geometryMetrics"]
    assert not m["intersections"] and not m["zFighting"]
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    geo = scene.geometry[next(k for k in scene.geometry if "frame" in k.lower())]
    assert geo.volume < 0.5 * np.prod(geo.extents)  # a frame, not a filled board
    assert pane["material"]["class"] == "mirror" and pane["material"]["source"] == "physical"


def test_the_left_door_carries_its_two_mouldings_as_geometry(model):
    glb, doc = model
    rings = [d for d in doc["hybrid"]["decorations"] if d["component"] == "door_left" and "moulding" in d["decoration"]]
    assert len(rings) == 2 and all(r["method"] == "geometry" for r in rings)
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    door = scene.geometry[next(k for k in scene.geometry if "door" in k.lower() and "mirror" not in k.lower() and "frame" not in k.lower())]
    assert door.bounds[1][2] > parts(doc)["door_left"]["box"][1][2] - solids.MOULDING_HEIGHT  # relief stands off the face


def test_materials_are_clean_samples_never_photo_crops(model):
    glb, doc = model
    p = parts(doc)
    assert not read_glb(glb)[0].get("images")  # painted and veneered surfaces: colours, no pasted photographs
    cream = np.array([int(p["door_left"]["material"]["color"][i:i + 2], 16) for i in (1, 3, 5)])
    assert cream.min() > 150 and cream[0] >= cream[2]  # light and warm: cream, not grey or black
    dark = np.array([int(p["plinth_front"]["material"]["color"][i:i + 2], 16) for i in (1, 3, 5)])
    assert dark.max() < 120  # the dark brown trim
    for c in doc["components"]:
        # a side seen edge-on in shadow takes the reading's colour for it (recorded as an estimate); the rest is sampled
        # ... or the finish of the reliable part its reading colour matches (a material group)
        allowed = ("colour sample", "texture sample", "physical") + (("reading colour", "material group") if c["kind"] in ("side_panel", "back_panel", "top_panel", "base_panel") else ("material group",) if c["kind"] == "door_frame" else ())
        assert c["material"]["source"] in allowed, c


def test_the_model_matches_the_photo_part_by_part(model):
    glb, doc = model
    seen = visual_check.compare(glb, PHOTO, doc)
    seen.pop("_images")
    by = {p["component"]: p for p in seen["parts"]}
    assert by["door_left"]["iou"] > 0.9 and by["door_right_mirror"]["mirrorIoU"] > 0.9 and by["door_right_frame"]["iou"] > 0.8
    assert all(p.get("colourDE", 0) < visual_check.DE_MAX for p in seen["parts"]) and not seen["issues"]


def test_uncertainty_is_a_note_that_never_blocks_publishing(model):
    _, doc = model
    v = doc["validation"]
    assert v["errors"] == [] and v["status"] == "valid_with_warnings" and v["publishable"]
    assert "confirm" not in v and any(n["kind"] == "movement" for n in v["notes"])
    assert len(doc["animation"]["uncertain"]) == 2  # sliding or hinged: recorded, not required to confirm
    assert all(a["enabled"] and a["issues"] == [] for a in doc["animation"]["animations"])  # never disabled by a note
    assert doc["animation"]["authoritative"]


# ---- the mechanisms on their own ----


def test_face_uvs_are_real_size_on_every_face_and_never_degenerate():
    pos, nrm, uv, idx = solids.mesh_arrays(solids.box([0, 0, 0], [0.6, 1.2, 0.018]))
    tri, uvt = pos.reshape(-1, 3, 3), uv.reshape(-1, 3, 2)
    a3 = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    au = 0.5 * np.abs((uvt[:, 1, 0] - uvt[:, 0, 0]) * (uvt[:, 2, 1] - uvt[:, 0, 1]) - (uvt[:, 2, 0] - uvt[:, 0, 0]) * (uvt[:, 1, 1] - uvt[:, 0, 1]))
    assert np.allclose(au / a3, 1 / solids.TILE ** 2)  # one texel density on every face: nothing stretched


def test_masks_go_through_the_same_perspective_as_the_photo():
    rgb = np.zeros((200, 200, 3), np.uint8)
    rgb[:, :] = (200, 180, 160)
    quad = [[0.1, 0.15], [0.9, 0.05], [0.9, 0.95], [0.1, 0.85]]
    mask = np.full((200, 200), 255, np.uint8)
    mask[80:120, :] = 0  # a reflection band across the face
    face, valid = materials.rectify(rgb, mask, quad)
    assert face.shape[:2] == valid.shape and 0.15 < 1 - (valid > 0).mean() < 0.3  # the band, warped with the face


def test_a_reflection_never_colours_a_part():
    analysis = json.loads(json.dumps(SPEC["analysis"]))
    frame = next(c for c in analysis["components"] if c["id"] == "door_right_frame")
    rgb = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(io.BytesIO(PHOTO)).convert("RGB"))
    mat = materials.extract(rgb, frame, analysis, None, 0.5)
    c = np.array(mat["color"]) * 255
    assert c.min() > 140  # the cream frame, not the room's browns and greens in the mirror


def test_a_logical_door_listed_as_a_solid_is_still_only_a_parent():
    analysis = json.loads(json.dumps(SPEC["analysis"]))
    next(c for c in analysis["components"] if c["id"] == "door_right")["logical"] = False  # the reading forgot to say
    plan = assembly.assemble(analysis, SPEC["details"], (2.2, 2.15, 0.65), "wardrobe")
    assert "door_right" not in {p["id"] for p in plan["parts"]}  # its frame and mirror are its parts: never a third solid


def test_a_side_listed_twice_is_built_once():
    analysis = json.loads(json.dumps(SPEC["analysis"]))
    twin = {**next(c for c in analysis["components"] if c["id"] == "side_right"), "id": "right_panel", "name": "Right panel"}
    analysis["components"].append(twin)
    plan = assembly.assemble(analysis, SPEC["details"], (2.2, 2.15, 0.65), "wardrobe")
    assert len([p for p in plan["parts"] if p["kind"] == "side_panel"]) == 2


def test_hinged_doors_are_never_assumed_from_the_word_door():
    analysis = json.loads(json.dumps(SPEC["analysis"]))
    for c in analysis["components"]:
        if c.get("motion"):
            c["motion"] = {"type": "none", "confidence": 0.9}
    glb, doc = hybrid.build(PHOTO, {"details": SPEC["details"], "analysis": analysis}, "wardrobe", None)
    assert doc["animation"]["animations"] == []  # the reading says they do not move: nothing invented
