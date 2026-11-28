"""Design preservation: the rebuilt furniture keeps the photo's proportions, trims and finishes (spec v5).

Regression case: a grey three-door wardrobe (model 3cef1e2a) — a plain left door with a bar handle, two mirrored doors
with thin borders, a minimal top edge and a thin dark base. Its shipped model had a 12.8 cm crown invented from the
top panel's photographed depth, doors 15 cm short and a warm brown base. Its photo, the stored reading and the shipped
GLB are fixtures; every check runs on the GLB that is produced."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from app.services import assembly, hybrid, materials, model_validation, visual_check, visual_review

FIX = Path(__file__).parent / "fixtures"
PHOTO = (FIX / "grey_mirror_wardrobe.png").read_bytes()
READING = json.loads((FIX / "grey_mirror_wardrobe.reading.json").read_text())
SIZE = tuple(READING["size"])
W, H, D = SIZE


def spec(analysis=None):
    a = copy.deepcopy(analysis or READING["analysis"])
    return {"details": hybrid.details_from(a, SIZE), "analysis": a}


def by_id(doc):
    return {c["id"]: c for c in doc["components"]}


@pytest.fixture(scope="module")
def rebuilt():
    return hybrid.build_corrected(PHOTO, spec(), "wardrobe", None, "hybrid", SIZE)


def with_component(extra):
    a = copy.deepcopy(READING["analysis"])
    a["components"].append(extra)
    return a


# ---- the regression wardrobe ----


def test_no_crown_from_a_structural_top_panel_and_the_doors_are_full_height(rebuilt):
    glb, doc = rebuilt
    kinds = {c["kind"] for c in doc["components"]}
    assert "crown_trim" not in kinds  # the reading's top panel is the carcass's top board, not a trim
    for d in ("door_left", "frame_center", "frame_right"):
        top, bottom = by_id(doc)[d]["box"][1][1], by_id(doc)[d]["box"][0][1]
        assert top > H - 0.025 and bottom < 0.1  # base to top edge, as photographed
    widths = [by_id(doc)[d]["box"][1][0] - by_id(doc)[d]["box"][0][0] for d in ("door_left", "frame_center", "frame_right")]
    Hm = assembly.homography(READING["analysis"]["frontQuad"])
    rects = [assembly.front_rect(next(c for c in READING["analysis"]["components"] if c["id"] == i), Hm, None) for i in ("door_left", "frame_center", "frame_right")]
    photo = [r[2] - r[0] for r in rects]  # widths on the straightened front (the nearer door looks wider in the photo)
    assert np.allclose(np.array(widths) / sum(widths), np.array(photo) / sum(photo), atol=0.03)  # relative widths kept


def test_thin_mirror_borders_and_a_thin_dark_neutral_base(rebuilt):
    _, doc = rebuilt
    vc = doc["visualCheck"]
    for m in vc["design"]["mirrors"]:
        assert m["borderM"] < 0.02 and abs(m["borderM"] - m["photoBorderM"]) < 0.006
    plinth = by_id(doc)["plinth"]
    assert plinth["box"][1][1] - plinth["box"][0][1] < 0.1
    rgb = np.array([int(plinth["material"]["color"][i:i + 2], 16) for i in (1, 3, 5)])
    assert rgb.max() - rgb.min() < 14 and rgb.mean() < 120  # dark and neutral, not a warm brown


def test_one_neutral_finish_shared_by_door_carcass_and_frames(rebuilt):
    _, doc = rebuilt
    c = by_id(doc)
    finish = c["door_left"]["material"]["color"]
    for pid in ("side_left", "side_right", "top_panel", "frame_center", "frame_right"):
        assert c[pid]["material"]["color"] == finish and c[pid]["material"]["group"] == "door_left"
    rgb = np.array([int(finish[i:i + 2], 16) for i in (1, 3, 5)])
    assert rgb.max() - rgb.min() < 10  # grey, no beige cast from the room
    assert c["plinth"]["material"]["group"] == "plinth" and c["plinth"]["material"]["color"] != finish  # a trim keeps its own


def test_functional_parts_are_preserved(rebuilt):
    _, doc = rebuilt
    anims = {a["component"]: a for a in doc["animation"]["animations"]}
    assert {k for k, a in anims.items() if a["enabled"]} == {"door_left", "frame_center", "frame_right"}
    assert anims["frame_center"]["follow"] == ["mirror_center"] and "handle_left" in anims["door_left"]["follow"]
    assert by_id(doc)["handle_left"]["hardware"]["type"] == "bar"
    assert doc["validation"]["publishable"] and not doc["validation"]["errors"]
    assert doc["visualCheck"]["issues"] == [] and doc["visualCheck"]["meanIoU"] > 0.85


def test_the_shipped_model_is_flagged_for_what_was_wrong():
    from app.services.glb_size import read_glb

    old = (FIX / "grey_mirror_wardrobe.before.glb").read_bytes()
    _, doc = hybrid.build_corrected(PHOTO, spec(), "wardrobe", None, "hybrid", SIZE)
    # The shipped file's parts, under the same reading, as their boxes in that file.
    import io

    import trimesh

    d, _ = read_glb(old)
    scene = trimesh.load(io.BytesIO(old), file_type="glb", force="scene", process=False)
    names = {n.get("name"): (n.get("extras") or {}).get("partId") for n in d["nodes"] if n.get("name")}
    known = by_id(doc)
    comps = []
    for node in scene.graph.nodes_geometry:
        T, g = scene.graph[node]
        v = trimesh.transform_points(scene.geometry[g].vertices, T)
        pid = names.get(node) or node
        base = known.get(pid) or {"id": pid, "name": pid, "kind": "crown_trim" if "crown" in pid else "other"}
        comps.append({**base, "box": [v.min(0).tolist(), v.max(0).tolist()], "material": {}})
    vc = visual_check.compare(old, PHOTO, {**doc, "components": comps})
    assert any(u["kind"] == "crown_trim" and u["coversShown"] > 0.9 for u in vc["design"]["unexpected"])
    assert all(f["top"] > 0.06 for f in vc["design"]["fronts"])  # every door ~15 cm short at the top
    assert {c["op"] for c in vc["corrections"]} >= {"remove", "front_extent"}


# ---- the correction loop ----


def test_an_unsupported_crown_is_removed_in_bounded_rounds_keeping_ids():
    a = with_component({"id": "crown", "name": "Crown", "kind": "crown_trim", "box": [0, 0, 1, 0.07], "color": "#99998F", "visibility": "visible"})
    first = hybrid.build(PHOTO, spec(a), "wardrobe", None, "hybrid", SIZE)[1]
    assert "crown" in by_id(first)  # the reading's crown, built as told
    glb, doc = hybrid.build_corrected(PHOTO, spec(a), "wardrobe", None, "hybrid", SIZE)
    rounds = doc["visualCorrection"]["rounds"]
    assert 1 <= len(rounds) <= hybrid.MAX_ROUNDS and rounds[0]["accepted"] and rounds[0]["after"] > rounds[0]["before"]
    assert "crown" not in by_id(doc) and by_id(doc)["door_left"]["box"][1][1] > H - 0.025
    assert set(by_id(first)) - {"crown"} == set(by_id(doc))  # every other part kept under its id
    assert len([x for x in doc["animation"]["animations"] if x["enabled"]]) == 3
    assert doc["validation"]["publishable"]


def test_a_correction_that_breaks_movement_is_rejected():
    glb, doc = hybrid.build_corrected(PHOTO, spec(), "wardrobe", None, "hybrid", SIZE)
    current = (glb, doc, doc["visualCheck"])
    # squash the left door to a sliver: it no longer matches the photo (and is refused)
    _, _, entry = hybrid.refine(PHOTO, {**spec(), "analysis": doc["analysis"]}, "wardrobe", None, "hybrid", SIZE, current,
                                [{"op": "front_extent", "component": "door_left", "rect": [0.0, 0.4, 0.05, 0.6]}], "test")
    assert not entry["accepted"]


def test_a_vision_review_answer_is_validated_before_use():
    ids = {"door_left", "plinth", "frame_center"}
    raw = json.dumps({"corrections": [
        {"op": "remove", "component": "door_left", "kind": "door", "reason": "x"},  # only invented trims are removed
        {"op": "front_extent", "component": "door_left", "rect": [0, 0, 0.3, 1], "reason": "taller"},
        {"op": "frame_border", "component": "frame_center", "border": 3, "reason": "x"},  # out of range
        {"op": "rewrite_code", "component": "plinth", "reason": "x"},
        {"op": "plinth_height", "component": "nowhere", "height": 0.03, "reason": "x"},
    ]})
    assert visual_review.validate(raw, ids) == [{"op": "front_extent", "component": "door_left", "rect": [0, 0, 0.3, 1], "reason": "taller"}]
    assert visual_review.validate("not json", ids) == []


# ---- the mechanisms ----


def test_a_real_crown_measured_by_the_reading_is_kept_above_shorter_doors():
    a = copy.deepcopy(READING["analysis"])
    a["design"] = {"topTrim": {"type": "crown", "height": 0.06, "confidence": 0.8, "evidence": "a moulded cornice"}}
    _, doc = hybrid.build(PHOTO, spec(a), "wardrobe", None, "hybrid", SIZE)
    crown = by_id(doc)["crown_trim"]
    assert abs((crown["box"][1][1] - crown["box"][0][1]) - (0.06 * H - assembly.BOARD)) < 0.01
    assert by_id(doc)["door_left"]["box"][1][1] <= crown["box"][0][1] + 1e-6  # the doors stop under it


def test_a_top_surface_seen_from_above_is_never_a_crown():
    a = copy.deepcopy(READING["analysis"])
    top = next(c for c in a["components"] if c["id"] == "top_panel")
    top["quad"] = [[0.2, 0.17], [0.66, 0.15], [0.662, 0.191], [0.196, 0.231]]  # above the front's top edge
    _, doc = hybrid.build(PHOTO, spec(a), "wardrobe", None, "hybrid", SIZE)
    assert "crown_trim" not in {c["kind"] for c in doc["components"]}


def test_plinth_types_follow_the_measurement():
    for base, check in (({"type": "none", "confidence": 0.9}, lambda c: "plinth" not in c),
                        ({"type": "recessed_plinth", "height": 0.05, "confidence": 0.9}, lambda c: c["plinth"]["box"][1][2] < D / 2 - 0.02)):
        a = copy.deepcopy(READING["analysis"])
        a["components"] = [c for c in a["components"] if c["kind"] != "plinth"]
        a["design"] = {"base": base}
        _, doc = hybrid.build(PHOTO, spec(a), "wardrobe", None, "hybrid", SIZE)
        assert check(by_id(doc)), base


def test_frameless_mirrors_have_no_visible_frame():
    a = copy.deepcopy(READING["analysis"])
    a["design"] = {"frameBorder": {"type": "frameless", "confidence": 0.9}}
    glb, doc = hybrid.build(PHOTO, spec(a), "wardrobe", None, "hybrid", SIZE)
    glb, doc = model_validation.validate_and_repair(glb, doc)
    frame, mirror = by_id(doc)["frame_center"], by_id(doc)["mirror_center"]
    fw = frame["box"][1][0] - frame["box"][0][0]
    mw = mirror["box"][1][0] - mirror["box"][0][0]
    assert mw > fw - 0.005 and mirror["box"][0][2] >= frame["box"][1][2] - 1e-6  # the mirror covers the door, in front of its backing
    assert not doc["validation"]["errors"]
    assert next(x for x in doc["animation"]["animations"] if x["component"] == "frame_center")["follow"] == ["mirror_center"]


def test_box_only_parts_go_through_the_same_perspective_as_parts_with_corners():
    quad = [[0.2, 0.25], [0.65, 0.19], [0.66, 0.88], [0.2, 0.82]]
    Hm = assembly.homography(quad)
    obj = [0.2, 0.19, 0.66, 0.88]
    corner = [[0.2, 0.25], [0.34, 0.23], [0.34, 0.84], [0.2, 0.82]]  # a door's corners
    xs, ys = [p[0] for p in corner], [p[1] for p in corner]
    box = [(min(xs) - obj[0]) / (obj[2] - obj[0]), (min(ys) - obj[1]) / (obj[3] - obj[1]), (max(xs) - obj[0]) / (obj[2] - obj[0]), (max(ys) - obj[1]) / (obj[3] - obj[1])]
    with_quad = assembly.front_rect({"quad": corner}, Hm, obj)
    with_box = assembly.front_rect({"box": box}, Hm, obj)
    assert np.allclose(with_quad[::2], with_box[::2], atol=0.02)  # same place across the front
    assert abs(with_box[1]) < 0.06 and abs(with_box[3] - 1) < 0.06  # its box is corrected, not taken as already flat
    assert with_box != box


def test_quads_are_validated_before_a_homography():
    good = [[0.1, 0.1], [0.9, 0.12], [0.88, 0.9], [0.12, 0.88]]
    assert assembly.valid_quad(good)
    assert not assembly.valid_quad(good[::-1])  # wrong order
    assert not assembly.valid_quad([[0.1, 0.1], [0.9, 0.1], [0.2, 0.2], [0.1, 0.9]])  # concave
    assert not assembly.valid_quad([[0.1, 0.1], [0.1001, 0.1], [0.1001, 0.1001], [0.1, 0.1001]])  # degenerate
    assert not assembly.valid_quad([[0, 0], [1, 0], [0.6, 0.1], [0.4, 0.1]])  # unstable (an edge 5× its opposite)
    assert assembly.front_quad({"frontQuad": good[::-1], "components": []}, {}) is None


def test_a_neutral_finish_under_warm_light_stays_neutral_and_a_warm_one_stays_warm():
    warm_cast = np.array([2.0, 12.0])
    tinted = materials._lab(np.uint8([[150, 140, 120]]))[0]  # a grey door photographed in warm light
    out, why = materials.fuse(tinted, [0.6, 0.6, 0.59], warm_cast)
    assert why and abs(out[1] - 128) < 3 and abs(out[2] - 128) < 4 and out[0] == tinted[0]
    oak = materials._lab(np.uint8([[170, 130, 90]]))[0]
    same, why = materials.fuse(oak, [0.68, 0.52, 0.36], warm_cast)  # the reading says it is warm: kept as measured
    assert why is None and np.array_equal(same, oak)
    assert materials.extract(np.zeros((20, 20, 3), np.uint8), {"kind": "door"}, {}, None, 0.5, default=[0.3, 0.3, 0.3])["color"] == [0.3, 0.3, 0.3]


def test_details_are_rebuilt_for_a_model_whose_save_lost_them():
    d = hybrid.details_from(READING["analysis"], SIZE)
    assert d["width_m"] == W and d["dimension_source"] == "vendor" and 0 <= d["bbox"][0] < d["bbox"][2] <= 1
    assert hybrid.details_from(READING["analysis"], (W, None, D)) is None


def test_an_intentional_two_tone_design_keeps_both_finishes():
    a = copy.deepcopy(READING["analysis"])
    for c in a["components"]:
        if c["kind"] == "door_frame":
            c["color"] = "#3A3B38"  # dark frames around the mirrors, light doors: a design choice
    _, doc = hybrid.build(PHOTO, spec(a), "wardrobe", None, "hybrid", SIZE)
    c = by_id(doc)
    assert c["frame_center"]["material"]["group"] != "door_left" and c["frame_center"]["material"]["color"] != c["door_left"]["material"]["color"]
