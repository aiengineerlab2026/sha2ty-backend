"""Regression: the real 3-bedroom apartment drawing (tests/fixtures/three_bedroom_apartment.png).

OCR is replayed from a recording made with macOS Vision so the test runs on any OS; a separate
test runs the live engine when one is installed. Nothing here is used by production code: the
assertions describe the architecture of the drawing, not coordinates the pipeline was tuned to.

Visual output: tests/output/three_bedroom_final.png (drawing + final walls, openings, rooms).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import pytest
from shapely.geometry import LineString, Polygon

from app.services.plan import dimension_ocr, image_processor
from app.services.plan.dimension_ocr import OcrBox
from app.services.plan.geometry_builder import DEFAULT_HEIGHT, build_room_glb, room_extents

FIXTURES = Path(__file__).parent / "fixtures"
IMAGE = FIXTURES / "three_bedroom_apartment.png"
OUTPUT = Path(__file__).parent / "output"

# Tolerances (documented): printed room sizes are nominal and drawn by hand-ish CAD, so a room
# measures within 8% of its label; the drawing's global scale is about 16 px/ft.
ROOM_SIZE_TOLERANCE = 0.08
PX_PER_FT_RANGE = (15.2, 17.2)


def recorded_ocr(_image):
    rows = json.loads((FIXTURES / "three_bedroom_apartment.ocr.json").read_text())
    return [OcrBox(**r) for r in rows], []


@pytest.fixture(scope="module")
def plan():
    mp = pytest.MonkeyPatch()
    mp.setattr(image_processor, "run_ocr", recorded_ocr)
    try:
        payload = image_processor.plan_from_image(IMAGE.read_bytes())
    finally:
        mp.undo()
    return payload


@pytest.fixture(scope="module", autouse=True)
def visual_output(plan):
    from app.services.plan.room_segmenter import segment

    img = cv2.imread(str(IMAGE))
    walls = [image_processor._wall(w) for w in plan["walls_px"]]
    openings = [image_processor._opening(o) for o in plan["openings_px"]]
    labels = [image_processor._label(l) for l in plan["labels"]]
    seg = segment(img.shape, walls, openings, plan["thickness_px"], labels)
    OUTPUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT / "three_bedroom_final.png"), image_processor._draw_final(img, seg, walls, openings, plan["scale"]))


def zones(plan, kind):
    return [z for z in plan["zones_px"] if z["kind"] == kind]


def test_three_bedrooms_with_printed_sizes(plan):
    bedrooms = zones(plan, "bedroom")
    assert len(bedrooms) == 3
    assert sorted(z["dims_text"].replace(" ", "") for z in bedrooms) == ["13'x12'", "13'x12'", "14'x10'"]
    # Each bedroom is its own enclosed room, not a zone of something bigger.
    assert all(z["boundary"] == "wall" for z in bedrooms)
    assert len({z["room_id"] for z in bedrooms}) == 3


def test_two_bathrooms_are_separate_enclosed_rooms(plan):
    baths = zones(plan, "bathroom")
    assert len(baths) == 2
    assert len({z["room_id"] for z in baths}) == 2
    assert all(z["boundary"] == "wall" for z in baths)


def test_living_room_and_patio(plan):
    living = zones(plan, "living")
    patio = zones(plan, "patio")
    assert len(living) == 1 and len(patio) == 1
    assert living[0]["dims_text"].replace(" ", "") == "16'x13'"
    assert patio[0]["room_id"] != living[0]["room_id"], "patio is enclosed by walls and the slider"


def test_open_plan_dining_kitchen_is_not_split_by_an_invented_wall(plan):
    dining, kitchen, living = zones(plan, "dining"), zones(plan, "kitchen"), zones(plan, "living")
    assert len(dining) == 1 and len(kitchen) == 1
    # There is no wall between living, dining and kitchen in the drawing: one physical room, three zones.
    assert dining[0]["room_id"] == kitchen[0]["room_id"] == living[0]["room_id"]
    room = next(r for r in plan["rooms_px"] if r["id"] == dining[0]["room_id"])
    assert room["kind"] == "open_plan"
    # Living/dining follows the drawn floor-finish line; dining/kitchen has no line and is flagged.
    assert living[0]["boundary"] == "drawn_line"
    assert kitchen[0]["boundary"] == "inferred"


def test_dimension_parsing_feet_and_inches(plan):
    dining = zones(plan, "dining")[0]
    a, b = sorted(dining["dims_m"])
    assert abs(a - 3.048) < 1e-3 and abs(b - 3.81) < 1e-3  # 10' and 12'6"


def test_rooms_do_not_overlap_or_cross_walls_or_leave_footprint(plan):
    footprint = Polygon(plan["footprint_px"])
    polys = [Polygon(r["polygon"], r["holes"]) for r in plan["rooms_px"]]
    for i, a in enumerate(polys):
        assert a.is_valid
        assert a.difference(footprint.buffer(2)).area < a.area * 0.01
        for b in polys[i + 1 :]:
            assert a.intersection(b).area < 1.0
    assert all(r["crosses_walls"] < 0.03 for r in plan["rooms_px"])
    assert all(not r["invalid"] for r in plan["rooms_px"])


def test_exterior_walls_detected(plan):
    t = plan["thickness_px"]
    xs = [p[0] for p in plan["footprint_px"]]
    ys = [p[1] for p in plan["footprint_px"]]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    walls = [w for w in plan["walls_px"]]

    def longest_near(axis, value):
        best = 0.0
        for w in walls:
            (x0, y0), (x1, y1) = w["start"], w["end"]
            if axis == "y" and w["orientation"] == "h" and abs(y0 - value) < 2 * t:
                best = max(best, abs(x1 - x0))
            if axis == "x" and w["orientation"] == "v" and abs(x0 - value) < 2 * t:
                best = max(best, abs(y1 - y0))
        return best

    assert longest_near("y", bottom) > (right - left) * 0.9  # the long south wall
    assert longest_near("x", left) > (bottom - top) * 0.12  # west wall (broken by the entry door)
    assert longest_near("x", right) > (bottom - top) * 0.05
    assert longest_near("y", top) > (right - left) * 0.1


def test_internal_walls_preserved(plan):
    footprint = Polygon(plan["footprint_px"]).exterior
    t = plan["thickness_px"]
    internal = [w for w in plan["walls_px"] if LineString([w["start"], w["end"]]).distance(footprint) > 2 * t]
    assert len(internal) >= 12
    # Every bedroom and bathroom is reached through a door, never through a missing wall.
    by_room = {}
    for e in plan["adjacency"]:
        by_room.setdefault(e["a"], []).append(e)
        by_room.setdefault(e["b"], []).append(e)
    for z in zones(plan, "bedroom") + zones(plan, "bathroom"):
        assert any(e["via"] == "door" for e in by_room.get(z["room_id"], [])), z["name"]


def test_no_large_false_diagonal_walls(plan):
    t = plan["thickness_px"]
    for w in plan["walls_px"]:
        (x0, y0), (x1, y1) = w["start"], w["end"]
        length = math.hypot(x1 - x0, y1 - y0)
        ang = abs(math.degrees(math.atan2(y1 - y0, x1 - x0))) % 90
        assert length < 2 * t or min(ang, 90 - ang) < 1, w


def test_doors_found_by_symbol(plan):
    doors = [o for o in plan["openings_px"] if o["kind"] == "door"]
    assert len(doors) >= 10
    assert all("arc" in o["evidence"] for o in doors)


def test_scale_matches_printed_measurements(plan):
    scale = plan["scale"]
    assert scale["uniform"]
    assert PX_PER_FT_RANGE[0] < scale["px_per_ft"] < PX_PER_FT_RANGE[1]
    assert scale["confidence"] >= 0.8
    checked = 0
    for z in plan["zones_px"]:
        if z["boundary"] != "wall" or not z.get("dims_m"):
            continue
        assert z["dims_error"] < ROOM_SIZE_TOLERANCE, (z["name"], z["dims_text"], z["measured_m"])
        checked += 1
    assert checked >= 3


def test_validated_3d_model(plan):
    assert plan["validation"]["ok"], plan["validation"]["errors"]
    width, depth, height = room_extents(plan)
    # 740 px x 580 px of building at ~16 px/ft ≈ 14 m x 11 m
    assert 12 < width < 16 and 9 < depth < 13 and height == DEFAULT_HEIGHT
    assert len(plan["doors"]) >= 10
    assert build_room_glb(plan)[:4] == b"glTF"


def test_correction_roundtrip_rebuilds_same_rooms(plan):
    """The correction view re-segments from the vector walls; with no edits it must agree."""
    state = {k: plan[k] for k in ("image_size", "thickness_px", "walls_px", "openings_px", "labels", "zone_lines", "straight_lines")}
    again = image_processor.plan_from_state(state)
    assert [z["name"] for z in again["zones_px"]] == [z["name"] for z in plan["zones_px"]]


def test_removing_an_exterior_wall_blocks_3d(plan):
    state = {k: plan[k] for k in ("image_size", "thickness_px", "walls_px", "openings_px", "labels", "zone_lines")}
    longest = max(state["walls_px"], key=lambda w: abs(w["end"][0] - w["start"][0]))
    state["walls_px"] = [w for w in state["walls_px"] if w is not longest]
    broken = image_processor.plan_from_state(state)
    assert not broken["validation"]["ok"]
    assert broken["validation"]["errors"]


@pytest.mark.skipif(dimension_ocr._engine() is None, reason="no OCR engine installed")
def test_live_ocr_reads_printed_dimensions():
    boxes, notes = dimension_ocr.run_ocr(cv2.imread(str(IMAGE)))
    labels, unit = dimension_ocr.room_labels(boxes)
    assert unit == "ft", notes
    dims = sorted(l.dims_text.replace(" ", "") for l in labels if l.dims_text)
    assert len(dims) == 5, dims
    assert sum(l.kind == "bedroom" for l in labels) == 3
