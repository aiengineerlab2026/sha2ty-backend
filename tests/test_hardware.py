"""Furniture hardware (bars, knobs, recesses), handleless doors and the nonblocking verdict."""

from __future__ import annotations

import copy

import numpy as np

from app.services import hybrid, model_validation, photo_tools, solids
from tests import hybrid_fixtures as F


def _watertight(m) -> bool:
    return m.status().name == "NoError" and m.volume() > 0 and m.genus() >= 0


def test_hardware_primitives_are_watertight_and_fill_only_part_of_their_box():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([0.016, 0.6, 0.032])
    bar = solids.bar_handle(lo, hi, "vertical", 0.016)
    knob = solids.knob([0, 0, 0], [0.03, 0.03, 0.022])
    for m, box_vol in ((bar, 0.016 * 0.6 * 0.032), (knob, 0.03 * 0.03 * 0.022)):
        assert _watertight(m)
        assert 0.2 < m.volume() / box_vol < 0.7  # a grip on posts, a stem and cap: never a solid slab
    b = np.array(bar.bounding_box())
    assert np.allclose(b[:3], lo, atol=1e-3) and np.allclose(b[3:], hi, atol=1e-3)
    r = solids.recess([0.1, 0.1, 0.02], [0.14, 0.3, 0.02])
    assert _watertight(r) and np.array(r.bounding_box())[2] < 0.02  # cut into the face behind it


def test_a_bar_is_found_measured_and_oriented_on_its_door():
    face = np.full((300, 120, 3), (200, 200, 195), np.uint8)
    face[60:240, 96:102] = (70, 70, 72)  # a dark vertical bar near the free edge
    bar = photo_tools.find_bar(face)
    assert bar and bar["orientation"] == "vertical" and bar["aspect"] >= 3
    x0, y0, x1, y1 = bar["rect"]
    assert abs(y0 - 0.2) < 0.03 and abs(y1 - 0.8) < 0.03 and abs((x0 + x1) / 2 - 0.825) < 0.03
    knob = np.full((300, 120, 3), (200, 200, 195), np.uint8)
    knob[140:152, 90:102] = (60, 60, 60)
    assert photo_tools.find_bar(knob)["orientation"] not in ("vertical", "horizontal")
    assert photo_tools.find_bar(np.full((300, 120, 3), 200, np.uint8)) is None  # a plain door shows no handle


def test_handles_are_owned_by_their_door_and_handleless_doors_get_none_invented():
    glb, doc = hybrid.build(F.photo(), F.spec(), "wardrobe", None)
    for c in (c for c in doc["components"] if c["kind"] == "handle"):
        assert c["host"].startswith("door_") and c["hardware"]["type"] in ("bar", "knob", "recessed")
    spec = copy.deepcopy(F.spec())
    spec["analysis"]["components"] = [c for c in spec["analysis"]["components"] if "handle" not in c["id"]]
    glb, doc = hybrid.build(F.photo(), spec, "wardrobe", None)
    assert not [c for c in doc["components"] if c["kind"] in ("handle", "knob")]  # nothing invented
    doors = [a for a in doc["animation"]["animations"] if a["type"] == "rotation"]
    assert len(doors) == 4 and all(a["enabled"] for a in doors)  # each still opens


def test_only_a_missing_model_blocks_publishing():
    base = {"components": [{"id": "door"}], "animation": {"uncertain": ["Door 1"]}, "hybrid": {"materialFallbacks": [{"component": "side", "reason": "in shadow"}]}}
    v = model_validation.verdict({**base, "validation": {"errors": [], "warnings": []}})
    assert v["status"] == "valid_with_warnings" and v["publishable"] and {n["kind"] for n in v["notes"]} == {"movement", "material"}
    legacy = model_validation.verdict({**base, "validation": {"status": "needs_confirmation", "errors": [], "confirm": ["How Door 1 opens…"]}})
    assert legacy["publishable"] and "confirm" not in legacy  # an older save with nothing broken publishes
    broken = model_validation.verdict({**base, "validation": {"errors": ["Door 1: inside out."]}})
    assert broken["status"] == "needs_repair" and broken["technical"] == "issues" and broken["publishable"]  # quality, not eligibility
    assert broken["errors"] == ["Door 1: inside out."]  # recorded, never cleared to let it through
    empty = model_validation.verdict({"components": [], "validation": {}})
    assert empty["status"] == "failed" and not empty["publishable"]
