"""Technical quality never decides publishing; validation repairs what it safely can, records the rest, and never
repeats itself. Regression case: the wardrobe from the report (model 43a5b0c2), whose central divider ran through
its bottom board ("pass through each other (176 cm³)", "98 cm² of faces in the same place"), each message stored
twice, and which was held as an unpublishable draft. Its stored GLB, document and photo are fixtures."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services import assembly, hybrid, model_validation, solids
from app.services.cdn import CdnError

FIX = Path(__file__).parent / "fixtures"
GLB = (FIX / "divider_wardrobe.glb").read_bytes()
DOC = json.loads((FIX / "divider_wardrobe.doc.json").read_text())
PHOTO = (FIX / "divider_wardrobe.png").read_bytes()


def part(doc, pid):
    return next(c for c in doc["components"] if c["id"] == pid)


@pytest.fixture(scope="module")
def repaired():
    return model_validation.validate_and_repair(GLB, copy.deepcopy(DOC))


def test_the_stored_model_was_held_back_for_its_geometry():
    v = DOC["validation"]
    assert v["status"] == "needs_repair" and v["publishable"] is False
    assert len(v["errors"]) == 4 and len(set(v["errors"])) == 2  # each message stored twice


def test_the_divider_is_repaired_automatically_to_stand_on_the_base(repaired):
    glb, doc = repaired
    v = doc["validation"]
    assert v["errors"] == [] and v["technical"] == "passed" and v["publishable"] and v["usable"]
    assert any("Central vertical divider" in r and "stops at Bottom panel" in r for r in v["repairs"])
    divider, base, top = part(doc, "divider_center"), part(doc, "base_panel"), part(doc, "top_panel")
    assert abs(divider["box"][0][1] - base["box"][1][1]) < 1e-4  # on the base's upper face: a joint
    assert abs(divider["box"][1][1] - top["box"][0][1]) < 1e-4  # under the top board
    metrics = doc["hybrid"]["geometryMetrics"]
    assert not metrics["intersections"] and not metrics["zFighting"]
    assert len([a for a in doc["animation"]["animations"] if a["enabled"]]) == len(DOC["animation"]["animations"])


def test_validation_is_idempotent_and_never_repeats_a_message(repaired):
    glb, doc = repaired
    again_glb, again = model_validation.validate_and_repair(glb, copy.deepcopy(doc))
    assert again["validation"]["errors"] == doc["validation"]["errors"] and again["validation"]["warnings"] == doc["validation"]["warnings"]
    assert again["validation"]["repairs"] == []  # nothing left to repair
    # a model whose problems cannot be repaired lists each one once, however often it is validated
    broken = copy.deepcopy(DOC)
    for c in broken["components"]:
        c["generation"] = "meshy"  # generated parts are never trimmed: the overlap stays
    _, once = model_validation.validate_and_repair(GLB, broken)
    _, twice = model_validation.validate_and_repair(GLB, copy.deepcopy(once))
    for v in (once["validation"], twice["validation"]):
        assert len(v["warnings"]) == len(set(v["warnings"])) and len(v["errors"]) == len(set(v["errors"]))
        ids = [i["id"] for i in v["issues"]]
        assert len(ids) == len(set(ids))
    assert [i["id"] for i in once["validation"]["issues"]] == [i["id"] for i in twice["validation"]["issues"]]


def test_a_repaired_issue_leaves_the_current_list():
    broken = copy.deepcopy(DOC)
    for c in broken["components"]:
        c["generation"] = "meshy"
    glb, before = model_validation.validate_and_repair(GLB, broken)
    assert any(i["type"] == "intersection" for i in before["validation"]["issues"])
    for c in before["components"]:
        c["generation"] = "procedural"
    _, after = model_validation.validate_and_repair(glb, before)
    assert not any(i["type"] == "intersection" for i in after["validation"]["issues"])
    assert any(i["type"] == "intersection" for i in after["validation"]["resolved"])


def test_boards_touching_at_a_joint_are_not_a_defect():
    doc = {"components": [{"id": "base", "name": "Base", "kind": "base_panel", "generation": "procedural", "box": [[-0.5, 0.1, -0.3], [0.5, 0.118, 0.28]]},
                          {"id": "div", "name": "Divider", "kind": "divider", "generation": "procedural", "box": [[-0.009, 0.118, -0.3], [0.009, 2.0, 0.27]]}]}
    built = [{"name": c["name"], "id": c["id"], "kind": c["kind"], "assembly": None, "lo": np.array(c["box"][0]), "hi": np.array(c["box"][1]),
              "mesh": solids.mesh_arrays(solids.box(*c["box"])), "texture": None, "colour": [0.6, 0.6, 0.6], "alpha": 1.0, "metallic": 0.0, "roughness": 0.6,
              "extras": {"partId": c["id"], "generation": "procedural"}} for c in doc["components"]]
    glb = hybrid.write(built, [])
    between = model_validation.geometry_between(glb, doc)
    assert between["errors"] == [] and not between["metrics"]["intersections"] and not between["metrics"]["zFighting"]


def test_the_assembly_places_dividers_and_shelves_inside_the_carcass_boards():
    r = DOC
    spec = {"details": r["specDetails"], "analysis": copy.deepcopy(r["analysis"])}
    spec["analysis"]["components"].append({"id": "shelf_wide", "name": "Shelf", "kind": "shelf", "box": [0.05, 0.5, 0.95, 0.52], "visibility": "inferred"})
    size = tuple(r["sizeM"][k] for k in ("width", "height", "depth"))
    glb, doc = hybrid.build(PHOTO, spec, "wardrobe", None, "hybrid", size)
    doc.pop("_pendingImages", None)
    glb, doc = model_validation.validate_and_repair(glb, doc)
    divider, base, top = part(doc, "divider_center"), part(doc, "base_panel"), part(doc, "top_panel")
    assert divider["box"][0][1] == pytest.approx(base["box"][1][1]) and divider["box"][1][1] == pytest.approx(top["box"][0][1])
    shelf = part(doc, "shelf_wide")
    assert shelf["box"][1][0] <= divider["box"][0][0] + 1e-6 or shelf["box"][0][0] >= divider["box"][1][0] - 1e-6  # in one compartment
    assert doc["validation"]["repairs"] == [] and not doc["hybrid"]["geometryMetrics"]["intersections"]  # built right, nothing to repair


def test_quality_and_publishing_are_separate_properties():
    v = model_validation.verdict({"components": [{"id": "a"}], "validation": {"errors": ["A and B pass through each other (5 cm³)."], "warnings": ["Door: its colour differs from the photo (ΔE 30)."]}, "sizeSource": "photo"})
    assert v["status"] == "needs_repair" and v["technical"] == "issues" and v["repair"]["unresolved"] == 1
    assert v["usable"] and v["publishable"]
    assert v["errors"] == ["A and B pass through each other (5 cm³)."]  # never cleared to look valid
    assert {n["kind"] for n in v["notes"]} >= {"check", "dimension"}
    legacy = model_validation.verdict(copy.deepcopy(DOC))  # an old needs_repair / publishable false save
    assert legacy["publishable"]
    assert not model_validation.verdict({"components": [], "validation": {}})["publishable"]  # no model at all


def test_operational_failures_are_still_reported():
    with pytest.raises(ValueError):
        model_validation.validate_and_repair(b"not a glb", {})  # a corrupt file is never passed off as validated
    from app.main import app

    @app.get("/__test_storage_failure")
    async def fail():
        raise CdnError("The file could not be stored.", 503)

    res = TestClient(app).get("/__test_storage_failure")
    assert res.status_code == 503 and res.json()["detail"] == "The file could not be stored."
    assert hybrid.build(PHOTO, None, "wardrobe", None)[0] is None  # no reading: no model, said so


def test_board_repair_never_trims_moving_parts():
    assert "door" not in model_validation.YIELD and "drawer_front" not in model_validation.YIELD
    assert set(model_validation.YIELD) >= assembly.CARCASS_KINDS | assembly.INSIDE_KINDS - {"crown_trim"}
