"""The hybrid engine across furniture types: what is built from boards, what is generated on its own, what goes to the
whole-model generator. The readings follow what a vision model returns for such photos (types, parts, materials,
relations); the photo is drawn, so these check the engine's choices per type, not visual fidelity."""

from __future__ import annotations

import pytest

from app.services import hybrid, model_validation
from tests import hybrid_fixtures as F

C = F.comp
BODY = {"type": "attached_to", "target": "frame"}


def reading(object_type, comps, size):
    w, h, d = size
    return {"details": {"bbox": F.OBJ, "width_m": w, "height_m": h, "depth_m": d, "dimension_source": "inferred", "object_type": object_type},
            "analysis": {"objectType": object_type, "components": comps, "movable": any(c["motion"]["type"] != "none" for c in comps)}}


def legs(n=4, cls="solid_wood", complexity="regular"):
    xs = [0.02, 0.9][:2] if n == 2 else [0.02, 0.9, 0.02, 0.9]
    return [C(f"leg_{i + 1}", f"Leg {i + 1}", [x, 0.85, x + 0.08, 1.0], cls=cls, complexity=complexity, relations=[BODY]) for i, x in enumerate(xs[:n])]


CASES = {
    "bed": (reading("double bed", [
        C("frame", "Bed frame", [0, 0.55, 1, 0.85], cls="solid_wood"),
        C("headboard", "Carved headboard", [0.05, 0, 0.95, 0.55], cls="solid_wood", deco="carved", complexity="complex", relations=[BODY]),
        C("footboard", "Footboard", [0.05, 0.6, 0.95, 0.85], cls="solid_wood", relations=[BODY]), *legs()], (1.7, 1.2, 2.1)), "furniture", "double bed"),
    "table": (reading("dining table", [C("top", "Tabletop", [0, 0, 1, 0.1], cls="solid_wood"), C("apron", "Apron rail", [0.03, 0.1, 0.97, 0.2], cls="solid_wood"), *legs()], (1.6, 0.75, 0.9)), "table", None),
    "table_curved_legs": (reading("dining table with cabriole legs", [C("top", "Tabletop", [0, 0, 1, 0.1], cls="solid_wood"),
                                                                      *legs(complexity="complex")], (1.6, 0.75, 0.9)), "table", None),
    "desk": (reading("writing desk", [C("top", "Desk top", [0, 0, 1, 0.08], cls="laminate"),
        C("pedestal", "Drawer pedestal", [0.65, 0.08, 1, 1], cls="laminate"),
        C("drawer", "Top drawer", [0.67, 0.1, 0.98, 0.35], cls="laminate", role="functional", motion={"type": "slide", "direction": "front", "confidence": 0.8}),
        C("drawer_handle", "Top drawer handle", [0.78, 0.2, 0.86, 0.24], cls="metal", relations=[{"type": "attached_to", "target": "drawer"}]),
        C("side", "Left side panel", [0, 0.08, 0.04, 1], cls="laminate")], (1.4, 0.76, 0.7)), "furniture", "desk"),
    "cabinet": (reading("kitchen base cabinet", [C("body", "Cabinet body", [0, 0, 1, 1], cls="mdf"),
        C("door_l", "Left door", [0.02, 0.3, 0.49, 0.97], cls="painted_wood", role="functional", motion={"type": "rotate", "hinge": "left", "confidence": 0.8}),
        C("door_r", "Right door", [0.51, 0.3, 0.98, 0.97], cls="painted_wood", role="functional", motion={"type": "rotate", "hinge": "right", "confidence": 0.8}),
        C("drawer", "Drawer", [0.02, 0.03, 0.98, 0.27], cls="painted_wood", role="functional", deco="recessed_panel", motion={"type": "slide", "direction": "front", "confidence": 0.8})],
        (0.8, 0.9, 0.6)), "furniture", "cabinet"),
    "bookshelf": (reading("open bookshelf", [C("body", "Bookshelf frame", [0, 0, 1, 1], cls="wood_veneer"),
        *[C(f"shelf_{i}", f"Shelf {i}", [0.04, 0.2 * i, 0.96, 0.2 * i + 0.03], cls="wood_veneer") for i in range(1, 5)]], (0.9, 1.9, 0.35)), "furniture", "bookshelf"),
    "chair": (reading("dining chair", [C("frame", "Chair frame", [0.05, 0.45, 0.95, 0.6], cls="solid_wood"),
        C("seat", "Upholstered seat", [0.05, 0.45, 0.95, 0.55], cls="fabric", deco="upholstered", complexity="complex", relations=[BODY]),
        C("back", "Backrest", [0.1, 0, 0.9, 0.45], cls="solid_wood", deco="grooved", relations=[BODY]), *legs()], (0.5, 0.95, 0.55)), "furniture", "dining chair"),
    "sofa": (reading("three-seat sofa", [C("frame", "Sofa base", [0, 0.8, 1, 0.95], cls="solid_wood"),
        C("seat", "Seat cushions", [0.08, 0.45, 0.92, 0.8], cls="fabric", deco="upholstered", complexity="complex"),
        C("back", "Back cushions", [0.08, 0.05, 0.92, 0.45], cls="fabric", deco="upholstered", complexity="complex"),
        C("arm_l", "Left armrest", [0, 0.25, 0.08, 0.8], cls="fabric", deco="upholstered"), C("arm_r", "Right armrest", [0.92, 0.25, 1, 0.8], cls="fabric", deco="upholstered")],
        (2.2, 0.85, 0.95)), "furniture", "sofa"),
    "armchair": (reading("armchair", [C("frame", "Armchair base", [0, 0.85, 1, 1], cls="solid_wood"),
        C("seat", "Seat cushion", [0.15, 0.5, 0.85, 0.85], cls="fabric", deco="upholstered", complexity="complex"),
        C("back", "Back", [0.15, 0, 0.85, 0.5], cls="fabric", deco="upholstered", complexity="complex"),
        C("arm_l", "Left arm", [0, 0.3, 0.15, 0.85], cls="fabric", deco="upholstered"), C("arm_r", "Right arm", [0.85, 0.3, 1, 0.85], cls="fabric", deco="upholstered")],
        (0.85, 0.9, 0.85)), "furniture", "armchair"),
}


def build(name):
    spec, category, kind = CASES[name]
    glb, doc = hybrid.build(F.photo(), spec, category, kind)
    if glb is None:
        return None, doc
    doc.pop("_pendingImages")
    return model_validation.validate_and_repair(glb, doc)


@pytest.mark.parametrize("name", ["bed", "table", "table_curved_legs", "desk", "cabinet", "bookshelf", "chair"])
def test_rebuildable_types_are_built_validated_and_modular(name):
    glb, doc = build(name)
    assert glb is not None, doc.get("reason")
    assert doc["validation"]["ok"], doc["validation"]["errors"]
    assert doc["separated"] and len(doc["components"]) + len(doc.get("pendingComponents") or []) >= 3
    assert len({c["id"] for c in doc["components"]}) == len(doc["components"])


def test_what_moves_is_only_what_the_photo_says_moves():
    for name, movers in (("table", set()), ("bookshelf", set()), ("desk", {"Top drawer"}), ("cabinet", {"Left door", "Right door", "Drawer"})):
        _, doc = build(name)
        got = {a["label"] for a in doc["animation"]["animations"] if a["enabled"]}
        assert got == movers, (name, got)
    _, desk = build("desk")
    drawer = next(a for a in desk["animation"]["animations"] if a["label"] == "Top drawer")
    assert any("handle" in f for f in drawer["follow"])  # the drawer's own handle slides with it


def test_organic_parts_go_to_the_mesh_provider_one_by_one_and_the_rest_is_built():
    for name, generated in (("bed", {"Carved headboard"}), ("table_curved_legs", {"Leg 1", "Leg 2", "Leg 3", "Leg 4"}), ("chair", {"Upholstered seat"})):
        _, doc = build(name)
        assert {p["name"] for p in doc["pendingComponents"]} == generated, name
        assert not any(c["name"] in generated for c in doc["components"])  # never a block in their place
    _, table = build("table_curved_legs")
    assert len({p["group"] for p in table["pendingComponents"]}) == 1  # four identical legs: one generation


def test_regular_decoration_is_geometry_on_any_type():
    _, chair = build("chair")
    _, cabinet = build("cabinet")
    assert any(d["decoration"] == "recessed_panel" and d["method"] == "geometry" for d in cabinet["hybrid"]["decorations"])
    grooved = next(d for d in chair["hybrid"]["decorations"] if d["decoration"] == "grooved")
    assert grooved["method"] in ("geometry", "not built")  # built only where the photo shows the lines


@pytest.mark.parametrize("name", ["sofa", "armchair"])
def test_mostly_upholstered_pieces_go_to_the_whole_model_generator_with_the_reason(name):
    glb, doc = build(name)
    assert glb is None and "upholstered" in doc["reason"]
