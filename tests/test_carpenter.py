"""Carpenter reconstruction: closed solids from measured boxes, decorative parts left out, the source list untouched."""

import copy

from app.services.carpenter import reconstruct
from app.services.glb_size import read_glb


def _part(name, kind, box):
    return {"id": name.lower(), "name": name, "type": kind, "box": box}


def test_a_bed_drops_bedding_and_marks_missing_wood_as_inferred():
    components = {"components": [
        _part("Headboard", "headboard", [[0, 0.4, 0], [1.8, 1.2, 0.05]]),
        _part("Mattress", "mattress", [[0.05, 0.25, 0.08], [1.75, 0.55, 1.9]]),
        _part("Pillow", "cushion", [[0.2, 0.55, 0.1], [1.6, 0.75, 0.4]]),
    ]}
    before = copy.deepcopy(components)
    built = reconstruct(components, {"width_m": 1.8, "height_m": 1.2, "depth_m": 2.0}, kind="bed", category="furniture")
    assert components == before
    assert built is not None
    glb, report = built
    assert report["fabricationReady"] is False
    assert report["estimated"] is True
    names = {c["name"] for c in report["components"]}
    assert "Mattress" not in names and "Pillow" not in names
    assert {e["name"] for e in report["excluded"]} == {"Mattress", "Pillow"}
    head = next(c for c in report["components"] if c["name"] == "Headboard")
    assert head["source"] == "observed" and head["materialCategory"] == "wood"
    assert any(c["source"] == "inferred" and c["role"] == "footboard" for c in report["components"])
    assert report["checks"]["closed"] is True
    doc, _ = read_glb(glb)
    assert len(doc["meshes"]) == len(report["components"])
    assert len({m["primitives"][0]["attributes"]["POSITION"] for m in doc["meshes"]}) == len(doc["meshes"])
    assert report["hierarchy"]["name"] == "Furniture Model"


def test_an_unknown_object_keeps_the_panel_and_invents_no_structure():
    built = reconstruct({"components": [
        _part("Vase", "decor", [[0, 0, 0], [0.2, 0.4, 0.2]]),
        _part("Panel", "panel", [[0, 0, 0], [1, 0.8, 0.02]]),
    ]}, kind="sculpture", category="decor")
    assert built is not None
    _, report = built
    assert [c["name"] for c in report["components"]] == ["Panel"]
    assert report["components"][0]["source"] == "observed"
    assert report["excluded"][0]["name"] == "Vase"
    assert report["furnitureType"] == "decor"


def test_known_dimensions_set_the_carpenter_size_and_clear_the_estimate():
    built = reconstruct(
        {"components": [_part("Top", "top", [[0, 0, 0], [2, 1, 1]])]},
        {"width_m": 1.0, "height_m": 0.5, "depth_m": 0.4, "scale_known": True},
        kind="panel",
        category="furniture",
    )
    assert built is not None
    glb, report = built
    assert report["estimated"] is False
    assert report["components"][0]["sizeCm"] == {"width": 100.0, "height": 50.0, "depth": 40.0}
    doc, binary = read_glb(glb)
    view = doc["bufferViews"][0]
    positions = binary[view["byteOffset"]: view["byteOffset"] + view["byteLength"]]
    import numpy as np
    verts = np.frombuffer(positions, dtype=np.float32).reshape(-1, 3)
    assert abs(float(verts[:, 0].max()) - 1.0) < 0.01
    assert abs(float(verts[:, 1].max()) - 0.5) < 0.01


def test_glass_stays_and_identical_solids_are_flagged():
    built = reconstruct({"components": [
        _part("Mirror", "glass", [[0.2, 0.2, 0], [0.8, 1.2, 0.01]]),
        _part("Left door", "door", [[0, 0, 0], [0.5, 1.2, 0.02]]),
        _part("Left door copy", "door", [[0, 0, 0], [0.5, 1.2, 0.02]]),
    ]}, kind="wardrobe", category="furniture")
    assert built is not None
    _, report = built
    mirror = next(c for c in report["components"] if c["name"] == "Mirror")
    assert mirror["materialCategory"] == "glass" and mirror["source"] == "observed"
    assert any("occupy the same space" in warning for warning in report["checks"]["warnings"])


def test_a_mirror_pane_keeps_its_thickness_in_front_of_its_backing():
    """Padded to an 18 mm board, a 5 mm mirror would reach through its wooden backing and show on the inside face."""
    _, report = reconstruct({"components": [
        _part("Door mirror", "glass", [[0.1, 0.1, 0.3125], [0.9, 1.9, 0.3175]]),
        _part("Door mirror backing", "door", [[0.1, 0.1, 0.298], [0.9, 1.9, 0.310]]),
    ]}, kind="wardrobe", category="furniture")
    by = {c["name"]: c for c in report["components"]}
    assert by["Door mirror"]["sizeCm"]["depth"] == 0.6  # 5 mm read, the 6 mm pane minimum; never a board
    mirror_back = by["Door mirror"]["positionCm"]["z"] - 0.3
    backing_front = by["Door mirror backing"]["positionCm"]["z"] + by["Door mirror backing"]["sizeCm"]["depth"] / 2
    assert mirror_back >= backing_front - 0.05  # the pane stays in front of the wood behind it


def test_nothing_measurable_makes_no_carpenter_file():
    assert reconstruct({"components": [{"name": "Blob", "type": "other"}]}) is None
