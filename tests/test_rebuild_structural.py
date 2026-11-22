"""scripts/rebuild_structural.py: the saved photo links of a mirrored door repaired by layer, deterministically."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rebuild_structural import details_of, relink  # noqa: E402


def test_links_on_the_wrong_layer_are_dropped_and_the_right_ones_kept():
    doc = {
        "analysis": {"components": [
            {"id": "door_far_left", "name": "Far-left door frame", "material": "walnut"},
            {"id": "mirror_center_right", "name": "Central right door mirror", "material": "mirror", "materialClass": "mirror"},
            {"id": "door_center_right", "name": "Central right open door frame", "material": "walnut"},
        ]},
        "components": [
            {"id": "left_outer_mirrored_door_wood_frame", "name": "Far-left door frame", "naming": "vision", "analysisId": "door_far_left"},
            {"id": "right_inner_mirrored_door_mirror_backing", "name": "Central right door mirror", "nameAr": "مرآة", "naming": "vision", "analysisId": "mirror_center_right"},
            {"id": "right_inner_mirrored_door_mirror", "name": "Central right open door frame", "naming": "vision", "analysisId": "door_center_right"},
            {"id": "hinge_1", "name": "Hinge 1", "naming": "file"},
        ],
    }
    changed = relink(doc)
    assert len(changed) == 2
    by = {c["id"]: c for c in doc["components"]}
    assert by["left_outer_mirrored_door_wood_frame"]["analysisId"] == "door_far_left"
    assert by["right_inner_mirrored_door_mirror_backing"] == {"id": "right_inner_mirrored_door_mirror_backing", "name": "Right inner mirrored door mirror backing", "naming": "file"}
    assert by["right_inner_mirrored_door_mirror"]["name"] == "Right inner mirrored door mirror" and "analysisId" not in by["right_inner_mirrored_door_mirror"]
    assert relink(doc) == []  # run again: nothing left to change


def test_the_saved_size_is_the_rebuild_s_size():
    assert details_of({"carpenter": {"sizeM": {"width": 2.4, "height": 2.3, "depth": 0.6}, "scaleKnown": True}}) == {
        "width_m": 2.4, "height_m": 2.3, "depth_m": 0.6, "dimension_source": "annotated"}
    assert details_of({}) is None
