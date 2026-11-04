import json

import anyio
import trimesh

from app.services.glb_parts import _gltf_json, inspect_glb, write_names
from app.services.part_names import is_generic, name_components


def _box(w, h, d, x=0.0, y=0.0, z=0.0):
    return trimesh.creation.box(extents=(w, h, d), transform=trimesh.transformations.translation_matrix((x, y + h / 2, z)))


def _sofa_glb() -> bytes:
    """Like Meshy Auto Split output: one node per part, all named "Model partN"."""
    parts = [_box(2.0, 0.8, 0.9)]  # body
    parts += [_box(0.05, 0.1, 0.05, x=sx * 0.95, z=sz * 0.4) for sx in (-1, 1) for sz in (-1, 1)]  # legs
    parts += [_box(0.6, 0.15, 0.6, x=x, y=0.45, z=0.1) for x in (-0.6, 0, 0.6)]  # seat cushions
    scene = trimesh.Scene()
    for i, mesh in enumerate(parts):
        scene.add_geometry(mesh, geom_name=f"Model part{i}", node_name=f"Model part{i}")
    return scene.export(file_type="glb")


def test_generic_names_are_detected():
    assert all(is_generic(n) for n in ["Model part0", "Model part 12", "Part 3", "mesh_2", "Whole model", ""])
    assert not any(is_generic(n) for n in ["Leg 1", "Tabletop", "Seat cushion"])


def test_sofa_parts_get_names_from_geometry_and_confident_ai_answers_only():
    glb = _sofa_glb()
    doc = inspect_glb(glb, "sofa")
    ids_before = [c["id"] for c in doc["components"]]
    cushions = [c["id"] for c in doc["components"][5:]]
    asked = []

    async def ask(prompt, schema):
        asked.append(prompt)
        # Two confident cushions, one unsure, one name outside the vocabulary.
        return json.dumps({"parts": [
            {"id": cushions[0], "name": "Seat Cushion", "confidence": 0.9},
            {"id": cushions[1], "name": "Seat Cushion", "confidence": 0.8},
            {"id": cushions[2], "name": "Seat Cushion", "confidence": 0.4},
            {"id": doc["components"][0]["id"], "name": "Spaceship", "confidence": 1.0},
        ]})

    named = anyio.run(name_components, doc, "sofa", ask)
    names = [c["name"] for c in named["components"]]
    assert names[0] == "Frame" and named["components"][0]["naming"] == "geometry"
    assert names[1:5] == ["Leg 1", "Leg 2", "Leg 3", "Leg 4"]
    assert names[5:7] == ["Seat Cushion 1", "Seat Cushion 2"]
    assert names[7] == "Part 1" and named["components"][7]["naming"] == "unidentified"  # not guessed
    assert [c["id"] for c in named["components"]] == ids_before  # ids are stable
    assert len(asked) == 1

    out = write_names(glb, named["components"])
    gltf = _gltf_json(out)
    assert {n["name"] for n in gltf["nodes"] if "mesh" in n} >= {"Frame", "Leg 1", "Seat Cushion 2", "Part 1"}
    loaded = trimesh.load(trimesh.util.wrap_as_stream(out), file_type="glb", force="scene")
    assert len(loaded.geometry) == 8  # the binary chunk is intact


def test_without_ai_only_geometry_names_are_used():
    doc = anyio.run(name_components, inspect_glb(_sofa_glb(), "sofa"), "sofa", None)
    assert [c["naming"] for c in doc["components"]].count("unidentified") == 3
