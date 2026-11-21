"""Carpenter-friendly components: fragments folded, AI groupings validated against the geometry, one mesh per
component, and the model's geometry, UVs and materials unchanged (any furniture type, any mesh structure)."""

import asyncio
import json

import numpy as np
import pytest

from app.services import component_grouping as cg
from app.services.glb_size import _views, read_glb, write_glb
from app.services.realistic_materials import _accessor


def box(lo, hi):
    """A closed box: 24 vertices (flat normals, UVs) and 12 triangles."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    faces = [
        ([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0, 0, 1]), ([[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [0, 0, -1]),
        ([[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [1, 0, 0]), ([[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], [-1, 0, 0]),
        ([[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [0, 1, 0]), ([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], [0, -1, 0]),
    ]
    pos, nor, uv, idx = [], [], [], []
    for quad, n in faces:
        base = len(pos)
        pos += quad
        nor += [n] * 4
        uv += [[0, 0], [1, 0], [1, 1], [0, 1]]
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(pos, np.float32), np.array(nor, np.float32), np.array(uv, np.float32), np.array(idx, np.uint16)


def glb(pieces, materials=None, uv=True):
    """pieces: (lo, hi, material, name, node extras such as translation / scale)."""
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "accessors": [], "bufferViews": [],
           "buffers": [{"byteLength": 0}], "materials": materials or [{"name": "wood", "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.3, 0.2, 1]}}]}
    views = []

    def add(arr, kind, ctype):
        views.append(arr.tobytes())
        doc["bufferViews"].append({"buffer": 0, "byteOffset": 0, "byteLength": arr.nbytes})
        acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": len(arr), "type": kind}
        if kind == "VEC3":
            acc["min"], acc["max"] = arr.min(axis=0).tolist(), arr.max(axis=0).tolist()
        doc["accessors"].append(acc)
        return len(doc["accessors"]) - 1

    for lo, hi, material, name, node in pieces:
        p, n, t, i = box(lo, hi)
        attrs = {"POSITION": add(p, "VEC3", 5126), "NORMAL": add(n, "VEC3", 5126)}
        if uv:
            attrs["TEXCOORD_0"] = add(t, "VEC2", 5126)
        doc["meshes"].append({"name": name, "primitives": [{"attributes": attrs, "indices": add(i, "SCALAR", 5123), "material": material}]})
        doc["nodes"].append({"name": name, "mesh": len(doc["meshes"]) - 1, **node})
        doc["scenes"][0]["nodes"].append(len(doc["nodes"]) - 1)
    return write_glb(doc, views)


BED = [  # 2 × 0.5 × 1.6 m (+ headboard): generic Meshy names, a speck on the frame
    ((-1, 0.1, -0.8), (1, 0.3, 0.8), 0, "model_part0", {}),  # frame
    ((-0.95, 0.3, -0.75), (0.95, 0.5, 0.75), 0, "model_part1", {}),  # mattress
    ((-1, 0.1, -0.85), (1, 1.1, -0.8), 0, "model_part2", {}),  # headboard
    ((-1, 0, -0.8), (-0.9, 0.1, -0.7), 0, "Mesh_001", {}), ((0.9, 0, -0.8), (1, 0.1, -0.7), 0, "Mesh_002", {}),
    ((-1, 0, 0.7), (-0.9, 0.1, 0.8), 0, "Mesh_003", {}), ((0.9, 0, 0.7), (1, 0.1, 0.8), 0, "Mesh_004", {}),  # legs
    ((0.5, 0.3, 0.79), (0.52, 0.31, 0.8), 0, "Object_023", {}),  # a speck on the frame
]


def answer(*components, kind="double bed"):
    async def ask(content, schema):
        ask.calls.append(content)
        return json.dumps({"furniture_type": kind, "components": list(components)})
    ask.calls = []
    return ask


def run(data, ask, **kw):
    return asyncio.run(cg.group(data, b"\x89PNG photo", ask, **kw))


def totals(data):
    tris, verts, lo, hi = cg._totals(data)
    return tris, verts, lo.round(5).tolist(), hi.round(5).tolist()


def by_name(doc):
    return {c["name"]: c for c in doc["components"]}


def test_fragments_fold_into_the_part_they_touch_only():
    _, _, found = cg.parts(glb(BED))
    assert len(found) == 7  # 8 meshes, the speck folded into the frame
    frame = next(p for p in found if p.name == "model_part0")
    assert len(frame.members) == 2


def test_a_bed_becomes_named_carpenter_components_with_the_same_geometry():
    data = glb(BED)
    _, _, found = cg.parts(data)
    ids = {p.name: p.id for p in found}
    ask = answer(
        {"name": "Bed Frame", "type": "frame", "parts": [ids["model_part0"]], "confidence": 0.9},
        {"name": "Mattress", "type": "mattress", "parts": [ids["model_part1"]], "confidence": 0.95},
        {"name": "headboard", "type": "headboard", "parts": [ids["model_part2"]], "confidence": 0.9},
        {"name": "Legs", "type": "leg", "parts": [ids[f"Mesh_00{i}"] for i in range(1, 5)], "confidence": 0.85},
    )
    out, doc = run(data, ask, real_height_m=1.1)
    assert len(ask.calls[0]) == 4  # the question, the photo, the look and the numbered map
    names = by_name(doc)
    assert set(names) == {"Bed Frame", "Mattress", "Headboard", "Legs"}  # four legs are one component; no Mesh_001
    assert names["Legs"]["count"] == 4 and names["Legs"]["type"] == "leg" and names["Legs"]["confidence"] == 0.85
    assert names["Headboard"]["size"]["m"] == [2.0, 1.0, 0.05]  # metres from the geometry and the real height
    assert names["Mattress"]["position"][1] > names["Legs"]["position"][1]
    assert names["Bed Frame"]["materialInfo"]["name"] == "wood"
    assert totals(out) == totals(data)  # every triangle and vertex, same bounds
    doc_out, binary = read_glb(out)
    assert len(doc_out["meshes"]) == 4 and doc_out["materials"] == read_glb(data)[0]["materials"]
    tags = {n["extras"]["partId"] for n in doc_out["nodes"] if "mesh" in n}
    assert tags == {c["id"] for c in doc["components"]}  # each component selectable by its id in the editor
    assert doc["grouping"]["partsBefore"] == 8 and doc["grouping"]["components"] == 4


def test_uvs_and_normals_survive_baked_node_transforms_including_a_mirror():
    pieces = [((0, 0, 0), (1, 0.05, 0.6), 0, "top", {"translation": [0, 0.7, 0]}),
              ((0, 0, 0), (0.05, 0.7, 0.05), 0, "leg_a", {}),
              ((0, 0, 0), (0.05, 0.7, 0.05), 0, "leg_b", {"scale": [-1, 1, 1], "translation": [1, 0, 0]})]  # mirrored copy
    data = glb(pieces)
    _, _, found = cg.parts(data)
    ids = {p.name: p.id for p in found}
    out, doc = run(data, answer({"name": "Tabletop", "type": "top", "parts": [ids["top"]], "confidence": 0.9},
                                {"name": "Legs", "type": "leg", "parts": [ids["leg_a"], ids["leg_b"]], "confidence": 0.9}, kind="side table"))
    assert totals(out) == totals(data)
    d, binary = read_glb(out)
    views = _views(d, binary)
    legs = next(m for m in d["meshes"] if m["name"] == "Legs")["primitives"][0]
    uv = _accessor(d, views, legs["attributes"]["TEXCOORD_0"])
    assert uv.min() == 0 and uv.max() == 1 and len(uv) == 48  # both legs' UVs, untouched
    normals = _accessor(d, views, legs["attributes"]["NORMAL"])
    assert np.allclose(np.linalg.norm(normals, axis=1), 1)
    # The mirrored leg's faces still point outwards: each triangle's winding agrees with its normal.
    pos = _accessor(d, views, legs["attributes"]["POSITION"])
    tri = _accessor(d, views, legs["indices"])[:, 0].astype(int).reshape(-1, 3)
    face = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    assert np.all((face * normals[tri[:, 0]]).sum(axis=1) > 0)


def test_unsafe_answers_are_refused_and_flagged():
    data = glb(BED)
    _, _, found = cg.parts(data)
    ids = {p.name: p.id for p in found}
    out, doc = run(data, answer(
        {"name": "Mesh 001", "type": "other", "parts": [ids["model_part1"]], "confidence": 0.9},  # a generic name
        {"name": "Bed Frame", "type": "frame", "parts": [ids["model_part0"], "P99"], "confidence": 0.9},  # an unknown id ignored
        {"name": "Frame Again", "type": "frame", "parts": [ids["model_part0"]], "confidence": 0.9},  # already taken
        {"name": "Headboard", "type": "headboard", "parts": [ids["model_part2"]], "confidence": 0.3},  # too unsure
        {"name": "Legs", "type": "leg", "parts": [ids["Mesh_001"], ids["Mesh_004"]], "confidence": 0.6},  # unsure: flagged
        {"name": "Foot", "type": "leg", "parts": [ids["Mesh_002"], ids["model_part1"]], "confidence": 0.9},  # not one piece, not repeats
    ))
    names = by_name(doc)
    assert "Mesh 001" not in names and "Frame Again" not in names
    assert names["Bed Frame"]["naming"] == "ai" and names["Legs"].get("uncertain") is True
    assert {n for n in names if n.startswith("Unidentified Part")}  # what was not identified says so
    assert any("not a usable component name" in w for w in doc["grouping"]["warnings"])
    assert any("separate places" in w for w in doc["grouping"]["warnings"])  # Foot: split per place, not one span
    assert names["Foot 1"]["count"] == 1 and names["Foot 2"]["count"] == 1  # one per place, numbered
    assert totals(out) == totals(data)


def test_a_component_with_two_materials_stays_one_component_in_the_file():
    mats = [{"name": "oak", "pbrMetallicRoughness": {"baseColorFactor": [0.6, 0.4, 0.2, 1]}},
            {"name": "brass", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.7, 0.3, 1], "metallicFactor": 1}}]
    pieces = [((0, 0, 0), (1, 2, 0.6), 0, "body", {}), ((0, 0, 0.6), (0.5, 2, 0.62), 0, "door_left", {}),
              ((0.4, 1, 0.62), (0.45, 1.2, 0.66), 1, "knob", {})]
    data = glb(pieces, mats)
    _, _, found = cg.parts(data)
    ids = {p.name: p.id for p in found}
    out, doc = run(data, answer({"name": "Wardrobe Body", "type": "frame", "parts": [ids["body"]], "confidence": 0.9},
                                {"name": "Left Door", "type": "door", "parts": [ids["door_left"], ids["knob"]], "confidence": 0.8}, kind="wardrobe"))
    d, _ = read_glb(out)
    door = next(n for n in d["nodes"] if n.get("name") == "Left Door" and "children" in n)  # one component node…
    tagged = [d["nodes"][i]["extras"]["partId"] for i in door["children"]]
    assert tagged == ["left_door", "left_door_2"]  # …a selectable mesh per material, each tagged
    assert {c["materialInfo"]["name"] for c in doc["components"] if c["component"] == "left_door"} == {"oak", "brass"}
    assert d["materials"] == mats


def test_the_model_is_kept_as_it_was_when_grouping_is_not_safe():
    no_ai = answer()
    assert run(glb(BED), no_ai) is None  # nothing identified: the split model stays
    assert asyncio.run(cg.group(glb(BED), None, None)) is None  # no vision model configured
    assert run(b"junk", answer()) is None
    assert run(glb(BED[:1]), answer()) is None  # one part: nothing to group
    mixed = glb(BED[:2])  # the frame with UVs, the mattress without: never merged without its UVs
    d, binary = read_glb(mixed)
    del d["meshes"][1]["primitives"][0]["attributes"]["TEXCOORD_0"]
    mixed = write_glb(d, _views(d, binary))
    _, _, found = cg.parts(mixed)
    assert run(mixed, answer({"name": "Bed", "type": "frame", "parts": [p.id for p in found], "confidence": 0.9})) is None


@pytest.mark.parametrize("name, expected", [("left side panel", "Left Side Panel"), ("Mesh_001", None), ("Object 23", None), ("Part 4", None),
                                            ("TV Unit Doors", "TV Unit Doors"), ("x" * 60, None), ("شماعة", None)])
def test_component_names_are_understandable_english(name, expected):
    assert cg.clean_name(name) == expected


def test_a_set_of_doors_is_one_component_with_one_mesh_per_door():
    pieces = [((0, 0, 0), (1.2, 2, 0.6), 0, "body", {}), ((0.6, 0, 0.6), (1.2, 2, 0.62), 0, "d2", {}), ((0, 0, 0.6), (0.6, 2, 0.62), 0, "d1", {}),
              ((0.1, 0.1, 0.62), (0.15, 0.12, 0.64), 0, "leg_speck", {}),  # a speck on the left door
              ((0, -0.1, 0), (0.05, 0, 0.05), 0, "l1", {}), ((1.15, -0.1, 0), (1.2, 0, 0.05), 0, "l2", {}),
              ((0, -0.1, 0.55), (0.05, 0, 0.6), 0, "l3", {}), ((1.15, -0.1, 0.55), (1.2, 0, 0.6), 0, "l4", {})]
    data = glb(pieces)
    _, _, found = cg.parts(data)
    ids = {p.name: p.id for p in found}
    out, doc = run(data, answer({"name": "Carcass", "type": "frame", "parts": [ids["body"]], "confidence": 0.9},
                                {"name": "Doors", "type": "door", "parts": [ids["d2"], ids["d1"]], "confidence": 0.9},
                                {"name": "Legs", "type": "leg", "parts": [ids[f"l{i}"] for i in range(1, 5)], "confidence": 0.9}, kind="wardrobe"))
    d, _ = read_glb(out)
    doors = next(n for n in d["nodes"] if n.get("name") == "Doors")
    assert [d["nodes"][i]["name"] for i in doors["children"]] == ["Door 1", "Door 2"]  # each opens on its own, left to right
    assert [d["nodes"][i]["extras"]["partId"] for i in doors["children"]] == ["doors_1", "doors_2"]
    legs = [n for n in d["nodes"] if n.get("name") == "Legs"]
    assert len(legs) == 1 and "mesh" in legs[0]  # static repeats: one mesh
    names = by_name(doc)
    assert names["Door 1"]["position"][0] < names["Door 2"]["position"][0] and names["Legs"]["count"] == 4
    assert totals(out) == totals(data)


def test_plural_names_become_one_element_names():
    assert [cg.singular(n) for n in ["Legs", "Shelves", "Drawers", "Hanging Clothes", "Glass", "Bench", "Boxes"]] == ["Leg", "Shelf", "Drawer", "Hanging Clothes", "Glass", "Bench", "Box"]
