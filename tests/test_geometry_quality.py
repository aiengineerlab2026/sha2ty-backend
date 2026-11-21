"""Geometry validation, safe repair, material isolation, classes, sizes and the AR copy (difficult geometry included)."""

import json

import numpy as np
import pytest

from app.services import geometry_quality as gq
from app.services.glb_size import _views, read_glb, write_glb
from app.services.realistic_materials import _accessor
from tests.test_component_grouping import box, glb


def raw_glb(prims, materials=None, extra=None):
    """prims: (positions, indices, normals or None, uvs or None, material, name). One node + mesh each."""
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "accessors": [], "bufferViews": [],
           "buffers": [{"byteLength": 0}], "materials": materials or [{"name": "oak", "pbrMetallicRoughness": {"baseColorFactor": [0.6, 0.4, 0.2, 1]}}], **(extra or {})}
    views = []

    def add(arr, kind, ctype):
        views.append(arr.tobytes())
        doc["bufferViews"].append({"buffer": 0, "byteOffset": 0, "byteLength": arr.nbytes})
        acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": len(arr), "type": kind}
        if kind == "VEC3":
            acc["min"], acc["max"] = np.nanmin(arr, axis=0).tolist(), np.nanmax(arr, axis=0).tolist()
        doc["accessors"].append(acc)
        return len(doc["accessors"]) - 1

    for pos, idx, nor, uv, material, name in prims:
        attrs = {"POSITION": add(np.asarray(pos, np.float32), "VEC3", 5126)}
        if nor is not None:
            attrs["NORMAL"] = add(np.asarray(nor, np.float32), "VEC3", 5126)
        if uv is not None:
            attrs["TEXCOORD_0"] = add(np.asarray(uv, np.float32), "VEC2", 5126)
        doc["meshes"].append({"name": name, "primitives": [{"attributes": attrs, "indices": add(np.asarray(idx, np.uint32).reshape(-1), "SCALAR", 5125), "material": material}]})
        doc["nodes"].append({"name": name, "mesh": len(doc["meshes"]) - 1})
        doc["scenes"][0]["nodes"].append(len(doc["nodes"]) - 1)
    return write_glb(doc, views)


def components(data, names):
    return {"separated": True, "components": [{"id": n.lower().replace(" ", "_"), "name": n, "mesh": i, "primitive": 0} for i, n in enumerate(names)]}


def sphere(n=100, radius=0.5, centre=(0, 0.5, 0)):
    """A UV sphere: n × n quads (2n² triangles), positions, normals, UVs (a seam where u wraps)."""
    u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
    theta, phi = u * 2 * np.pi, v * np.pi
    nor = np.stack([np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)], axis=-1).reshape(-1, 3)
    pos = nor * radius + np.array(centre)
    uv = np.stack([u, v], axis=-1).reshape(-1, 2)
    idx = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1
            idx += [a, c, b, b, c, d]
    return pos, np.array(idx), nor, uv


def test_invisible_defects_are_repaired_and_the_shape_kept():
    p, n, t, i = box((0, 0, 0), (1, 1, 1))
    n = n.copy()
    n[0] = np.nan  # a broken normal
    tris = np.concatenate([i, i[:3], [0, 0, 1], [0, 1, 2], [1, 2, 0]])  # the first triangle three more times (once rotated), a repeated index
    data = raw_glb([(p, tris, n, t, 0, "Cube")])
    out, doc, _ = gq.process(data, components(data, ["Cabinet Body"]))
    report = doc["quality"]
    stats = report["parts"]["cabinet_body"]
    assert stats["degenerate"] == 1 and stats["duplicate"] == 3 and stats["invalidNormals"] == 1
    assert any("zero-area" in r for r in report["repairs"]) and any("duplicate" in r for r in report["repairs"]) and any("normal" in r for r in report["repairs"])
    d, b = read_glb(out)
    views = _views(d, b)
    prim = d["meshes"][0]["primitives"][0]
    assert _accessor(d, views, prim["indices"]).size == 36  # the box's 12 triangles, nothing else
    assert np.isfinite(_accessor(d, views, prim["attributes"]["NORMAL"])).all()
    assert d["accessors"][prim["attributes"]["POSITION"]]["min"] == [0, 0, 0] and d["accessors"][prim["attributes"]["POSITION"]]["max"] == [1, 1, 1]
    assert stats["afterRepair"]["degenerate"] == 0 and stats["afterRepair"]["watertight"]


def test_open_surfaces_are_reported_never_closed():
    seat = ([[0, 0.45, 0], [1, 0.45, 0], [1, 0.45, 1], [0, 0.45, 1]], [0, 1, 2, 0, 2, 3], None, [[0, 0], [1, 0], [1, 1], [0, 1]], 0, "Seat")
    data = raw_glb([seat])
    out, doc, _ = gq.process(data, components(data, ["Seat Panel"]))
    stats = doc["quality"]["parts"]["seat_panel"]
    assert stats["openEdges"] == 4 and not stats["watertight"]
    assert out is data  # nothing invisible to repair: the file is untouched (no hole closed)
    assert any("open edges" in w for w in doc["quality"]["warnings"])


def test_uv_seams_are_not_open_edges_and_pieces_are_counted():
    pos, idx, nor, uv = sphere(20)
    a, b = box((2, 0, 0), (2.5, 0.5, 0.5))[0], box((2, 0, 0), (2.5, 0.5, 0.5))[3]
    data = raw_glb([(np.concatenate([pos, a]), np.concatenate([idx, b + len(pos)]), None, None, 0, "Pieces")])
    _, doc, _ = gq.process(data, components(data, ["Two Pieces"]))
    stats = doc["quality"]["parts"]["two_pieces"]
    assert stats["pieces"] == 2
    assert stats["openEdges"] == 0  # the sphere's duplicated seam / pole vertices are welded for topology, not holes


def test_non_manifold_edges_are_left_and_flagged():
    fin = [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0, 1]]
    data = raw_glb([(fin, [0, 1, 2, 1, 0, 3, 0, 1, 4], None, None, 0, "Fin")])  # three triangles on one edge
    out, doc, _ = gq.process(data, components(data, ["Bracket"]))
    assert doc["quality"]["parts"]["bracket"]["nonManifoldEdges"] == 1
    assert any("non-manifold" in u for u in doc["quality"]["unresolved"])
    assert out is data


def test_parts_sharing_a_material_get_their_own_copy_and_keep_the_texture():
    mats = [{"name": "linen", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    a, b = box((0, 0, 0), (1, 0.4, 1)), box((0, 0.4, 0), (1, 0.6, 1))
    data = raw_glb([(a[0], a[3], a[1], a[2], 0, "Base"), (b[0], b[3], b[1], b[2], 0, "Cushion")], mats, {"textures": [{"source": 0}], "images": []})
    out, doc, _ = gq.process(data, components(data, ["Sofa Base", "Seat Cushion"]))
    d, _ = read_glb(out)
    used = [m["primitives"][0]["material"] for m in d["meshes"]]
    assert used[0] != used[1] and len(d["materials"]) == 2
    assert all(m["pbrMetallicRoughness"]["baseColorTexture"] == {"index": 0} for m in d["materials"])  # one texture, shared
    assert [c["material"] for c in doc["components"]] == used


@pytest.mark.parametrize("name, kind, expected", [
    ("Headboard", "headboard", "structural"), ("Legs", "leg", "structural"), ("Seat Cushion", "cushion", "upholstery"),
    ("Upholstered Headboard Panel", "headboard", "upholstery"), ("Throw Pillows", "other", "decorative"), ("Mattress", "mattress", "decorative"),
    ("Hanging Clothes", "decor", "decorative"), ("Wardrobe Carcass", "frame", "structural"), ("Unidentified Part 1", "other", "other"),
    ("لوح رأس منجد", None, "upholstery"), ("أرجل", None, "structural"), ("مخدة", None, "decorative"),
])
def test_components_are_classed_for_the_carpenter(name, kind, expected):
    assert gq.classify({"name": name, "type": kind}) == expected


def test_sizes_are_bounding_boxes_in_cm_only_with_a_real_size():
    a, b = box((0, 0, 0), (2, 0.3, 1.6)), box((0, 0.3, -0.05), (2, 1.2, 0))
    data = raw_glb([(a[0], a[3], None, None, 0, "frame"), (b[0], b[3], None, None, 0, "board")])
    _, doc, _ = gq.process(data, components(data, ["Bed Frame", "Headboard"]), real_height_m=1.2)
    head = next(c for c in doc["components"] if c["name"] == "Headboard")["dimensions"]
    assert head["cm"] == [200.0, 90.0, 5.0] and head["basis"] == "bounding box" and head["calibrated"] is False
    _, doc2, _ = gq.process(data, components(data, ["Bed Frame", "Headboard"]))
    assert doc2["components"][1]["dimensions"]["cm"] is None and "not known" in " ".join(doc2["quality"]["warnings"])
    assert "Not assessed" in doc["quality"]["manufacturing"]  # never claims manufacturing readiness


def test_the_ar_copy_is_lighter_with_the_same_shape_and_original_uvs():
    pos, idx, nor, uv = sphere(140)  # 39 200 triangles of smooth curve
    data = raw_glb([(pos, idx, nor, uv, 0, "Ball")])
    _, doc, job = gq.process(data, components(data, ["Ball Finial"]))
    assert job is not None
    out, info = gq.ar_copy(job)
    assert info["triangles"] <= 0.5 * info["original"]
    d, b = read_glb(out)
    views = _views(d, b)
    prim = d["meshes"][0]["primitives"][0]
    p = _accessor(d, views, prim["attributes"]["POSITION"])
    assert np.allclose(p.min(axis=0), pos.min(axis=0), atol=0.005) and np.allclose(p.max(axis=0), pos.max(axis=0), atol=0.005)
    new_uv = _accessor(d, views, prim["attributes"]["TEXCOORD_0"])
    original = {tuple(np.round(x, 5)) for x in np.c_[pos, uv]}
    assert all(tuple(np.round(x, 5)) in original for x in np.c_[p, new_uv])  # every vertex is an original one, with its own UV
    assert info["parts"]["0_0"]["surfaceMatch"] >= gq.SURFACE_MATCH


def test_light_models_get_no_ar_copy_and_compressed_files_are_left_alone():
    a = box((0, 0, 0), (1, 1, 1))
    data = raw_glb([(a[0], a[3], a[1], a[2], 0, "Cube")])
    assert gq.process(data, components(data, ["Cube"]))[2] is None
    d, b = read_glb(data)
    d["extensionsUsed"] = ["KHR_draco_mesh_compression"]
    packed = write_glb(d, _views(d, b))
    out, doc, job = gq.process(packed, components(packed, ["Cube"]))
    assert out is packed and job is None and "not checked" in doc["quality"]["unresolved"][0]


def test_a_grouped_model_from_the_previous_stage_is_processed_whole():
    from tests.test_component_grouping import BED, answer, run
    from app.services import component_grouping as cg

    data = glb(BED)
    _, _, found = cg.parts(data)
    ids = {p.name: p.id for p in found}
    grouped, doc = run(data, answer({"name": "Bed Frame", "type": "frame", "parts": [ids["model_part0"]], "confidence": 0.9},
                                    {"name": "Mattress", "type": "mattress", "parts": [ids["model_part1"]], "confidence": 0.9},
                                    {"name": "Headboard", "type": "headboard", "parts": [ids["model_part2"]], "confidence": 0.9},
                                    {"name": "Legs", "type": "leg", "parts": [ids[f"Mesh_00{i}"] for i in range(1, 5)], "confidence": 0.9}))
    out, checked, _ = gq.process(grouped, {"separated": True, **doc}, real_height_m=1.1)
    classes = {c["name"]: c["class"] for c in checked["components"]}
    assert classes == {"Bed Frame": "structural", "Mattress": "decorative", "Headboard": "structural", "Legs": "structural"}
    assert all(c.get("triangles") for c in checked["components"])
    assert json.dumps(checked["quality"])  # the report is plain JSON, stored with the model


def test_the_ar_copy_is_made_after_the_response_where_the_document_points():
    from fastapi import BackgroundTasks

    from app.api import vendor

    class Cdn:
        uploads: dict = {}

        def public_url(self, key):
            return f"https://cdn.example/{key}"

        def upload(self, key, data, content_type=None):
            self.uploads[key] = data
            return self.public_url(key)

    pos, idx, nor, uv = sphere(140)
    data = raw_glb([(pos, idx, nor, uv, 0, "Ball")])
    _, doc, job = gq.process(data, components(data, ["Ball Finial"]))
    background, cdn = BackgroundTasks(), Cdn()
    doc = vendor.schedule_ar(background, job, cdn, "owner", "model1", doc)
    assert doc["ar"] == {"url": "https://cdn.example/owner/model1.ar.glb", "status": "processing"} and not cdn.uploads  # nothing yet
    for task in background.tasks:
        task.func(*task.args, **task.kwargs)
    assert cdn.uploads["owner/model1.ar.glb"][:4] == b"glTF"
    assert vendor.schedule_ar(BackgroundTasks(), None, cdn, "owner", "light", {"components": []}) == {"components": []}  # light: no AR link
