"""Realistic materials: regions read from any GLB, the vision answer validated, presets applied without touching
geometry or colours, the cache per file version, and the fallbacks."""

import asyncio
import json

import numpy as np
import pytest

from app.services import realistic_materials as rm
from app.services.glb_size import _views, read_glb, write_glb


def _quad(y: float, size: float = 1.0) -> np.ndarray:
    return np.array([[0, y, 0], [size, y, 0], [size, y, size], [0, y, size]], dtype=np.float32)


def make_glb(materials: list[dict], quads: list[tuple[np.ndarray, int, str | None]]) -> bytes:
    """One mesh per quad (two triangles), each with its own material; names may be missing."""
    doc: dict = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "accessors": [],
                 "bufferViews": [], "buffers": [{"byteLength": 0}], "materials": materials}
    views: list[bytes] = []
    for i, (positions, material, name) in enumerate(quads):
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
        for blob, kind, ctype, count in ((positions.tobytes(), "VEC3", 5126, 4), (indices.tobytes(), "SCALAR", 5123, 6)):
            doc["bufferViews"].append({"buffer": 0, "byteOffset": 0, "byteLength": len(blob)})
            views.append(blob)
            acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": count, "type": kind}
            if kind == "VEC3":
                acc["min"], acc["max"] = positions.min(axis=0).tolist(), positions.max(axis=0).tolist()
            doc["accessors"].append(acc)
        mesh = {"primitives": [{"attributes": {"POSITION": len(doc["accessors"]) - 2}, "indices": len(doc["accessors"]) - 1, "material": material}]}
        if name:
            mesh["name"] = name
        doc["meshes"].append(mesh)
        doc["nodes"].append({"mesh": i})
        doc["scenes"][0]["nodes"].append(i)
    return write_glb(doc, views)


SOFA = [
    {"name": "Material.001", "pbrMetallicRoughness": {"baseColorFactor": [0.4, 0.1, 0.1, 1], "metallicFactor": 0.2, "roughnessFactor": 0.5}},
    {"pbrMetallicRoughness": {"baseColorFactor": [0.3, 0.2, 0.1, 1]}},  # unnamed material on an unnamed mesh
    {"name": "Scanned", "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1], "metallicRoughnessTexture": {"index": 0}}},
    {"name": "Lamp", "extensions": {"KHR_materials_unlit": {}}, "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 0.8, 1]}},
]


def sofa() -> bytes:
    return make_glb(SOFA, [(_quad(0.4, 2.0), 0, "Cube.002"), (_quad(0.0), 1, None), (_quad(0.8, 0.5), 2, "Frame"), (_quad(1.2, 0.2), 3, "Bulb")])


def answer(*regions: dict) -> str:
    return json.dumps({"regions": list(regions)})


def positions(glb: bytes) -> list[bytes]:
    doc, binary = read_glb(glb)
    views = _views(doc, binary)
    return [views[doc["accessors"][p["attributes"]["POSITION"]]["bufferView"]] for m in doc["meshes"] for p in m["primitives"]]


def test_regions_follow_the_materials_whatever_the_names():
    _, _, found = rm.regions(sofa())
    assert [r.id for r in found] == ["R1", "R2", "R3", "R4"]  # by surface share: the 2 m seat first
    seat = found[0]
    assert seat.material == 0 and seat.meshes == ["Cube.002"] and seat.share > 0.7
    assert found[1].material == 1 and found[1].meshes == [] and found[1].name == ""  # unnamed parts are regions too
    assert found[2].pbr_maps and found[3].unlit
    assert seat.band in {"bottom", "middle", "top"} and len(seat.points) > 100


def test_unreadable_or_empty_models_are_unsupported():
    with pytest.raises(rm.Unsupported):
        rm.regions(b"not a glb")
    with pytest.raises(rm.Unsupported):
        rm.regions(make_glb([], []))


def test_the_views_are_two_pictures_for_the_vision_model():
    _, _, found = rm.regions(sofa())
    look, regionmap = rm.render(found)
    assert look[:8] == b"\x89PNG\r\n\x1a\n" and regionmap[:8] == b"\x89PNG\r\n\x1a\n"
    assert "R2" in rm.prompt(found, False) and "Picture 3" in rm.prompt(found, True)


def test_parse_keeps_only_valid_listed_regions():
    ids = {"R1", "R2"}
    profile = rm.parse(answer({"id": "R1", "material": "velvet", "finish": "matte", "confidence": 0.9},
                              {"id": "R1", "material": "wood", "confidence": 0.9},  # a duplicate: the first answer counts
                              {"id": "R9", "material": "metal", "confidence": 0.9}), ids)  # not a region
    assert profile is not None and [(r.id, r.material) for r in profile.regions] == [("R1", "velvet")]
    assert rm.parse("no json", ids) is None
    assert rm.parse(answer({"id": "R1", "material": "unobtainium", "confidence": 0.9}), ids) is None  # outside the schema
    assert rm.parse(answer({"id": "R1", "material": "wood", "confidence": 3}), ids) is None


def test_apply_writes_presets_and_keeps_geometry_and_colours():
    data = sofa()
    doc, views, found = rm.regions(data)
    profile = rm.MaterialProfile.model_validate({"regions": [
        {"id": "R1", "material": "velvet", "finish": "matte", "confidence": 0.9},
        {"id": "R2", "material": "metal", "finish": "gloss", "confidence": 0.4},  # unsure: kept as it was
        {"id": "R3", "material": "wood", "finish": "gloss", "confidence": 0.95},  # has its own roughness map
        {"id": "R4", "material": "glass", "confidence": 0.99},  # unlit: never changed
    ]})
    report = rm.apply(doc, found, profile)
    assert [r["applied"] for r in report] == [True, False, True, False]
    seat, legs, frame, lamp = doc["materials"]
    assert seat["pbrMetallicRoughness"]["roughnessFactor"] == 0.9 and seat["pbrMetallicRoughness"]["metallicFactor"] == 0  # velvet 0.75 + matte
    assert seat["pbrMetallicRoughness"]["baseColorFactor"] == [0.4, 0.1, 0.1, 1]  # the colour is the model's own
    assert seat["extensions"]["KHR_materials_sheen"]["sheenColorFactor"][0] > seat["extensions"]["KHR_materials_sheen"]["sheenColorFactor"][2]
    assert legs == SOFA[1] and lamp == SOFA[3]
    assert "roughnessFactor" not in frame["pbrMetallicRoughness"] and frame["extensions"]["KHR_materials_clearcoat"]["clearcoatFactor"] == 0.15
    assert set(doc["extensionsUsed"]) == {"KHR_materials_sheen", "KHR_materials_clearcoat"}
    out = rm.realistic_glb(doc, views)
    assert positions(out) == positions(data)  # geometry byte for byte
    _, _, again = rm.regions(out)
    assert [r.points.shape for r in again] == [r.points.shape for r in found]


class Store:
    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.uploads: list[str] = []

    async def get(self, key):
        return self.rows.get(key)

    async def put(self, key, result):
        self.rows[key] = result

    def upload(self, key, glb):
        self.uploads.append(key)
        assert glb[:4] == b"glTF"
        return f"https://cdn.example/{key}"


def run(data, ask, store):
    return asyncio.run(rm.make(data, None, ask, store.get, store.put, store.upload))


def test_make_caches_per_file_version_and_asks_once():
    store, calls = Store(), []

    async def ask(content, schema):
        calls.append(content)
        assert content[0]["type"] == "input_text" and len(content) == 3 and "regions" in json.dumps(schema)
        return answer({"id": "R1", "material": "fabric", "confidence": 0.8})

    first = run(sofa(), ask, store)
    assert first["status"] == "ready" and first["glb_url"].startswith("https://cdn.example/realistic/")
    assert run(sofa(), ask, store)["cached"] is True and len(calls) == 1  # the same file: from the cache
    changed = make_glb(SOFA, [(_quad(0.4, 2.5), 0, "Cube.002"), (_quad(0.0), 1, None)])
    assert run(changed, ask, store)["status"] == "ready" and len(calls) == 2  # a changed model is a new version
    assert len(store.uploads) == 2


def test_failures_fall_back_and_are_not_cached():
    store = Store()

    async def broken(content, schema):
        return None

    assert run(sofa(), broken, store)["status"] == "failed" and not store.rows and not store.uploads

    async def unsure(content, schema):
        return answer({"id": "R1", "material": "unknown", "confidence": 0.2})

    result = run(sofa(), unsure, store)
    assert result["status"] == "unchanged" and not store.uploads  # nothing to change: the app keeps the original
    assert run(b"junk", unsure, store)["status"] == "unsupported"


def test_the_route_only_reads_models_stored_on_sha2ty(monkeypatch):
    from fastapi.testclient import TestClient
    from pydantic import SecretStr

    from app.api import designs as designs_api, realistic
    from app.core.config import Settings
    from app.main import app

    settings = Settings(supabase_url="https://sb.example", supabase_service_role_key=SecretStr("srv"))
    monkeypatch.setattr(realistic, "get_settings", lambda: settings)
    app.dependency_overrides[designs_api.current_user] = lambda: "user-1"
    try:
        with TestClient(app) as client:
            refused = client.post("/api/realistic-materials", json={"glb_url": "https://elsewhere.example/chair.glb"})
            assert refused.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_meshes_sharing_one_material_get_their_own_copies():
    """A generated model: seat and legs are separate meshes with one shared (textured) material."""
    shared = [{"name": "meshy_textured", "pbrMetallicRoughness": {"baseColorFactor": [0.6, 0.5, 0.4, 1]}}]
    data = make_glb(shared, [(_quad(0.5, 1.5), 0, "seat"), (_quad(0.0, 0.6), 0, "legs")])
    doc, views, found = rm.regions(data)
    assert [r.meshes for r in found] == [["seat"], ["legs"]] and {r.material for r in found} == {0}
    report = rm.apply(doc, found, rm.MaterialProfile.model_validate({"regions": [
        {"id": "R1", "material": "fabric", "confidence": 0.8}, {"id": "R2", "material": "wood", "confidence": 0.8}]}))
    assert all(r["applied"] for r in report)
    seat_mat, legs_mat = (doc["meshes"][i]["primitives"][0]["material"] for i in (0, 1))
    assert seat_mat != legs_mat and len(doc["materials"]) == 2  # the last part on the shared material keeps it
    assert doc["materials"][seat_mat]["pbrMetallicRoughness"]["roughnessFactor"] == 0.92
    assert doc["materials"][legs_mat]["pbrMetallicRoughness"]["roughnessFactor"] == 0.55
    assert all(doc["materials"][i]["pbrMetallicRoughness"]["baseColorFactor"] == [0.6, 0.5, 0.4, 1] for i in (seat_mat, legs_mat))
    assert positions(rm.realistic_glb(doc, views)) == positions(data)
