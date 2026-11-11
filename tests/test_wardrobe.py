"""Child wardrobe from a photo: real, separately editable panels, validated before saving."""

import asyncio
import io
import json

import trimesh
from fastapi.testclient import TestClient

from app.api import vendor
from app.core.config import Settings
from app.main import app
from app.services import wardrobe
from app.services.model_check import check_glb

# The reference photo: two pink doors over two drawers on the left, open shelves over two drawers on the right.
CHILD = {
    "width_m": 1.2, "height_m": 1.8, "depth_m": 0.48, "base": "plinth", "handle": "knob",
    "body_color": "#efe9dd", "front_color": "#e8b7c0", "handle_color": "#f2f2f2",
    "columns": [
        {"width": 0.62, "sections": [{"kind": "drawer", "height": 0.11}, {"kind": "drawer", "height": 0.11}, {"kind": "door", "height": 0.78, "leaves": 2}]},
        {"width": 0.38, "sections": [{"kind": "drawer", "height": 0.11}, {"kind": "drawer", "height": 0.11}, {"kind": "open", "height": 0.78, "shelves": 3}]},
    ],
}


def _built(spec=CHILD, lang="ar"):
    return wardrobe.build(wardrobe.normalize(wardrobe.WardrobeSpec.model_validate(spec), "child wardrobe"), lang)


def _scene(glb):
    return trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")


def test_child_wardrobe_is_built_from_real_named_components():
    glb, names = _built()
    ids = [n["id"] for n in names]
    assert ids[:3] == ["wardrobe_body", "back_panel", "plinth"]
    assert {"door_left", "door_right", "drawer_1", "drawer_2", "drawer_3", "drawer_4", "shelf_1", "shelf_2", "shelf_3"} <= set(ids)
    assert sorted(i for i in ids if i.startswith("handle")) == [f"handle_{i}" for i in range(1, 7)]  # 2 doors + 4 drawers
    assert dict((n["id"], n["name"]) for n in names)["door_left"] == "الباب الأيسر"
    assert "drawer_2" in ids and dict((n["id"], n["name"]) for n in names)["drawer_2"] == "درج 2"

    scene = _scene(glb)
    assert set(scene.graph.nodes_geometry) == set(ids)  # one node + mesh per component, nothing else
    mats = [scene.geometry[scene.graph[n][1]].visual.material for n in ids]
    assert len({id(m) for m in mats}) == len(ids)  # its own material each: recolour one, the rest stay
    color = dict(zip(ids, mats))
    assert list(color["door_left"].baseColorFactor[:3]) == [0xE8, 0xB7, 0xC0]  # the photo's pink fronts
    assert list(color["wardrobe_body"].baseColorFactor[:3]) == [0xEF, 0xE9, 0xDD]
    assert abs(scene.bounds[1][0] - scene.bounds[0][0] - 1.2) < 1e-6 and abs(scene.bounds[1][1] - 1.8) < 1e-6  # real size
    assert scene.bounds[0][1] == 0  # standing on the floor


def test_parts_are_straight_boxes_not_fragments():
    glb, names = _built()
    scene = _scene(glb)
    door = scene.geometry[scene.graph["door_left"][1]]
    # A door is one flat rectangular panel: 8 corners, 12 triangles, a closed volume of exactly w × h × t.
    assert len(door.vertices) == 8 and len(door.faces) == 12 and door.is_watertight
    w, h, t = sorted(door.extents, reverse=True)[1], max(door.extents), min(door.extents)
    assert abs(door.volume - w * h * t) < 1e-9 and abs(t - wardrobe.PANEL) < 1e-6  # float32 in the GLB
    assert abs(door.volume - door.bounding_box.volume) < 1e-9  # no bend, no skew: it fills its own box


def test_check_passes_the_built_wardrobe_and_its_doors_meet_without_overlap():
    glb, _ = _built()
    report = check_glb(glb, "wardrobe")
    assert report == {"ok": True, "errors": [], "warnings": [], "parts": 19}  # with the inferred hanging rail (rail_1)


def test_check_rejects_overlapping_floating_and_degenerate_parts():
    def scene(*parts):
        s = trimesh.Scene()
        for name, mesh in parts:
            s.add_geometry(mesh, geom_name=name, node_name=name)
        return s.export(file_type="glb")

    def box(w, h, d, x=0, y=0, z=0):
        return trimesh.creation.box(extents=(w, h, d), transform=trimesh.transformations.translation_matrix((x, y + h / 2, z)))

    body = ("wardrobe_body", box(1.0, 1.8, 0.5))
    overlapping = check_glb(scene(body, ("door_1", box(0.6, 1.5, 0.02, -0.15, 0.1, 0.26)), ("door_2", box(0.6, 1.5, 0.02, 0.15, 0.1, 0.26))), "wardrobe")
    assert not overlapping["ok"] and "door_1 and door_2 overlap." in overlapping["errors"]

    floating = check_glb(scene(body, ("handle_1", box(0.03, 0.03, 0.03, 0, 1.0, 0.6))), "wardrobe")
    assert "handle_1 floats, touching no other part." in floating["errors"]

    flat = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]], faces=[[0, 1, 2]], process=False)
    assert not check_glb(scene(body, ("shelf_1", flat)), "wardrobe")["ok"]

    tiny = check_glb(scene(("wardrobe_body", box(0.05, 0.05, 0.05))), "wardrobe")
    assert any("not a real piece of furniture" in e for e in tiny["errors"])

    generic = check_glb(scene(("wardrobe_body", box(1.0, 1.8, 0.5)), ("Part 2", box(0.5, 0.5, 0.02, 0, 0.2, 0.26))), "wardrobe")
    assert any("without a meaningful name" in w for w in generic["warnings"])

    assert not check_glb(b"not a glb")["ok"]


def test_layout_is_normalized_to_real_child_sizes():
    spec = wardrobe.normalize(wardrobe.WardrobeSpec.model_validate({**CHILD, "height_m": 2.6, "depth_m": 0.05}), "Child Wardrobe")
    assert spec.height_m == 2.0 and spec.depth_m == 0.3
    assert abs(sum(c.width for c in spec.columns) - 1) < 1e-9
    assert all(abs(sum(s.height for s in c.sections) - 1) < 1e-9 for c in spec.columns)


def test_read_spec_sends_the_type_and_rejects_unreadable_answers():
    prompts = []

    async def ask(prompt, schema):
        prompts.append(prompt)
        assert "columns" in schema["properties"]
        return json.dumps(CHILD)

    spec = asyncio.run(wardrobe.read_spec(ask, "a single child wardrobe", "child wardrobe"))
    assert spec and len(spec.columns) == 2 and "child wardrobe" in prompts[0] and "child's wardrobe" in prompts[0]

    async def junk(_prompt, _schema):
        return '{"columns": []}'

    assert asyncio.run(wardrobe.read_spec(junk, "a wardrobe", None)) is None


def test_generate_builds_a_child_wardrobe_without_meshy(tmp_path, monkeypatch):
    uploads = {}

    async def no_details(*_args, **_kwargs):
        return None

    async def layout(_content, _schema, name, *_args):
        assert name == "wardrobe_layout"
        return json.dumps(CHILD)

    class NoMeshy:
        async def create_image_to_3d_task(self, *_a, **_k):
            raise AssertionError("a wardrobe is not sent to image-to-3D")

    class FakeCdn:
        def upload(self, path, data, _type):
            uploads[path] = data
            return f"https://cdn.test/{path}"

    async def no_try(*_args):
        raise AssertionError("no generation try is used")

    monkeypatch.setattr(vendor, "extract_product_details", no_details)
    monkeypatch.setattr(vendor.vision, "ask_json", layout)
    monkeypatch.setattr(vendor, "take_try", no_try)
    s = Settings(meshy_api_key="msy-test", gemini_api_key="gem-test", storage_dir=tmp_path)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: NoMeshy(), vendor.get_cdn: lambda: FakeCdn(), vendor.get_settings: lambda: s})
    try:
        with TestClient(app) as client:
            png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
            body = client.post("/api/vendor/generate", files={"image": ("w.png", png, "image/png")}, data={"category": "wardrobe", "kind": "Child Wardrobe"}).json()
    finally:
        app.dependency_overrides.clear()
    assert body["status"] == "SUCCEEDED" and body["glb_url"].endswith(".glb")
    doc = body["components"]
    assert doc["method"] == "procedural" and doc["separated"] and doc["validation"]["ok"]
    assert [c["id"] for c in doc["components"]][:4] == ["wardrobe_body", "back_panel", "plinth", "door_left"]
    assert all(not c["name"].startswith(("Part", "جزء")) for c in doc["components"])
    assert check_glb(next(iter(uploads.values())), "wardrobe")["ok"]
