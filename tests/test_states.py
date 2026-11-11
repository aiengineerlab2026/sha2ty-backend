"""One wardrobe, two states: the same panels, the open one really open, both saved as one product."""

import base64
import io
import json

import numpy as np
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from app.api import vendor
from app.core.config import Settings
from app.main import app
from app.services import wardrobe
from app.services.model_check import check_glb, check_states
from tests.test_wardrobe import CHILD

OPEN = {"same_wardrobe": True, "columns": [{"sections": [{"shelves": 0, "rail": False}, {"shelves": 0, "rail": False}, {"shelves": 1, "rail": True}]},
                                           {"sections": [{"shelves": 0, "rail": False}, {"shelves": 0, "rail": False}, {"shelves": 4, "rail": False}]}],
        "interior_color": "#d9d2c5", "shelf_color": "#fbf8f2"}


def _spec():
    return wardrobe.normalize(wardrobe.WardrobeSpec.model_validate(CHILD), "child wardrobe")


def _bounds(glb):
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    out = {}
    for node in scene.graph.nodes_geometry:
        transform, geom = scene.graph[node]
        mesh = scene.geometry[geom].copy()
        mesh.apply_transform(transform)
        out[node] = mesh.bounds
    return out


def test_open_state_is_the_same_panels_with_doors_turned_and_drawers_out():
    closed, names = wardrobe.build(_spec())
    opened, open_names = wardrobe.build(_spec(), state="open")
    assert names == open_names  # same parts, same names, same order: an edit to one part means the same panel
    a, b = _bounds(closed), _bounds(opened)
    for fixed in ("wardrobe_body", "back_panel", "plinth", "shelf_1", "rail_1"):
        assert np.allclose(a[fixed], b[fixed], atol=1e-6)  # same size, orientation and origin
    depth = 0.48
    for door in ("door_left", "door_right"):
        assert b[door][1][2] - b[door][0][2] > 0.25 and b[door][1][2] > depth / 2 + 0.25  # turned out, not hidden
    assert b["door_left"][1][0] < a["door_left"][1][0] and b["door_right"][0][0] > a["door_right"][0][0]  # each to its side
    for drawer in ("drawer_1", "drawer_2", "drawer_3", "drawer_4"):
        dz = b[drawer][0][2] - a[drawer][0][2]
        assert dz > 0.15 and np.allclose(a[drawer][:, :2], b[drawer][:, :2], atol=1e-6)  # pulled straight out
    # Handles go with their front: every one still touches a door or drawer, none is left floating in front.
    def touching(h, f):
        return all(h[0][i] <= f[1][i] + 1e-3 and h[1][i] >= f[0][i] - 1e-3 for i in range(3))

    fronts = [n for n in b if n.startswith(("door", "drawer"))]
    for handle in (n for n in b if n.startswith("handle")):
        assert any(touching(b[handle], b[f]) for f in fronts), handle


def test_states_are_validated_as_one_product_and_a_faked_open_state_is_refused():
    closed, _ = wardrobe.build(_spec())
    opened, _ = wardrobe.build(_spec(), state="open")
    assert check_glb(opened, "wardrobe")["ok"]
    assert check_states(closed, opened) == {"ok": True, "errors": [], "warnings": []}

    assert not check_states(closed, closed)["ok"]  # nothing moved: not open

    scene = trimesh.load(io.BytesIO(closed), file_type="glb", force="scene")
    for node in ("door_left", "door_right"):
        scene.delete_geometry(scene.graph[node][1])  # "open" by hiding the doors
    hidden = check_states(closed, scene.export(file_type="glb"))
    assert not hidden["ok"] and "open lacks door_left, door_right" in hidden["errors"][0]

    bigger = wardrobe.normalize(wardrobe.WardrobeSpec.model_validate({**CHILD, "width_m": 1.4}), "child wardrobe")
    other, _ = wardrobe.build(bigger, state="open")
    assert any("wardrobe_body is not where it is" in e for e in check_states(closed, other)["errors"])  # another size


def test_the_open_photo_fills_the_inside_of_both_states():
    spec, notes = wardrobe.with_inside(_spec(), wardrobe.OpenReading.model_validate(OPEN))
    assert not notes
    door_bay = spec.columns[0].sections[2]
    assert door_bay.shelves == 1 and door_bay.rail is True and spec.columns[1].sections[2].shelves == 4
    assert spec.interior_color == "#d9d2c5"
    _, names = wardrobe.build(spec)
    ids = [n["id"] for n in names]
    assert "rail_1" in ids and sum(i.startswith("shelf") for i in ids) == 5  # closed doors hide the inside, not remove it

    other = {**OPEN, "columns": OPEN["columns"][:1]}
    same, notes = wardrobe.with_inside(_spec(), wardrobe.OpenReading.model_validate(other))
    assert notes and same.columns[1].sections[2].shelves == 3  # another shape: the closed reading stays


def _png(color="#ffffff", size=(200, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _client(monkeypatch, tmp_path, answers, uploads):
    async def no_details(*_args, **_kwargs):
        return None

    async def vision_answer(_content, _schema, name, *_args):
        return answers(name)

    class FakeCdn:
        def upload(self, path, data, _type):
            uploads[path] = data
            return f"https://cdn.test/{path}"

    class NoMeshy:
        async def create_image_to_3d_task(self, *_a, **_k):
            raise AssertionError("a wardrobe with two states is not sent to image-to-3D")

    monkeypatch.setattr(vendor, "extract_product_details", no_details)
    monkeypatch.setattr(vendor.vision, "ask_json", vision_answer)
    s = Settings(meshy_api_key="msy-test", gemini_api_key="gem-test", storage_dir=tmp_path)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: NoMeshy(), vendor.get_cdn: lambda: FakeCdn(), vendor.get_settings: lambda: s})
    return TestClient(app)


def test_generate_saves_both_states_of_one_product(tmp_path, monkeypatch):
    uploads = {}
    answers = {"wardrobe_layout": json.dumps(CHILD), "wardrobe_open": json.dumps(OPEN)}
    try:
        with _client(monkeypatch, tmp_path, answers.get, uploads) as client:
            body = client.post("/api/vendor/generate", files={"image": ("closed.png", _png(), "image/png"), "open_image": ("open.png", _png(), "image/png")},
                               data={"category": "wardrobe", "kind": "Child Wardrobe"}).json()
    finally:
        app.dependency_overrides.clear()
    assert body["status"] == "SUCCEEDED", body
    open_state = body["states"]["open"]
    assert open_state["glb_url"].endswith(".open.glb") and body["glb_url"].endswith(".glb") and open_state["glb_url"] != body["glb_url"]
    assert [c["id"] for c in open_state["components"]["components"]] == [c["id"] for c in body["components"]["components"]]
    assert open_state["components"]["validation"]["states"]["ok"]
    files = {k: v for k, v in uploads.items()}
    assert len(files) == 2 and check_states(files[next(k for k in files if not k.endswith(".open.glb"))], files[next(k for k in files if k.endswith(".open.glb"))])["ok"]


def test_different_wardrobes_are_flagged_and_unreadable_open_photos_refused(tmp_path, monkeypatch):
    uploads = {}
    other = {**OPEN, "same_wardrobe": False, "differences": "the open one is blue"}
    answers = {"wardrobe_layout": json.dumps(CHILD), "wardrobe_open": json.dumps(other)}
    try:
        with _client(monkeypatch, tmp_path, answers.get, uploads) as client:
            files = {"image": ("c.png", _png(), "image/png"), "open_image": ("o.png", _png(), "image/png")}
            flagged = client.post("/api/vendor/generate", files=files, data={"category": "wardrobe"}).json()
            answers["wardrobe_open"] = "not json"
            refused = client.post("/api/vendor/generate", files=files, data={"category": "wardrobe", "title": "again"})
            sofa = client.post("/api/vendor/generate", files=files, data={"category": "furniture", "kind": "sofa"})
    finally:
        app.dependency_overrides.clear()
    validation = flagged["states"]["open"]["components"]["validation"]
    assert not validation["ok"] and "the open one is blue" in validation["errors"][0]
    assert refused.status_code == 422 and "open photo could not be read" in refused.json()["detail"]
    assert sofa.status_code == 400


def test_one_photo_of_both_states_is_cut_in_two(tmp_path, monkeypatch):
    img = Image.new("RGB", (400, 200), "#ffffff")
    img.paste(Image.new("RGB", (180, 200), "#a9b99c"), (0, 0))  # closed, left
    img.paste(Image.new("RGB", (180, 200), "#d9d2c5"), (220, 0))  # open, right
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    answers = {"wardrobe_states": json.dumps({"closed": [0, 0, 0.45, 1], "open": [0.55, 0, 1, 1]})}
    try:
        with _client(monkeypatch, tmp_path, answers.get, {}) as client:
            both = client.post("/api/vendor/split-states", files={"image": ("both.png", buf.getvalue(), "image/png")}).json()
            answers["wardrobe_states"] = json.dumps({"closed": [0, 0, 0.45, 1], "open": None})
            one = client.post("/api/vendor/split-states", files={"image": ("both.png", buf.getvalue(), "image/png")})
    finally:
        app.dependency_overrides.clear()
    colours = {}
    for name in ("closed", "open"):
        part = Image.open(io.BytesIO(base64.b64decode(both[name].split(",", 1)[1]))).convert("RGB")
        colours[name] = part.getpixel((part.width // 2, part.height // 2))
    assert abs(colours["closed"][0] - 0xA9) < 6 and abs(colours["open"][0] - 0xD9) < 6
    assert one.status_code == 422
