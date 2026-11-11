"""Colours and decoration survive from the reference photo to the saved model, and a model that lost them is flagged."""

import io
import json

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from app.services import wardrobe
from app.services.model_check import appearance, check_glb, with_appearance

SAGE, WHITE, GOLD, SHADOW = "#a9b99c", "#f2ede3", "#c9a54c", "#7d8873"

# The reference: sage arched doors over warm-white drawers, open shelves with a darker inside, gold knobs.
ARCHED = {
    "width_m": 1.2, "height_m": 1.8, "depth_m": 0.5, "base": "plinth", "handle": "knob", "handle_finish": "metal",
    "body_color": WHITE, "front_color": WHITE, "handle_color": GOLD, "interior_color": SHADOW, "shelf_color": WHITE,
    "columns": [
        {"width": 0.62, "sections": [{"kind": "drawer", "height": 0.12}, {"kind": "drawer", "height": 0.12},
                                     {"kind": "door", "height": 0.76, "leaves": 2, "front_color": SAGE, "decor": "arch", "decor_color": WHITE}]},
        {"width": 0.38, "sections": [{"kind": "drawer", "height": 0.24}, {"kind": "open", "height": 0.76, "shelves": 3}]},
    ],
}


def _photo() -> bytes:
    """A drawn front view of ARCHED on a light wall: what the vendor's photo shows."""
    img = Image.new("RGB", (300, 450), "#e9e6e1")
    d = ImageDraw.Draw(img)
    d.rectangle((30, 30, 270, 420), fill=WHITE)  # carcass
    d.rectangle((36, 36, 178, 300), fill=SAGE)  # the two sage doors
    for x in (40, 110):
        d.arc((x + 6, 60, x + 62, 116), 180, 360, fill=WHITE, width=3)
    d.rectangle((184, 36, 264, 300), fill=SHADOW)  # the open bay's darker inside
    for y in (100, 170, 240):
        d.rectangle((184, y, 264, y + 5), fill=WHITE)
    for x, y in ((100, 200), (116, 200), (107, 340), (107, 390), (224, 360)):
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=GOLD)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


BBOX = [0.1, 0.067, 0.9, 0.933]


def _close(a: str, b: str, tol: int = 8) -> bool:
    return all(abs(int(a[i:i + 2], 16) - int(b[i:i + 2], 16)) <= tol for i in (1, 3, 5))


def _build(spec=ARCHED):
    return wardrobe.build(wardrobe.normalize(wardrobe.WardrobeSpec.model_validate(spec), "child wardrobe"))


def _materials(glb):
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    return scene, {n: scene.geometry[scene.graph[n][1]].visual.material for n in scene.graph.nodes_geometry}


def _hex(mat):
    return "#%02x%02x%02x" % tuple(int(v) for v in mat.baseColorFactor[:3])


def test_every_component_keeps_its_own_photo_colour():
    glb, _ = _build()
    _, mats = _materials(glb)
    assert _hex(mats["door_left"]) == _hex(mats["door_right"]) == SAGE  # the section's own front colour
    assert _hex(mats["drawer_1"]) == WHITE and _hex(mats["wardrobe_body"]) == WHITE
    assert _hex(mats["back_panel"]) == SHADOW  # the darker inside seen through the shelves
    assert _hex(mats["shelf_1"]) == WHITE
    assert _hex(mats["handle_1"]) == GOLD and mats["handle_1"].metallicFactor > 0.8  # gold reads as metal
    assert check_glb(glb, "wardrobe")["ok"]


def test_arched_decorations_sit_on_their_doors():
    glb, names = _build()
    scene, _ = _materials(glb)
    decor = [n["id"] for n in names if n["id"].startswith("decor")]
    assert decor == ["decor_1", "decor_2"] and dict((n["id"], n["name"]) for n in names)["decor_1"] == "زخرفة 1"
    for d, door in zip(decor, ("door_left", "door_right")):
        (dx0, dy0, dz0), (dx1, dy1, dz1) = scene.geometry[scene.graph[d][1]].bounds
        (x0, y0, z0), (x1, y1, z1) = scene.geometry[scene.graph[door][1]].bounds
        assert x0 < dx0 < dx1 < x1 and y0 < dy0 < dy1 < y1  # inside its door's face
        assert abs(dz0 - z1) < 1e-6 and dz1 > z1  # on the face, raised
        top = scene.geometry[scene.graph[d][1]].vertices
        # An arch: the outline's highest points are in the middle, its sides lower (not a rectangle).
        assert top[np.argmax(top[:, 1]), 0] - (x0 + x1) / 2 < 0.02


def test_inlay_is_a_filled_closed_arch():
    spec = json.loads(json.dumps(ARCHED))
    spec["columns"][0]["sections"][2]["decor"] = "arch_inlay"
    glb, _ = _build(spec)
    scene, _ = _materials(glb)
    inlay = scene.geometry[scene.graph["decor_1"][1]]
    assert inlay.is_watertight and inlay.volume > 0


def test_a_reading_without_colours_is_not_a_layout():
    no_colours = {k: v for k, v in ARCHED.items() if not k.endswith("_color")}
    assert wardrobe.spec_from_json(json.dumps(no_colours)) is None  # never a silent white wardrobe


def test_the_built_wardrobe_matches_its_photo():
    glb, _ = _build()
    look = appearance(glb, _photo(), BBOX)
    assert look["ok"], look["issues"]
    assert any(_close(p["hex"], SAGE) and p["share"] > 0.2 for p in look["model"])  # sage is there
    assert any(_close(p["hex"], SHADOW) for p in look["model"]) and look["model"][0]["share"] < 0.5  # the inside, not all of it


def test_a_white_model_of_a_sage_wardrobe_is_flagged():
    white = json.loads(json.dumps(ARCHED))
    white["columns"][0]["sections"][2]["front_color"] = WHITE
    white["interior_color"] = WHITE
    glb, _ = _build(white)
    report = with_appearance(check_glb(glb, "wardrobe"), glb, _photo(), BBOX)
    assert not report["ok"]
    assert any("missing from the model" in e for e in report["errors"])


def test_texture_colours_are_read_from_the_uvs():
    """A Meshy-style model: one white material, the colour in its texture. Sampled at the UVs, it matches."""
    mesh = trimesh.creation.box(extents=(1.2, 1.8, 0.5))
    mesh.apply_translation((0, 0.9, 0))
    tex = Image.new("RGB", (64, 64), SAGE)
    uv = np.column_stack([(mesh.vertices[:, 0] + 0.6) / 1.2, mesh.vertices[:, 1] / 1.8])
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex, baseColorFactor=[255, 255, 255, 255]))
    sage_photo = io.BytesIO()
    Image.new("RGB", (100, 150), SAGE).save(sage_photo, format="PNG")
    look = appearance(mesh.export(file_type="glb"), sage_photo.getvalue())
    assert look["ok"] and _close(look["model"][0]["hex"], SAGE)

    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorFactor=[255, 255, 255, 255]))
    lost = appearance(mesh.export(file_type="glb"), sage_photo.getvalue())  # the texture dropped: a white model
    assert not lost["ok"] and any("almost entirely white" in i for i in lost["issues"])
