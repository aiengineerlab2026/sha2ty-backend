import trimesh

from app.schemas.product_details import ProductDetails
from app.services.generation_prompt import MESHY_TEXTURE_PROMPT_MAX, furniture_family, meshy_texture_prompt
from app.services.glb_parts import inspect_glb, segment_glb
from tests.test_vision_service import SOFA


def test_separate_meshes_are_listed_and_a_single_mesh_is_not_split():
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), geom_name="seat_cushion", node_name="Seat cushion")
    scene.add_geometry(trimesh.creation.cylinder(radius=0.04, height=0.2), geom_name="leg", node_name="Leg")
    info = inspect_glb(scene.export(file_type="glb"), "sofa")
    assert info["separated"] is True
    assert {c["name"] for c in info["components"]} >= {"Seat cushion", "Leg"}
    assert all(c["editable"] for c in info["components"])

    one_info = inspect_glb(trimesh.creation.box().export(file_type="glb"), "sofa")
    assert one_info["separated"] is False
    assert len(one_info["components"]) == 1
    assert one_info["limits"]


def test_sofa_parts_go_to_auto_split_not_to_the_texture_prompt():
    """texture_prompt only guides the texture (docs.meshy.ai): the parts are Auto Split's prompt, which cuts them."""
    from app.services.generation_prompt import split_prompt

    details = ProductDetails.model_validate(SOFA)
    prompt = meshy_texture_prompt("furniture", "Oslo sofa", details, "corner sofa")
    assert "Separate named meshes" not in prompt and "corner sofa" in prompt
    assert "seat cushions" in split_prompt("furniture", "corner sofa")
    assert len(prompt) <= MESHY_TEXTURE_PROMPT_MAX


# ---- splitting one generated mesh ----

import numpy as np
from PIL import Image


def _one_mesh_glb(*parts: trimesh.Trimesh, textured: bool = False) -> bytes:
    """Parts concatenated into ONE mesh (as Meshy returns them), optionally with a texture and UVs."""
    mesh = trimesh.util.concatenate(parts)
    if textured:
        uv = np.random.default_rng(0).random((len(mesh.vertices), 2))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.new("RGB", (8, 8), (40, 40, 40)))
    return mesh.export(file_type="glb")


def _box(w, h, d, x=0.0, y=0.0, z=0.0):
    return trimesh.creation.box(extents=(w, h, d), transform=trimesh.transformations.translation_matrix((x, y + h / 2, z)))


def _table(textured=False):
    # 1.30 × 0.75 m table, 4 cm top on four 5 cm legs (separate islands, like many generated tables).
    top = _box(1.3, 0.04, 0.75, y=0.71)
    legs = [_box(0.05, 0.71, 0.05, x=sx * 0.6, z=sz * 0.33) for sx in (-1, 1) for sz in (-1, 1)]
    return _one_mesh_glb(top, *legs, textured=textured)


def test_a_table_with_separate_legs_becomes_a_tabletop_and_four_legs():
    glb, doc = segment_glb(_table(), "table")
    names = [c["name"] for c in doc["components"]]
    assert doc["separated"] and doc["segmented"]
    assert "Tabletop" in names and sum(n.startswith("Leg") for n in names) == 4
    assert inspect_glb(glb, "table")["separated"]  # the exported file really has the meshes


def test_legs_fused_to_a_sofa_body_are_cut_from_the_bottom_band():
    # Legs sharing a corner vertex with the body: one connected surface, split only by the leg band.
    body = _box(2.0, 0.7, 0.9, y=0.1)
    legs = [_box(0.06, 0.1, 0.06, x=sx * (1.0 - 0.03), z=sz * (0.45 - 0.03)) for sx in (-1, 1) for sz in (-1, 1)]
    glb, doc = segment_glb(_one_mesh_glb(body, *legs), "sofa")
    names = [c["name"] for c in doc["components"]]
    assert "Legs" in names or sum(n.startswith("Leg") for n in names) == 4
    assert "Frame" in names


def test_a_solid_block_is_not_split_and_says_so():
    # Nothing in the geometry separates cushions from the frame: stays one mesh (inspect_glb says so).
    assert segment_glb(_one_mesh_glb(_box(2.0, 0.8, 0.9)), "sofa") is None
    info = inspect_glb(_one_mesh_glb(_box(2.0, 0.8, 0.9)), "sofa")
    assert info["separated"] is False and info["limits"]


def test_split_parts_keep_the_generated_texture_and_uvs():
    glb, _ = segment_glb(_table(textured=True), "table")
    scene = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb", force="scene")
    assert len(scene.geometry) == 5
    for part in scene.geometry.values():
        assert isinstance(part.visual, trimesh.visual.TextureVisuals)
        assert part.visual.uv is not None and len(part.visual.uv) == len(part.vertices)


def test_a_file_that_already_has_parts_is_left_alone():
    scene = trimesh.Scene()
    scene.add_geometry(_box(1, 1, 1), geom_name="body", node_name="Body")
    scene.add_geometry(_box(0.1, 0.1, 0.1, x=2), geom_name="handle", node_name="Handle")
    assert segment_glb(scene.export(file_type="glb"), "wardrobe") is None


def test_parts_are_named_from_the_selected_type_not_the_photo():
    # The vendor's category / typed furniture type decides; what the photo reading says does not.
    assert furniture_family("furniture", "corner sofa") == "sofa"
    assert furniture_family("table") == "table"
    assert furniture_family("wardrobe") == "wardrobe"
    assert furniture_family("furniture", "") == "furniture"


def test_the_generate_step_saves_the_split_model_and_keeps_the_original():
    from app.api.vendor import modular

    one = _table()
    glb, source, doc = modular(one, "table", "")
    assert source == one  # the untouched generated GLB, kept as the model's original
    assert doc["separated"] and doc["segmented"] and inspect_glb(glb, "table")["separated"]
    # A solid block cannot be split: saved as generated, reported as one mesh (never claimed modular).
    block = _one_mesh_glb(_box(2.0, 0.8, 0.9))
    glb, source, doc = modular(block, "furniture", "corner sofa")
    assert glb == block and source is None and doc["separated"] is False


def _chair(textured=False):
    # 0.5 m chair, front +Z: 4 legs to 0.45 m, an 8 cm seat cushion, a backrest at the back up to 1 m.
    legs = [_box(0.04, 0.45, 0.04, x=sx * 0.23, z=sz * 0.23) for sx in (-1, 1) for sz in (-1, 1)]
    seat = _box(0.5, 0.08, 0.5, y=0.45)
    back = _box(0.5, 0.47, 0.05, y=0.53, z=-0.225)
    return _one_mesh_glb(*(p.subdivide().subdivide() for p in (*legs, seat, back)), textured=textured)  # dense, like generated meshes


def test_a_chair_from_one_mesh_gets_backrest_seat_and_front_and_rear_legs():
    glb, doc = segment_glb(_chair(textured=True), "chair")
    scene = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb", force="scene")
    assert {"backrest", "seat_cushion", "front_legs", "rear_legs"} <= set(scene.geometry)
    assert doc["separated"] and doc["method"] == "geometry"
    for part in scene.geometry.values():  # independent meshes, each keeping its UVs
        assert part.visual.uv is not None and len(part.visual.uv) == len(part.vertices)
    assert sum(len(p.faces) for p in scene.geometry.values()) == 6 * 12 * 16  # every face kept, none duplicated
    # The back's centre is behind the seat's; front legs are in front.
    centre = {k: g.bounds.mean(axis=0) for k, g in scene.geometry.items()}
    assert centre["backrest"][2] < centre["seat_cushion"][2] < centre["front_legs"][2]
    assert centre["rear_legs"][2] < 0 < centre["front_legs"][2]


def test_meshy_split_parts_are_named_from_where_they_sit():
    import anyio
    from app.services.part_names import name_components

    scene = trimesh.Scene()
    for i, part in enumerate([_box(0.5, 0.08, 0.5, y=0.45), _box(0.5, 0.47, 0.05, y=0.53, z=-0.225), _box(0.04, 0.45, 0.04, x=0.23, z=0.23), _box(0.04, 0.45, 0.04, x=0.23, z=-0.23)]):
        scene.add_geometry(part, geom_name=f"model_part{i}", node_name=f"model_part{i}")  # as Meshy Auto Split names them
    doc = anyio.run(name_components, inspect_glb(scene.export(file_type="glb"), "chair"), "chair")
    assert [c["name"] for c in doc["components"]] == ["Seat Cushion", "Backrest", "Front Leg", "Rear Leg"]


def test_a_model_that_cannot_be_split_says_how_it_was_left():
    from app.api.vendor import modular

    _, _, doc = modular(_one_mesh_glb(_box(0.6, 0.9, 0.6)), "furniture", "armchair")
    assert doc["separated"] is False and doc["method"] == "none" and "Not split" in doc["limits"]
