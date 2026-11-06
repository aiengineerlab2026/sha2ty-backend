import trimesh

from app.services.part_sizes import measure


def glb() -> bytes:
    """A bed-like model: four 5 × 40 × 5 cm legs in one mesh, a frame and a pillow (2 m wide overall)."""
    legs = trimesh.util.concatenate([trimesh.creation.box((0.05, 0.4, 0.05)).apply_translation((x, 0.2, z)) for x in (-0.95, 0.95) for z in (-1, 1)])
    frame = trimesh.creation.box((2.0, 0.3, 2.1)).apply_translation((0, 0.55, 0))
    pillow = trimesh.creation.box((0.6, 0.15, 0.4)).apply_translation((0, 0.8, -0.8))
    scene = trimesh.Scene()
    for name, mesh in (("legs", legs), ("frame", frame), ("pillow", pillow)):
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    return scene.export(file_type="glb")


def test_parts_are_measured_with_repeated_pieces_once_and_soft_furnishings_left_out():
    parts = [{"id": "legs", "name": "Legs"}, {"id": "frame", "name": "Frame"}, {"id": "pillow", "name": "Pillow"}]
    rows = {r["name"]: r for r in measure(glb(), parts, None)}
    assert set(rows) == {"Legs", "Frame"}
    legs = rows["Legs"]
    assert legs["count"] == 4 and round(legs["width"], 3) == 0.05 and round(legs["height"], 3) == 0.4
    assert rows["Frame"]["count"] == 1 and round(rows["Frame"]["width"], 2) == 2.0


def test_sizes_follow_the_requested_size():
    rows = {r["name"]: r for r in measure(glb(), [{"id": "legs", "name": "Legs"}, {"id": "frame", "name": "Frame"}], {"width": 1.0, "height": 0.875, "depth": 1.05})}
    assert round(rows["Frame"]["width"], 2) == 1.0 and round(rows["Legs"]["height"], 2) == 0.4


def test_fabric_covered_parts_get_a_size_soft_furnishings_do_not():
    parts = [{"id": "legs", "name": "Legs"}, {"id": "frame", "name": "Headboard Upholstery"}]
    rows = measure(glb(), parts, None, [{"part_id": "frame", "material": {"kind": "fabric", "name": "Linen"}}])
    assert {r["name"] for r in rows} == {"Legs", "Headboard Upholstery"}
    assert [r["name"] for r in measure(glb(), [{"id": "legs", "name": "Legs"}, {"id": "frame", "name": "Mattress"}], None)] == ["Legs"]
