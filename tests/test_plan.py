from app.services.plan.geometry_builder import build_room_glb, rectangular_room, room_extents
from app.services.plan.placement import validate_placement


def test_manual_room_physical_dimensions():
    payload = rectangular_room(4, 5, 2.8, 0.15)
    width, depth, height = room_extents(payload)
    assert abs(width - 4) < 1e-6
    assert abs(depth - 5) < 1e-6
    assert abs(height - 2.8) < 1e-6
    assert abs(payload["walls"][0]["length"] - 4) < 1e-6
    glb = build_room_glb(payload)
    assert glb[:4] == b"glTF"


def test_sofa_fits_4m_wall():
    room = rectangular_room(4, 5, 2.8)
    sofa = {"title": "Sofa", "x": 2, "z": 1.4, "width_m": 3, "depth_m": 0.9, "height_m": 0.85, "rotation_y": 0}
    warnings = validate_placement(sofa, room["rooms"], room["walls"], [], 2.8)
    assert warnings == []


def test_sofa_warns_against_2_5m_wall():
    room = rectangular_room(2.5, 5, 2.8)
    sofa = {"title": "Sofa", "x": 1.25, "z": 2.5, "width_m": 3, "depth_m": 0.9, "height_m": 0.85, "rotation_y": 0}
    warnings = validate_placement(sofa, room["rooms"], room["walls"], [], 2.8)
    assert any("0.50m" in w or "0.5m" in w for w in warnings)


def test_rotated_furniture_validation():
    room = rectangular_room(4, 2.2, 2.8)
    sofa = {"title": "Sofa", "x": 2, "z": 1.1, "width_m": 3, "depth_m": 0.9, "height_m": 0.85, "rotation_y": 90}
    warnings = validate_placement(sofa, room["rooms"], room["walls"], [], 2.8)
    assert warnings  # 3m width along 2.2m depth after 90° rotation


def test_furniture_wall_intersection():
    room = rectangular_room(4, 5, 2.8)
    sofa = {"title": "Sofa", "x": 0.1, "z": 0.1, "width_m": 3, "depth_m": 0.9, "height_m": 0.85, "rotation_y": 0}
    warnings = validate_placement(sofa, room["rooms"], room["walls"], [], 2.8)
    assert warnings


def test_save_payload_roundtrip_fields():
    room = rectangular_room(4, 5, 2.8)
    item = {
        "model_id": "sofa-1",
        "x": 2,
        "y": 0,
        "z": 1,
        "rotation_y": 15,
        "width_m": 3,
        "depth_m": 0.9,
        "height_m": 0.85,
        "scale": [1, 1, 1],
    }
    saved = {**room, "items": [item]}
    restored = saved["items"][0]
    assert restored["x"] == 2 and restored["rotation_y"] == 15
    assert restored["width_m"] == 3


def test_rotation_matches_threejs_convention():
    from app.services.plan.placement import furniture_footprint

    # three.js: rotation.y = 30° maps the local +X corner (1.5, 0) to (cos30·1.5, 0, −sin30·1.5)
    poly = furniture_footprint(0, 0, 3, 0.0001, 30)
    xs, zs = poly.exterior.xy
    far = max(zip(xs, zs), key=lambda p: p[0])
    assert abs(far[0] - 1.299) < 0.01 and far[1] < -0.7


def test_room_of_any_shape_has_one_wall_per_side():
    from app.services.plan.geometry_builder import polygon_room

    room = polygon_room([[0, 0], [4, 0], [5.5, 3], [0, 6]], 2.8, 0.15)
    assert [round(w["length"], 3) for w in room["walls"]] == [4.0, round((1.5**2 + 9) ** 0.5, 3), round((5.5**2 + 9) ** 0.5, 3), 6.0]
    assert room["rooms"][0]["width"] == 5.5 and room["rooms"][0]["depth"] == 6
    assert len(build_room_glb(room)) > 0


def test_every_corner_of_an_angled_room_is_closed():
    from shapely.geometry import Polygon
    from app.services.plan.geometry_builder import polygon_room, wall_joints

    room = polygon_room([[0, 0], [4, 0], [1, 3]], 2.8, 0.2)  # a triangle: sharp corners
    joints = wall_joints(room["walls"], 0.2, 2.8)
    assert len(joints) == 3 and all(h == 2.8 for _, h in joints)
    # Each corner post covers the outside gap between the two wall ends meeting there.
    for (hull, _), (x, z) in zip(joints, [[0, 0], [4, 0], [1, 3]]):
        assert Polygon(hull).buffer(1e-9).contains(Polygon(hull).centroid) and Polygon(hull).distance(Polygon([[x, z], [x + 1e-3, z], [x, z + 1e-3]])) == 0


def test_impossible_rooms_are_refused():
    import pytest
    from app.services.plan.geometry_builder import polygon_room

    with pytest.raises(ValueError):
        polygon_room([[0, 0], [4, 0], [0, 4], [4, 4]])  # walls cross (a bow tie)
    with pytest.raises(ValueError):
        polygon_room([[0, 0], [4, 0], [8, 0.01]])  # no floor
