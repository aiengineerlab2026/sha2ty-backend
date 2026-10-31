"""Build dimensionally accurate room meshes. 1 unit = 1 meter. Y-up."""

from __future__ import annotations

import io
from typing import Any
from uuid import uuid4

import numpy as np
import trimesh

DEFAULT_HEIGHT = 2.7
DEFAULT_THICKNESS = 0.15


def rectangular_room(
    width: float,
    depth: float,
    height: float = DEFAULT_HEIGHT,
    wall_thickness: float = DEFAULT_THICKNESS,
    name: str = "Room",
) -> dict[str, Any]:
    if width <= 0 or depth <= 0 or height <= 0:
        raise ValueError("Width, depth and height must be greater than zero.")
    if wall_thickness <= 0 or wall_thickness * 2 >= min(width, depth):
        raise ValueError("Wall thickness is too large for this room.")

    room_id = str(uuid4())
    polygon = [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]]
    walls = _walls_from_polygon(polygon, height, wall_thickness)
    return {
        "rooms": [
            {
                "id": room_id,
                "name": name,
                "polygon": polygon,
                "width": width,
                "depth": depth,
                "height": height,
            }
        ],
        "walls": walls,
        "openings": [],
        "warnings": [],
        "scale_confidence": 1.0,
        "source": "dimensions",
        "wall_thickness": wall_thickness,
        "height": height,
    }


DOOR_HEIGHT = 2.1
SILL_HEIGHT = 0.9


def geometry_from_plan(
    rooms_px: list[dict[str, Any]],
    zones_px: list[dict[str, Any]],
    walls_px: list[dict[str, Any]],
    openings_px: list[dict[str, Any]],
    footprint_px: list[list[float]],
    meters_per_pixel: float,
    height: float = DEFAULT_HEIGHT,
    default_thickness: float = DEFAULT_THICKNESS,
) -> dict[str, Any]:
    """Pixel-space plan → metres. Walls are the detected wall bodies (real thickness), so door and
    window gaps already exist; doors get a lintel, windows a sill and a lintel. Floors are the room
    polygons plus a strip under every doorway. Origin = footprint's top-left corner."""
    base = {"unit": "meters", "source": "image", "height": height, "wall_thickness": default_thickness}
    if not meters_per_pixel or not footprint_px:
        return {**base, "rooms": [], "zones": [], "walls": [], "openings": [], "doors": [], "windows": [], "floors": [], "exterior": []}
    ox = min(p[0] for p in footprint_px)
    oy = min(p[1] for p in footprint_px)
    s = meters_per_pixel

    def m(p):
        return [round((p[0] - ox) * s, 4), round((p[1] - oy) * s, 4)]

    zones_by_room: dict[str, list[dict]] = {}
    zones = []
    for z in zones_px:
        item = {
            "id": z["id"],
            "room_id": z["room_id"],
            "name": z["name"],
            "kind": z["kind"],
            "polygon": [m(p) for p in z["polygon"]],
            "label_point": m(z["label_point"]),
            "boundary": z["boundary"],
            "dimensions": _dims(z, s),
            "printed": z.get("dims_text"),
            "confidence": z.get("confidence"),
        }
        zones.append(item)
        zones_by_room.setdefault(z["room_id"], []).append(item)

    rooms = []
    for r in rooms_px:
        poly = [m(p) for p in r["polygon"]]
        if len(poly) < 3:
            continue
        xs, zs = [p[0] for p in poly], [p[1] for p in poly]
        own = zones_by_room.get(r["id"], [])
        dims = own[0]["dimensions"] if len(own) == 1 else {"width": max(xs) - min(xs), "length": max(zs) - min(zs)}
        rooms.append(
            {
                "id": r["id"],
                "name": r["name"],
                "kind": r["kind"],
                "polygon": poly,
                "holes": [[m(p) for p in ring] for ring in r.get("holes") or []],
                "width": max(xs) - min(xs),
                "depth": max(zs) - min(zs),
                "height": height,
                "dimensions": dims,
                "zones": [z["id"] for z in own],
                "confidence": r.get("confidence"),
            }
        )

    walls = []
    for w in walls_px:
        a, b = m(w["start"]), m(w["end"])
        walls.append(_wall(w["id"], a, b, float(w.get("thickness") or 0) * s or default_thickness, height))

    doors, windows, openings, floors = [], [], [], []
    for o in openings_px:
        a, b = m(o["start"]), m(o["end"])
        t = float(o.get("thickness") or 0) * s or default_thickness
        item = {**o, "start": a, "end": b, "thickness": t, "length": float(np.hypot(b[0] - a[0], b[1] - a[1]))}
        openings.append(item)
        if o["kind"] == "door":
            doors.append(item)
            if height > DOOR_HEIGHT:
                walls.append(_wall(f"{o['id']}-lintel", a, b, t, height - DOOR_HEIGHT, DOOR_HEIGHT))
        elif o["kind"] == "window":
            windows.append(item)
            walls.append(_wall(f"{o['id']}-sill", a, b, t, SILL_HEIGHT))
            if height > DOOR_HEIGHT:
                walls.append(_wall(f"{o['id']}-lintel", a, b, t, height - DOOR_HEIGHT, DOOR_HEIGHT))
        if o["kind"] in ("door", "opening"):
            floors.append(_strip(a, b, t))

    return {
        **base,
        "rooms": rooms,
        "zones": zones,
        "walls": walls,
        "openings": openings,
        "doors": doors,
        "windows": windows,
        "floors": floors,
        "exterior": [m(p) for p in footprint_px],
    }


def _dims(z: dict, s: float) -> dict[str, Any]:
    measured = z.get("measure_px") or {}
    w, d = measured.get("width", 0) * s, measured.get("depth", 0) * s
    out: dict[str, Any] = {"width": round(w, 3), "length": round(d, 3), "measured": [round(w, 3), round(d, 3)]}
    if z.get("dims_m"):
        a, b = z["dims_m"]
        # Report printed values in the same orientation as the measured ones.
        if abs(a - w) + abs(b - d) > abs(b - w) + abs(a - d):
            a, b = b, a
        out["printed"] = [round(a, 4), round(b, 4)]
    return out


def _wall(wid: str, a: list[float], b: list[float], thickness: float, height: float, elevation: float = 0.0) -> dict[str, Any]:
    return {
        "id": wid,
        "start": a,
        "end": b,
        "height": height,
        "elevation": elevation,
        "thickness": thickness,
        "length": float(np.hypot(b[0] - a[0], b[1] - a[1])),
    }


def _strip(a, b, t: float) -> list[list[float]]:
    dx, dz = b[0] - a[0], b[1] - a[1]
    n = max(float(np.hypot(dx, dz)), 1e-9)
    px, pz = -dz / n * t / 2, dx / n * t / 2
    return [[a[0] + px, a[1] + pz], [b[0] + px, b[1] + pz], [b[0] - px, b[1] - pz], [a[0] - px, a[1] - pz]]


# Room type → (floor colour, wall paint). Keep in sync with ROOM_FINISHES in frontend lib/plan/roomMaterials.ts.
ROOM_FINISHES: dict[str, tuple[str, str]] = {
    "bedroom": ("#d9c6a5", "#ece4d6"),
    "living": ("#cdb48e", "#efe8dc"),
    "dining": ("#e2d3b8", "#f0eadf"),
    "kitchen": ("#e8dcc4", "#f4f2ed"),
    "bathroom": ("#efe6d4", "#cfe2e8"),
    "laundry": ("#ebe1cd", "#eef0f0"),
    "patio": ("#c8b593", "#d9d4cb"),
    "closet": ("#d4bf9c", "#ebe5da"),
    "hall": ("#dccaa9", "#eee8de"),
    "garage": ("#bfae90", "#d6d6d2"),
    "room": ("#d8c7a8", "#efe9df"),
}
FACADE = "#d9d4cb"
WALL_EDGE = "#e2dccf"


def build_room_glb(payload: dict[str, Any]) -> bytes:
    """Floors and walls only — no labels, markers or text. Floors are coloured by room type
    (per open-plan zone when available); each wall face takes the paint of the room it faces."""
    from shapely.geometry import Point, Polygon

    scene = trimesh.Scene()
    height = float(payload.get("height") or DEFAULT_HEIGHT)
    thickness = float(payload.get("wall_thickness") or DEFAULT_THICKNESS)
    areas = []
    for area in payload.get("zones") or payload.get("rooms") or []:
        poly = area.get("polygon") or []
        if len(poly) >= 3:
            shape = Polygon(poly)
            areas.append((area, shape if shape.is_valid else shape.buffer(0)))

    for area, _ in areas:
        floor = _floor_mesh(area["polygon"])
        if floor is not None:
            # Vendor floor paint wins over the room finish (textures export as their average colour).
            paint = (payload.get("floor_paints") or {}).get(area["id"]) or {}
            _paint(floor, paint.get("color") or _finish(area.get("kind"))[0])
            scene.add_geometry(floor, node_name=f"floor-{area['id']}")
    for i, strip in enumerate(payload.get("floors") or []):
        floor = _floor_mesh(strip)
        if floor is not None:
            _paint(floor, ROOM_FINISHES["hall"][0])
            scene.add_geometry(floor, node_name=f"floor-door-{i}")

    def kind_at(x: float, z: float) -> str | None:
        p = Point(x, z)
        return next((a.get("kind") or "room" for a, shape in areas if shape.contains(p)), None)

    for wall in payload.get("walls") or []:
        t = float(wall.get("thickness") or thickness)
        mesh = _wall_mesh(wall["start"], wall["end"], float(wall.get("height") or height), t, float(wall.get("elevation") or 0))
        (x0, z0), (x1, z1) = wall["start"], wall["end"]
        length = max(float(np.hypot(x1 - x0, z1 - z0)), 1e-9)
        nx, nz = -(z1 - z0) / length, (x1 - x0) / length  # left of start→end, seen from above
        mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
        reach = t / 2 + 0.08
        left = kind_at(mx + nx * reach, mz + nz * reach)
        right = kind_at(mx - nx * reach, mz - nz * reach)
        # Vendor paint on a face wins over the room finish (textures export as their average colour).
        paints = payload.get("wall_paints") or {}
        left_color = (paints.get(f"{wall['id']}:L") or {}).get("color") or (_finish(left)[1] if left else FACADE)
        right_color = (paints.get(f"{wall['id']}:R") or {}).get("color") or (_finish(right)[1] if right else FACADE)
        _paint_wall(mesh, (nx, nz), left_color, right_color)
        scene.add_geometry(mesh, node_name=f"wall-{wall['id']}")
    if len(scene.geometry) == 0:
        raise ValueError("No room geometry to export.")
    buffer = io.BytesIO()
    scene.export(file_obj=buffer, file_type="glb")
    return buffer.getvalue()


def _finish(kind: str | None) -> tuple[str, str]:
    return ROOM_FINISHES.get(kind or "", ROOM_FINISHES["room"])


def _rgba(hex_color: str) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [255]


# Vertex colours, set directly: glTF stores colours per vertex, and trimesh's face→vertex
# conversion needs scipy.
def _paint(mesh: trimesh.Trimesh, color: str) -> None:
    mesh.visual.vertex_colors = np.tile(_rgba(color), (len(mesh.vertices), 1))


def _paint_wall(mesh: trimesh.Trimesh, normal: tuple[float, float], left: str, right: str) -> None:
    """Colour the two long faces by the room on each side; ends and top get a neutral edge colour."""
    mesh.unmerge_vertices()  # a box shares corner vertices; each face needs its own to differ in colour
    face = np.tile(_rgba(WALL_EDGE), (len(mesh.faces), 1))
    side = mesh.face_normals[:, 0] * normal[0] + mesh.face_normals[:, 2] * normal[1]
    face[side > 0.5] = _rgba(left)
    face[side < -0.5] = _rgba(right)
    vertex = np.zeros((len(mesh.vertices), 4), np.uint8)
    vertex[mesh.faces.ravel()] = np.repeat(face, 3, axis=0)
    mesh.visual.vertex_colors = vertex


def room_extents(payload: dict[str, Any]) -> tuple[float, float, float]:
    rooms = payload.get("rooms") or []
    if not rooms:
        return 0.0, 0.0, float(payload.get("height") or 0)
    xs: list[float] = []
    zs: list[float] = []
    height = 0.0
    for room in rooms:
        for x, z in room.get("polygon") or []:
            xs.append(float(x))
            zs.append(float(z))
        height = max(height, float(room.get("height") or 0))
    return (max(xs) - min(xs) if xs else 0.0, max(zs) - min(zs) if zs else 0.0, height)


def _walls_from_polygon(polygon: list[list[float]], height: float, thickness: float, patio: bool = False) -> list[dict[str, Any]]:
    pts = _close(polygon)
    walls = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        walls.append(
            {
                "id": str(uuid4()),
                "start": [float(a[0]), float(a[1])],
                "end": [float(b[0]), float(b[1])],
                "height": 0.15 if patio else height,
                "thickness": thickness,
                "length": float(np.hypot(b[0] - a[0], b[1] - a[1])),
            }
        )
    return walls


def _close(polygon: list[list[float]]) -> list[list[float]]:
    pts = [[float(p[0]), float(p[1])] for p in polygon]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0][:])
    return pts


def _floor_mesh(polygon: list[list[float]]) -> trimesh.Trimesh | None:
    """Triangulated floor. Room outlines are concave (L-shapes, bays), so a fan would spill outside."""
    from shapely.geometry import Polygon
    from shapely.ops import triangulate

    if len(polygon) < 3:
        return None
    shape = Polygon(polygon)
    if not shape.is_valid:
        shape = shape.buffer(0)
    tris = [t for t in triangulate(shape) if shape.contains(t.representative_point())]
    if not tris:
        return None
    verts, faces = [], []
    for tri in tris:
        base = len(verts)
        for x, z in list(tri.exterior.coords)[:3]:
            verts.append([x, 0.01, z])
        faces.append([base, base + 2, base + 1])  # counter-clockwise seen from above (+Y)
    return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)


def _wall_mesh(start: list[float], end: list[float], height: float, thickness: float, elevation: float = 0.0) -> trimesh.Trimesh:
    x0, z0 = float(start[0]), float(start[1])
    x1, z1 = float(end[0]), float(end[1])
    length = max(np.hypot(x1 - x0, z1 - z0), 0.01)
    box = trimesh.creation.box(extents=[length, height, thickness])
    angle = np.arctan2(z1 - z0, x1 - x0)
    rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    box.apply_transform(rot)
    box.apply_translation([(x0 + x1) / 2, elevation + height / 2, (z0 + z1) / 2])
    return box
