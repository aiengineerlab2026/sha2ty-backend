"""2D footprint validation in meters. XZ plane, Y is height."""

from __future__ import annotations

from typing import Any

from shapely.geometry import LineString, Polygon, box
from shapely.affinity import rotate, translate


def furniture_footprint(
    x: float,
    z: float,
    width: float,
    depth: float,
    rotation_y_deg: float,
) -> Polygon:
    poly = box(-width / 2, -depth / 2, width / 2, depth / 2)
    # three.js rotation.y = +θ turns +X towards −Z; in the (x, z) plane that is shapely's −θ.
    poly = rotate(poly, -rotation_y_deg, origin=(0, 0), use_radians=False)
    return translate(poly, xoff=x, yoff=z)


def room_polygon(room: dict[str, Any]) -> Polygon:
    pts = [(float(p[0]), float(p[1])) for p in room["polygon"]]
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def wall_segment(wall: dict[str, Any]) -> LineString:
    return LineString([tuple(wall["start"]), tuple(wall["end"])])


def validate_placement(
    item: dict[str, Any],
    rooms: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    others: list[dict[str, Any]],
    height_limit: float,
) -> list[str]:
    warnings: list[str] = []
    name = str(item.get("title") or "Furniture")
    width = float(item["width_m"])
    depth = float(item["depth_m"])
    height = float(item.get("height_m") or 0)
    footprint = furniture_footprint(float(item["x"]), float(item["z"]), width, depth, float(item.get("rotation_y") or 0))
    if not rooms:
        warnings.append("Place furniture after a room exists.")
        return warnings

    containing = next((room for room in rooms if room_polygon(room).buffer(0.02).contains(footprint.centroid)), None)
    if containing is None:
        warnings.append(f"{name} is outside the room.")
        return warnings

    poly = room_polygon(containing)
    room_w = float(containing.get("width") or 0)
    room_d = float(containing.get("depth") or 0)
    if room_w and max(width, depth) > room_w + 0.005 and min(width, depth) <= room_d + 0.05:
        over = max(width, depth) - room_w
        warnings.append(f"{name} exceeds available wall length by {over:.2f}m.")
    if not poly.buffer(0.02).contains(footprint):
        overflow = footprint.difference(poly)
        extra = overflow.bounds
        span = 0.0
        if overflow and not overflow.is_empty:
            minx, minz, maxx, maxz = extra
            span = max(maxx - minx, maxz - minz)
        wall_len = _nearest_wall_length(footprint, walls)
        if wall_len is not None and max(width, depth) > wall_len + 0.005:
            over = max(width, depth) - wall_len
            warnings.append(f"{name} exceeds available wall length by {over:.2f}m.")
        elif span > 0:
            warnings.append(f"{name} exceeds available floor space by {span:.2f}m.")
        else:
            warnings.append(f"{name} intersects a wall.")

    for wall in walls:
        thick = float(wall.get("thickness") or 0.15)
        if footprint.distance(wall_segment(wall)) < thick / 2 - 0.01 and footprint.intersects(wall_segment(wall).buffer(thick / 2)):
            if f"{name} intersects a wall." not in warnings:
                warnings.append(f"{name} intersects a wall.")

    for other in others:
        other_fp = furniture_footprint(
            float(other["x"]),
            float(other["z"]),
            float(other["width_m"]),
            float(other["depth_m"]),
            float(other.get("rotation_y") or 0),
        )
        if footprint.intersects(other_fp.buffer(-0.01)):
            warnings.append(f"{name} intersects {other.get('title') or 'another object'}.")

    if height > height_limit + 0.01:
        warnings.append(f"{name} exceeds room height by {height - height_limit:.2f}m.")
    return warnings


def _nearest_wall_length(footprint: Polygon, walls: list[dict[str, Any]]) -> float | None:
    if not walls:
        return None
    nearest = min(walls, key=lambda wall: footprint.distance(wall_segment(wall)))
    return float(nearest.get("length") or LineString([tuple(nearest["start"]), tuple(nearest["end"])]).length)
