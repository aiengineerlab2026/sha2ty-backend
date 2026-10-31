"""Rooms = free space enclosed by structural walls; zones = labelled functional areas inside them.

Physical rooms come only from walls plus verified door/window closures. When one enclosed space
carries several room labels (open-plan living/dining/kitchen) it stays ONE physical room and is
divided into logical zones along the shortest cut between labels, preferring a drawn floor line.
No wall is ever invented to make a zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from shapely import make_valid, polygonize, unary_union
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import polylabel, split

from app.services.plan.wall_detector import Opening, WallSegment, rasterize_walls, seal_openings

MINOR_LABELS = {"closet"}


@dataclass
class RoomLabel:
    name: str
    kind: str
    x: float
    y: float
    box: tuple[float, float, float, float]
    text: str
    dims_m: tuple[float, float] | None = None
    dims_text: str | None = None
    dims_uncertain: bool = False


@dataclass
class Segmentation:
    rooms: list[dict]
    zones: list[dict]
    adjacency: list[dict]
    footprint: list[list[float]]
    sealed: np.ndarray
    free: np.ndarray
    room_masks: list[np.ndarray]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def segment(
    shape: tuple[int, ...],
    walls: list[WallSegment],
    openings: list[Opening],
    thickness: float,
    labels: list[RoomLabel],
    floor_lines: tuple[np.ndarray, np.ndarray] | None = None,
    zone_lines: list[dict] | None = None,
) -> Segmentation:
    warnings: list[str] = []
    errors: list[str] = []
    wall_mask = rasterize_walls(shape, walls)
    # Close only verified doors/windows, in this mask only. Unverified "opening" candidates stay
    # open so a passage is never turned into an invented wall.
    sealed = seal_openings(wall_mask, [o for o in openings if _kind(o) in ("door", "window")])
    footprint_mask = _footprint(sealed)
    if not footprint_mask.any():
        errors.append("No closed exterior wall loop was found, so the apartment footprint is undefined.")
        empty = np.zeros(shape[:2], bool)
        return Segmentation([], [], [], [], sealed, empty, [], warnings, errors)
    errors.extend(_envelope_leaks(sealed, footprint_mask, thickness))
    footprint_geom = _mask_polygon(footprint_mask, simplify=thickness * 0.5)
    footprint = _coords(footprint_geom) if footprint_geom is not None else []

    free = footprint_mask & ~sealed
    count, cc, stats, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=4)
    # Junction slivers are < 2 T²; the smallest real closets are ~8 T².
    min_area = max(thickness * thickness * 6, footprint_mask.sum() * 0.0015)
    rooms: list[dict] = []
    masks: list[np.ndarray] = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue  # slivers inside wall junctions, gaps between double lines
        mask = cc == i
        poly = _mask_polygon(mask, simplify=1.0)
        if poly is None:
            continue
        rooms.append({"geom": poly, "area_px": area, "measure_px": _chords(mask)})
        masks.append(mask)

    _enforce_topology(rooms, footprint_geom, wall_mask, masks, errors)
    for mask in _outdoor_spaces(sealed, footprint_mask, floor_lines, labels, thickness):
        poly = _mask_polygon(mask, simplify=1.0)
        if poly is None:
            continue
        rooms.append({"geom": poly, "area_px": int(mask.sum()), "measure_px": _chords(mask), "outdoor": True})
        masks.append(mask)
        for o in openings:  # a window giving onto the balcony is the balcony door
            start, end = (o["start"], o["end"]) if isinstance(o, dict) else (o.start, o.end)
            if _kind(o) == "window" and poly.distance(Point((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)) <= thickness * 1.6:
                if isinstance(o, dict):
                    o.update(kind="door", evidence="opens onto the balcony")
                else:
                    o.kind, o.evidence = "door", "opens onto the balcony"

    # Labels → rooms. Points outside every room (inside a wall, or outside) go to the nearest room.
    for room in rooms:
        room["labels"] = []
    for label in labels:
        pt = Point(label.x, label.y)
        host = next((r for r in rooms if r["geom"].buffer(thickness * 0.5).contains(pt)), None)
        if host is None and rooms:
            near = min(rooms, key=lambda r: r["geom"].distance(pt))
            if near["geom"].distance(pt) <= thickness * 3:
                host = near
        if host is not None:
            host["labels"].append(label)

    zones: list[dict] = []
    straight = floor_lines
    for index, room in enumerate(rooms):
        room["id"] = f"r{index + 1}"
        major = [l for l in room["labels"] if l.kind not in MINOR_LABELS]
        room["uncertain"] = False
        kinds = [l.kind for l in major]
        if len(kinds) != len(set(kinds)) and any(k in ("bedroom", "bathroom") for k in kinds):
            room["uncertain"] = True
            warnings.append(
                f"One enclosed space holds {kinds.count(max(set(kinds), key=kinds.count))} "
                f"'{max(set(kinds), key=kinds.count)}' labels; a wall is probably missing. Split it in the correction view."
            )
        user_lines = [z for z in (zone_lines or []) if room["geom"].buffer(1).intersects(LineString([z["start"], z["end"]]))]
        parts = _split_zones(room["geom"], major or room["labels"], straight, thickness, user_lines)
        for z_index, (geom, zone_labels, source, support) in enumerate(parts):
            zone_id = f"{room['id']}z{z_index + 1}"
            label = zone_labels[0] if zone_labels else None
            zones.append(
                {
                    "id": zone_id,
                    "room_id": room["id"],
                    "name": label.name if label else _fallback_name(room, thickness),
                    "kind": label.kind if label else _fallback_kind(room, thickness),
                    "polygon": _coords(geom),
                    "label_point": _label_point(geom),
                    "boundary": source,
                    "label_index": next((i for i, l in enumerate(labels) if l is label), None),
                    "boundary_support": round(support, 2),
                    "dims_m": list(label.dims_m) if label and label.dims_m else None,
                    "dims_text": label.dims_text if label else None,
                    "dims_uncertain": bool(label.dims_uncertain) if label else False,
                    "measure_px": room["measure_px"] if len(parts) == 1 else _chords(_raster(geom, sealed.shape) & ~sealed),
                }
            )
        room["zone_ids"] = [z["id"] for z in zones if z["room_id"] == room["id"]]

    adjacency = _adjacency(rooms, openings, thickness)
    out_rooms = []
    for room in rooms:
        names = [z["name"] for z in zones if z["room_id"] == room["id"]]
        kinds = [z["kind"] for z in zones if z["room_id"] == room["id"]]
        out_rooms.append(
            {
                "id": room["id"],
                "name": " / ".join(dict.fromkeys(names)),
                "kind": kinds[0] if len(set(kinds)) == 1 else "open_plan",
                "polygon": _coords(room["geom"]),
                "holes": [[[round(x, 1), round(y, 1)] for x, y in ring.coords[:-1]] for ring in room["geom"].interiors],
                "label_point": _label_point(room["geom"]),
                "area_px": room["area_px"],
                "zone_ids": room["zone_ids"],
                "uncertain": room["uncertain"],
                "invalid": room.get("invalid", []),
                "crosses_walls": room.get("crosses_walls", 0.0),
            }
        )
    return Segmentation(out_rooms, zones, adjacency, footprint or [], sealed, free, masks, warnings, errors)


def _kind(o) -> str:
    return o["kind"] if isinstance(o, dict) else o.kind


def _outdoor_spaces(sealed, footprint, floor_lines, labels: list[RoomLabel], thickness: float) -> list[np.ndarray]:
    """Balconies/terraces: a labelled area outside the envelope, closed by thin railing lines and
    touching the building. Open to the image edge or larger than the flat → not a balcony."""
    outside = [l for l in labels if l.kind == "patio" and not footprint[int(l.y), int(l.x)]]
    if not outside or floor_lines is None:
        return []
    railings = cv2.dilate((floor_lines[0] | floor_lines[1]).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    free = ~(sealed | railings | footprint)
    count, cc = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    near_building = cv2.dilate(footprint.astype(np.uint8), np.ones((int(thickness * 3) | 1,) * 2, np.uint8)) > 0
    out = []
    for label in outside:
        i = cc[int(label.y), int(label.x)]
        if i == 0:
            continue
        mask = cc == i
        border = mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
        if border or mask.sum() > footprint.sum() or not (mask & near_building).any():
            continue
        # Railing strokes belong to the balcony floor: grow back over them, never into the flat.
        out.append(cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~footprint & ~sealed)
    return out


def _footprint(sealed: np.ndarray) -> np.ndarray:
    """Everything the exterior flood cannot reach, minus detached islands (title text, legends)."""
    canvas = np.pad((~sealed).astype(np.uint8), 1, constant_values=1)  # 1-px frame so the flood surrounds everything
    cv2.floodFill(canvas, None, (0, 0), 0)
    inside = canvas[1:-1, 1:-1] > 0
    building = inside | sealed
    count, labels, stats, _ = cv2.connectedComponentsWithStats(building.astype(np.uint8), connectivity=4)
    if count <= 1:
        return np.zeros_like(sealed)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == best
    # A footprint with no enclosed interior means the envelope leaks; report rather than guess.
    return mask if inside[mask].any() else np.zeros_like(sealed)


def _envelope_leaks(sealed: np.ndarray, footprint: np.ndarray, thickness: float) -> list[str]:
    """Rooms lost to the outside through a missing exterior wall. The walls closed with a large
    kernel approximate the intended envelope; big free areas inside it but outside the footprint
    are rooms the exterior flood swallowed."""
    k = int(thickness * 12) | 1
    # Padded: near the image border, erosion treats outside-the-image as solid and fills the margin.
    padded = np.pad(sealed.astype(np.uint8), k)
    hull = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))[k:-k, k:-k] > 0
    hull = _footprint(hull)
    lost = hull & ~footprint & ~sealed
    count, _, stats, centroids = cv2.connectedComponentsWithStats(lost.astype(np.uint8), connectivity=4)
    out = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= thickness * thickness * 40:
            x, y = centroids[i]
            out.append(f"Exterior wall is open: the space around ({x:.0f}, {y:.0f}) leaks outside. Add the missing wall or mark the opening.")
    return out


def _mask_polygon(mask: np.ndarray, simplify: float) -> Polygon | None:
    # 1-px spurs and necks (between a wall and a seal) give zero-width contour spikes.
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hierarchy = hierarchy[0]
    outer = max((i for i in range(len(contours)) if hierarchy[i][3] < 0), key=lambda i: cv2.contourArea(contours[i]))
    shell = _pixel_ring(contours[outer])
    holes = [
        _pixel_ring(contours[i])
        for i in range(len(contours))
        if hierarchy[i][3] == outer and cv2.contourArea(contours[i]) > 16
    ]
    if len(shell) < 3:
        return None
    geom = _largest(make_valid(Polygon(shell, [h for h in holes if len(h) >= 3])))
    if geom.is_empty:
        return None
    geom = _largest(make_valid(_orthogonalize(geom.simplify(simplify, preserve_topology=True))))
    return None if geom.is_empty else geom


def _pixel_ring(contour: np.ndarray) -> list[tuple[float, float]]:
    # Contour points are pixel centres; pixel edges sit half a pixel out. Rooms are measured from
    # masks (chords), so the polygon only needs to be within half a pixel of the wall face.
    return [(float(p[0][0]), float(p[0][1])) for p in contour]


def _orthogonalize(poly: Polygon, deg: float = 6.0) -> Polygon:
    """Snap near-horizontal/vertical edges exactly; genuinely diagonal edges are kept as drawn."""

    def ring(coords):
        pts = [list(p) for p in coords[:-1]]
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            ang = abs(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))) % 180
            if min(ang, 180 - ang) <= deg:
                y = (a[1] + b[1]) / 2
                a[1] = b[1] = y
            elif abs(ang - 90) <= deg:
                x = (a[0] + b[0]) / 2
                a[0] = b[0] = x
        return pts

    try:
        snapped = Polygon(ring(list(poly.exterior.coords)), [ring(list(r.coords)) for r in poly.interiors])
    except Exception:
        return poly
    return snapped if snapped.is_valid and abs(snapped.area - poly.area) < poly.area * 0.05 else poly


def _chords(mask: np.ndarray) -> dict:
    """Clear width/depth = largest axis-aligned rectangle inside the room. Printed sizes describe the
    main rectangle; bays, alcoves and door recesses must not stretch it."""
    ys, xs = np.where(mask)
    if not len(xs):
        return {"width": 0.0, "depth": 0.0}
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    step = 2 if crop.size > 40000 else 1  # half resolution is ±1 px and 4x faster
    small = crop[::step, ::step]
    heights = np.zeros(small.shape[1], int)
    best = (0, 0, 0)
    for row in small:
        heights = np.where(row, heights + 1, 0)
        stack: list[int] = []
        for i in range(len(heights) + 1):
            h = heights[i] if i < len(heights) else 0
            while stack and heights[stack[-1]] >= h:
                top = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                area = top * (i - left)
                if area > best[0]:
                    best = (area, i - left, top)
            stack.append(i)
    return {"width": float(best[1] * step), "depth": float(best[2] * step)}


def _enforce_topology(rooms, footprint: Polygon | None, wall_mask: np.ndarray, masks, errors) -> None:
    """Rooms inside the footprint, not overlapping, not crossing walls. Problems are reported, not hidden."""
    for room, mask in zip(rooms, masks):
        geom = room["geom"]
        problems = []
        if not geom.is_valid:
            geom = make_valid(geom)
            problems.append("self-intersecting outline repaired")
        if footprint is not None and footprint.is_valid:
            outside = geom.difference(footprint.buffer(1.0)).area
            if outside > geom.area * 0.02:
                problems.append("extends outside the apartment footprint")
                geom = geom.intersection(footprint)
        poly_mask = _raster(geom, wall_mask.shape)
        crossing = float((poly_mask & wall_mask).sum()) / max(float(poly_mask.sum()), 1.0)
        room["crosses_walls"] = round(crossing, 3)
        if crossing > 0.03:
            problems.append(f"{crossing:.0%} of the outline lies on structural walls")
        room["geom"] = _largest(geom)
        room["invalid"] = problems
    # Pairwise overlap: components are disjoint so this only catches simplification artefacts.
    for i, a in enumerate(rooms):
        for b in rooms[i + 1 :]:
            inter = a["geom"].intersection(b["geom"]).area
            if inter > 0.5:
                b["geom"] = make_valid(b["geom"].difference(a["geom"]))
                if inter > min(a["geom"].area, b["geom"].area) * 0.01:
                    errors.append("Two room outlines overlapped; the overlap was removed from the smaller one.")
                b["geom"] = _largest(b["geom"])


def _split_zones(geom: Polygon, labels: list[RoomLabel], straight, thickness: float, user_lines: list[dict]):
    """Recursive shortest axis-aligned cut separating labels. Returns (geom, labels, source, support)."""
    if user_lines:
        cutters = unary_union([LineString([z["start"], z["end"]]) for z in user_lines] + [geom.boundary])
        pieces = [p for p in polygonize(cutters).geoms if geom.buffer(0.5).contains(p.representative_point())]
        if len(pieces) > 1:
            out = []
            for piece in pieces:
                inside = [l for l in labels if piece.buffer(thickness * 0.5).contains(Point(l.x, l.y))]
                out.append((piece, inside, "user", 1.0))
            return out
    if len(labels) <= 1:
        return [(geom, labels, "wall", 1.0)]

    best = None
    minx, miny, maxx, maxy = geom.bounds
    corners = [Point(c) for c in geom.exterior.coords]
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            for axis in ("x", "y"):
                lo_box, hi_box = (a, b) if (a.x if axis == "x" else a.y) <= (b.x if axis == "x" else b.y) else (b, a)
                lo = lo_box.box[0] + lo_box.box[2] if axis == "x" else lo_box.box[1] + lo_box.box[3]
                hi = hi_box.box[0] if axis == "x" else hi_box.box[1]
                if hi - lo < 4:
                    continue  # labels overlap along this axis; a cut here cannot separate them
                for c in np.arange(lo + 2, hi - 1, 2.0):
                    line = LineString([(c, miny - 1), (c, maxy + 1)]) if axis == "x" else LineString([(minx - 1, c), (maxx + 1, c)])
                    cut = geom.intersection(line)
                    length = cut.length
                    if length <= 0:
                        continue
                    support = max(_line_support(cut, straight, axis), _corner_support(cut, corners, thickness))
                    score = length * (1 - 0.6 * support)
                    if best is None or score < best[0]:
                        best = (score, axis, c, support)
    if best is None:
        return [(geom, labels, "inferred", 0.0)]
    _, axis, c, support = best
    lo_side = box(minx - 1, miny - 1, c, maxy + 1) if axis == "x" else box(minx - 1, miny - 1, maxx + 1, c)
    hi_side = box(c, miny - 1, maxx + 1, maxy + 1) if axis == "x" else box(minx - 1, c, maxx + 1, maxy + 1)
    source = "drawn_line" if support >= 0.6 else "inferred"
    out = []
    for side in (lo_side, hi_side):
        part = _largest(geom.intersection(side))
        inside = [l for l in labels if (l.x < c if axis == "x" else l.y < c) == (side is lo_side)]
        for piece, piece_labels, piece_source, piece_support in _split_zones(part, inside, straight, thickness, []):
            out.append((piece, piece_labels, piece_source if len(inside) > 1 else source, piece_support if len(inside) > 1 else support))
    return out


def _corner_support(cut, corners: list[Point], thickness: float) -> float:
    """1 when the cut runs corner to corner of the outline: where a wall stops, the room boundary carries on."""
    if cut.geom_type != "LineString":
        return 0.0
    ends = [Point(cut.coords[0]), Point(cut.coords[-1])]
    return 1.0 if all(min(e.distance(c) for c in corners) <= thickness for e in ends) else 0.0


def _line_support(cut, straight, axis: str) -> float:
    """How much of a cut lies on a distinct drawn hairline (floor-finish change). Lines that repeat
    on both sides of the cut are texture (tile grid), not a boundary, and give no support."""
    if straight is None:
        return 0.0
    mask = straight[1] if axis == "x" else straight[0]  # vertical cut ↔ vertical hairlines
    near = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    def cover(offset: float) -> float:
        h, w = near.shape
        hits = total = 0
        for seg in getattr(cut, "geoms", [cut]):
            if seg.length <= 0:
                continue
            for t in np.linspace(0, 1, max(2, int(seg.length // 2))):
                p = seg.interpolate(t, normalized=True)
                xi = int(round(p.x + (offset if axis == "x" else 0)))
                yi = int(round(p.y + (offset if axis == "y" else 0)))
                total += 1
                hits += bool(0 <= xi < w and 0 <= yi < h and near[yi, xi])
        return hits / total if total else 0.0

    on = cover(0)
    if on < 0.3:
        return on
    before = max(cover(-d) for d in range(5, 26, 2))
    after = max(cover(d) for d in range(5, 26, 2))
    return 0.0 if min(before, after) > on * 0.6 else on


def _adjacency(rooms: list[dict], openings, thickness: float) -> list[dict]:
    edges = []
    reach = thickness * 1.6
    for i, a in enumerate(rooms):
        ga = a["geom"].buffer(reach)
        for b in rooms[i + 1 :]:
            shared = ga.intersection(b["geom"].buffer(reach))
            if shared.is_empty or shared.area < thickness * thickness * 2:
                continue
            doors = []
            for index, o in enumerate(openings):
                start, end, kind = (o["start"], o["end"], o["kind"]) if isinstance(o, dict) else (o.start, o.end, o.kind)
                mid = Point((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                if a["geom"].distance(mid) <= reach and b["geom"].distance(mid) <= reach:
                    doors.append({"index": index, "kind": kind})
            edges.append({"a": a["id"], "b": b["id"], "via": "door" if any(d["kind"] == "door" for d in doors) else ("opening" if doors else "wall"), "openings": doors})
    return edges


def _fallback_kind(room: dict, thickness: float) -> str:
    m = room["measure_px"]
    if min(m["width"], m["depth"]) < thickness * 6 and room["area_px"] < thickness * thickness * 80:
        return "closet"
    return "room"


def _fallback_name(room: dict, thickness: float) -> str:
    # Never a named room type without a label: a mis-sized region once came out as one big "Closet".
    return "Room"


def _polygons(geom) -> list[Polygon]:
    if geom.geom_type == "Polygon":
        return [geom]
    return [p for g in getattr(geom, "geoms", []) for p in _polygons(g)]


def _largest(geom):
    parts = [p for p in _polygons(geom) if not p.is_empty]
    return max(parts, key=lambda g: g.area) if parts else Polygon()


def _raster(geom, shape) -> np.ndarray:
    canvas = np.zeros(shape[:2], np.uint8)
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        cv2.fillPoly(canvas, [np.round(np.array(poly.exterior.coords)).astype(np.int32)], 1)
        for ring in poly.interiors:
            cv2.fillPoly(canvas, [np.round(np.array(ring.coords)).astype(np.int32)], 0)
    return canvas > 0


def _coords(geom) -> list[list[float]]:
    geom = _largest(geom)
    if geom.is_empty:
        return []
    return [[round(x, 1), round(y, 1)] for x, y in geom.exterior.coords[:-1]]


def _label_point(geom) -> list[float]:
    geom = _largest(geom)
    if geom.is_empty:
        return [0.0, 0.0]
    p = polylabel(geom, tolerance=1.0)  # pole of inaccessibility: visual centre, always inside
    return [round(p.x, 1), round(p.y, 1)]
