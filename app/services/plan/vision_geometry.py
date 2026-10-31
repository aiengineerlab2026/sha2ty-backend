"""2D layout from the vision analysis, refined by OpenCV — used when OpenCV alone cannot read the plan.

Why: OpenCV finds rooms as free space between thick wall bodies. Drawings with thin, coloured or
hatched walls, heavy furniture or textured floors do not seal into rooms, and the whole flat came out
as one region (named by the size fallback, e.g. "Closet"). Here the model says WHAT is where; OpenCV
says exactly WHERE: model walls snap onto detected wall lines, door jambs onto detected openings.

Output has the same shape as room_segmenter.segment(), so validation, scale, 2D, 3D and GLB all run
on this one layout.
"""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from app.schemas.plan_analysis import PlanAnalysis
from app.services.plan.analysis_guidance import KIND, NAME
from app.services.plan.room_segmenter import Segmentation, _chords
from app.services.plan.wall_detector import Opening, WallSegment

MAIN_KINDS = {"living", "kitchen", "dining", "bedroom", "bathroom", "hall", "patio"}


def opencv_agreement(analysis: PlanAnalysis, zones: list[dict], size: tuple[int, int]) -> float:
    """Share of the model's main rooms that OpenCV found as a zone of the same type."""
    main = [r for r in analysis.rooms if KIND[r.type] in MAIN_KINDS and Polygon(r.polygon).area >= 0.01]
    if not main:
        return 1.0
    hits = 0
    for r in main:
        c = Polygon([(x * size[0], y * size[1]) for x, y in r.polygon]).representative_point()
        hits += any(z["kind"] == KIND[r.type] and Polygon(z["polygon"]).buffer(2).contains(c) for z in zones if len(z["polygon"]) >= 3)
    return hits / len(main)


def segment_from_analysis(
    analysis: PlanAnalysis, size: tuple[int, int], cv_walls: list[WallSegment], cv_openings: list[Opening], cv_thickness: float
) -> tuple[Segmentation, list[WallSegment], list[Opening], float]:
    W, H = size
    diag = float(np.hypot(W, H))
    thickness = cv_thickness if 0.002 * diag <= cv_thickness <= 0.03 * diag else 0.008 * diag
    px = lambda p: (float(p[0]) * W, float(p[1]) * H)  # noqa: E731
    errors: list[str] = []
    warnings: list[str] = []

    # Doors and windows first: jambs snap to a detected opening when one is close.
    reach = max(0.03 * diag, 2 * thickness)
    openings: list[Opening] = []
    for kind, items in (("door", analysis.doors), ("window", analysis.windows)):
        for item in items:
            a, b = px(item.start), px(item.end)
            mid = Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            near = [o for o in cv_openings if Point((o.start[0] + o.end[0]) / 2, (o.start[1] + o.end[1]) / 2).distance(mid) <= reach]
            if near:
                o = min(near, key=lambda o: Point((o.start[0] + o.end[0]) / 2, (o.start[1] + o.end[1]) / 2).distance(mid))
                openings.append(Opening(o.start, o.end, o.thickness, kind, f"vision {item.id} + opencv", max(o.confidence, item.confidence)))
            else:
                # ponytail: unmeasured jambs stay below the 0.6 bar, so they never set the scale
                openings.append(Opening(a, b, thickness, kind, f"vision {item.id}", min(item.confidence, 0.5)))

    # Walls: snap to OpenCV lines, merge duplicates (a shared wall listed by both rooms), cut at openings.
    walls = _merge([_snap(px(w.start), px(w.end), cv_walls, thickness, diag) for w in analysis.walls], thickness)
    walls = _cut(walls, openings, thickness)

    # Rooms: the model's polygons, valid, inside the image, overlaps given to the first (smaller) room.
    polys = []
    frame = Polygon([(0, 0), (W, 0), (W, H), (0, H)])
    for room in sorted(analysis.rooms, key=lambda r: Polygon(r.polygon).area):
        poly = Polygon([px(p) for p in room.polygon]).buffer(0).intersection(frame)
        for _, other in polys:
            poly = poly.difference(other)
        poly = max(getattr(poly, "geoms", [poly]), key=lambda g: g.area) if not poly.is_empty else poly
        if poly.is_empty or poly.area < max(thickness * thickness * 4, 1e-4 * W * H):
            errors.append(f"{room.name}: its outline is empty after removing overlaps.")
            continue
        polys.append((room, poly.simplify(1.0)))
    order = {r.id: i for i, r in enumerate(analysis.rooms)}
    polys.sort(key=lambda rp: order[rp[0].id])

    rooms, zones, masks = [], [], []
    for index, (room, poly) in enumerate(polys):
        rid = f"r{index + 1}"
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.array(poly.exterior.coords, np.int32)], 1)
        mask = mask.astype(bool)
        coords = [[round(x, 1), round(y, 1)] for x, y in list(poly.exterior.coords)[:-1]]
        label_point = [round(poly.representative_point().x, 1), round(poly.representative_point().y, 1)]
        kind = KIND[room.type]
        dims = [room.estimated_width_m, room.estimated_length_m] if room.dimension_source == "annotated" and room.estimated_width_m and room.estimated_length_m else None
        zones.append(
            {
                "id": f"{rid}z1", "room_id": rid, "name": NAME[room.type], "kind": kind, "polygon": coords, "label_point": label_point,
                "boundary": "vision", "label_index": None, "boundary_support": 1.0, "dims_m": dims, "dims_text": room.dimension_text if dims else None,
                "dims_uncertain": False, "measure_px": _chords(mask), "vision_id": room.id,
            }
        )
        rooms.append(
            {
                "id": rid, "name": NAME[room.type], "kind": kind, "polygon": coords, "holes": [], "label_point": label_point,
                "area_px": int(mask.sum()), "zone_ids": [f"{rid}z1"], "uncertain": False, "invalid": [], "crosses_walls": 0.0, "vision_id": room.id,
            }
        )
        masks.append(mask)

    # Footprint: the enclosed apartment (a balcony stays outside it).
    inside = [p for r, p in polys if KIND[r.type] != "patio"]
    footprint: list[list[float]] = []
    if inside:
        hull = unary_union([p.buffer(thickness) for p in inside]).buffer(-thickness * 0.5)
        hull = max(getattr(hull, "geoms", [hull]), key=lambda g: g.area)
        footprint = [[round(x, 1), round(y, 1)] for x, y in list(hull.simplify(thickness * 0.5).exterior.coords)[:-1]]
        if hull.area < 0.05 * W * H:
            errors.append("The detected apartment covers almost none of the image.")
    else:
        errors.append("No enclosed rooms were found.")

    adjacency = _adjacency(rooms, polys, walls, openings, thickness)
    sealed = np.zeros((H, W), bool)
    return Segmentation(rooms, zones, adjacency, footprint, sealed, ~sealed, masks, warnings, errors), walls, openings, thickness


def _snap(a, b, cv_walls: list[WallSegment], thickness: float, diag: float) -> WallSegment:
    """Move a model wall onto the parallel detected wall it overlaps, if one is within 2% of the image."""
    horizontal = abs(b[1] - a[1]) <= abs(b[0] - a[0])
    lo, hi = sorted((a[0], b[0])) if horizontal else sorted((a[1], b[1]))
    pos = (a[1] + b[1]) / 2 if horizontal else (a[0] + b[0]) / 2
    best = None
    for w in cv_walls:
        if w.orientation != ("h" if horizontal else "v"):
            continue
        wlo, whi = sorted((w.start[0], w.end[0])) if horizontal else sorted((w.start[1], w.end[1]))
        overlap = min(hi, whi) - max(lo, wlo)
        wpos = (w.start[1] + w.end[1]) / 2 if horizontal else (w.start[0] + w.end[0]) / 2
        if overlap >= 0.4 * (hi - lo) and abs(wpos - pos) <= 0.02 * diag and (best is None or abs(wpos - pos) < abs(best[0] - pos)):
            best = (wpos, w.thickness)
    pos, t = best if best else (pos, thickness)
    start, end = ((lo, pos), (hi, pos)) if horizontal else ((pos, lo), (pos, hi))
    # ponytail: model walls are treated as axis-aligned; diagonal plans need a third case here
    return WallSegment(start, end, t, "h" if horizontal else "v", 0.8 if best else 0.5)


def _merge(walls: list[WallSegment], thickness: float) -> list[WallSegment]:
    """Collinear walls within half a thickness that overlap or touch become one."""
    out: list[WallSegment] = []
    for orient in ("h", "v"):
        group = sorted((w for w in walls if w.orientation == orient), key=lambda w: (w.start[1], w.start[0]) if orient == "h" else (w.start[0], w.start[1]))
        rows: list[list] = []  # [pos, lo, hi, thickness, confidence]
        for w in group:
            pos = w.start[1] if orient == "h" else w.start[0]
            lo, hi = sorted((w.start[0], w.end[0])) if orient == "h" else sorted((w.start[1], w.end[1]))
            row = next((r for r in rows if abs(r[0] - pos) <= thickness * 0.5 and lo <= r[2] + thickness and hi >= r[1] - thickness), None)
            if row:
                row[1], row[2], row[3], row[4] = min(row[1], lo), max(row[2], hi), max(row[3], w.thickness), max(row[4], w.confidence)
            else:
                rows.append([pos, lo, hi, w.thickness, w.confidence])
        for pos, lo, hi, t, c in rows:
            out.append(WallSegment((lo, pos), (hi, pos), t, "h", c) if orient == "h" else WallSegment((pos, lo), (pos, hi), t, "v", c))
    return out


def _cut(walls: list[WallSegment], openings: list[Opening], thickness: float) -> list[WallSegment]:
    """Doors and windows are gaps in their wall, never solid geometry (3D adds lintels and sills)."""
    out = []
    for w in walls:
        line = LineString([w.start, w.end])
        for o in openings:
            span = LineString([o.start, o.end])
            if span.distance(line) <= max(w.thickness, thickness):
                # The opening's extent projected onto the wall.
                s0, s1 = sorted((line.project(Point(o.start)), line.project(Point(o.end))))
                if s1 - s0 > 1:
                    parts = [line.interpolate(0), line.interpolate(s0)], [line.interpolate(s1), line.interpolate(line.length)]
                    pieces = [LineString(p) for p in parts if p[0].distance(p[1]) > 1]
                    line = unary_union(pieces) if pieces else LineString()
            if line.is_empty:
                break
        for piece in getattr(line, "geoms", [line]):
            if not piece.is_empty and piece.length >= thickness:
                (x0, y0), (x1, y1) = piece.coords[0], piece.coords[-1]
                out.append(WallSegment((x0, y0), (x1, y1), w.thickness, w.orientation, w.confidence))
    return out


def _adjacency(rooms: list[dict], polys, walls: list[WallSegment], openings: list[Opening], thickness: float) -> list[dict]:
    """Rooms touching: through a door, an open side (no wall along the shared edge) or a wall."""
    wall_lines = unary_union([LineString([w.start, w.end]).buffer(w.thickness / 2 + 2) for w in walls]) if walls else None
    edges = []
    for i, (a, (_, pa)) in enumerate(zip(rooms, polys)):
        for b, (_, pb) in list(zip(rooms, polys))[i + 1 :]:
            if pa.distance(pb) > thickness * 1.5:
                continue
            shared = pa.buffer(thickness).intersection(pb.buffer(thickness))
            # A door joins two rooms when most of it lies on their shared edge (not just touches a corner).
            doors = [
                {"index": k, "kind": o.kind} for k, o in enumerate(openings)
                if o.kind == "door" and LineString([o.start, o.end]).intersection(shared).length >= 0.6 * LineString([o.start, o.end]).length
            ]
            open_side = wall_lines is None or shared.difference(wall_lines).area > 0.3 * shared.area
            edges.append({"a": a["id"], "b": b["id"], "via": "door" if doors else ("opening" if open_side else "wall"), "openings": doors})
    return edges


def draw_analysis(image: np.ndarray, analysis: PlanAnalysis) -> np.ndarray:
    """Debug B: the model's room polygons (distinct colours + labels), walls, doors and windows on the plan."""
    out = image.copy()
    H, W = out.shape[:2]
    rng = np.random.default_rng(7)
    overlay = out.copy()
    for room in analysis.rooms:
        pts = np.array([(x * W, y * H) for x, y in room.polygon], np.int32)
        color = tuple(int(c) for c in rng.integers(60, 230, 3))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], True, color, 2)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for w in analysis.walls:
        cv2.line(out, (int(w.start[0] * W), int(w.start[1] * H)), (int(w.end[0] * W), int(w.end[1] * H)), (40, 40, 200) if w.exterior else (40, 140, 220), 2)
    for color, items in (((0, 160, 0), analysis.doors), ((200, 120, 0), analysis.windows)):
        for d in items:
            cv2.line(out, (int(d.start[0] * W), int(d.start[1] * H)), (int(d.end[0] * W), int(d.end[1] * H)), color, 4)
    for f in analysis.furniture:
        cv2.rectangle(out, (int(f.bbox[0] * W), int(f.bbox[1] * H)), (int(f.bbox[2] * W), int(f.bbox[3] * H)), (150, 150, 150), 1)
    for room in analysis.rooms:
        c = Polygon([(x * W, y * H) for x, y in room.polygon]).representative_point()
        cv2.putText(out, f"{room.id} {room.name}"[:28], (int(c.x) - 30, int(c.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    return out
