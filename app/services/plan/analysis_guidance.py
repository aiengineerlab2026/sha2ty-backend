"""Use the vision analysis as semantic guidance for the OpenCV pipeline — never as final geometry.

- labels: room names/types (and written sizes) where OCR found none → the segmenter splits open plans
- walls: detected "walls" lying inside a furniture box are furniture edges and are dropped
- scale: the model points at references; OpenCV measures them in pixels (doors = detected openings,
  objects = ink outlines), standard sizes replace the model's own numbers
- fixtures: furniture boxes refined to their drawn outline, for simple 3D primitives
All inputs are normalised (schemas/plan_analysis.py); everything returned here is in pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from app.schemas.plan_analysis import PlanAnalysis
from app.services.plan.room_segmenter import RoomLabel
from app.services.plan.scale_estimator import DOOR_WIDTH_M

KIND = {"living": "living", "kitchen": "kitchen", "dining": "dining", "bedroom": "bedroom", "bathroom": "bathroom", "hall": "hall",
        "balcony": "patio", "closet": "closet", "laundry": "laundry", "office": "room", "other": "room"}
NAME = {"living": "Living Room", "kitchen": "Kitchen", "dining": "Dining", "bedroom": "Bedroom", "bathroom": "Bathroom", "hall": "Hall",
        "balcony": "Balcony", "closet": "Closet", "laundry": "Laundry", "office": "Office", "other": "Room"}  # short, like OCR labels
STANDARD_M = {"bed": 2.0, "bathtub": 1.6, "countertop": 0.6}  # long side, long side, depth
FIXTURE_HEIGHT_M = {"bed": 0.5, "sofa": 0.8, "table": 0.75, "dining_table": 0.75, "chair": 0.9, "wardrobe": 2.0, "cabinet": 0.9,
                    "countertop": 0.9, "stove": 0.9, "fridge": 1.8, "sink": 0.9, "toilet": 0.75, "shower": 0.05, "bathtub": 0.55,
                    "washer": 0.85, "desk": 0.75, "tv": 0.5, "other": 0.6}


def _px(p, size) -> tuple[float, float]:
    return float(p[0]) * size[0], float(p[1]) * size[1]


def labels_from_analysis(analysis: PlanAnalysis | None, labels: list[RoomLabel], size: tuple[int, int]) -> list[RoomLabel]:
    """Model rooms become labels unless OCR already labelled that room with the same type."""
    if not analysis:
        return []
    out = []
    for room in analysis.rooms:
        poly = Polygon([_px(p, size) for p in room.polygon])
        kind = KIND[room.type]
        if any(l.kind == kind and poly.contains(Point(l.x, l.y)) for l in labels):
            continue
        anchor = poly.representative_point()
        dims = None
        if room.dimension_source == "annotated" and room.estimated_width_m and room.estimated_length_m:
            dims = (room.estimated_width_m, room.estimated_length_m)  # written on the plan: calibrate() checks it against the drawing
        out.append(
            RoomLabel(
                name=NAME[room.type], kind=kind, x=anchor.x, y=anchor.y, box=(anchor.x - 1, anchor.y - 1, 2, 2), text=room.name,
                dims_m=dims, dims_text=room.dimension_text if dims else None,
            )
        )
    return out


def drop_furniture_walls(walls: list, analysis: PlanAnalysis | None, size: tuple[int, int]) -> tuple[list, int]:
    """A wall segment with both ends inside one furniture box is that furniture's outline."""
    if not analysis or not analysis.furniture:
        return walls, 0
    boxes = []
    for f in analysis.furniture:
        x0, y0 = _px(f.bbox[:2], size)
        x1, y1 = _px(f.bbox[2:], size)
        boxes.append((x0 - 3, y0 - 3, x1 + 3, y1 + 3))

    def inside(p, b):
        return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]

    kept = [w for w in walls if not any(inside(w.start, b) and inside(w.end, b) for b in boxes)]
    return kept, len(walls) - len(kept)


def door_references(openings: list, analysis: PlanAnalysis | None, size: tuple[int, int], thickness: float) -> list[dict]:
    """Every detected door opening; ones the model also saw (and sized) weigh more."""
    llm_doors = [(Point(*_px(((d.start[0] + d.end[0]) / 2, (d.start[1] + d.end[1]) / 2), size)), d) for d in (analysis.doors if analysis else [])]
    reach = max(0.03 * float(np.hypot(*size)), 2 * thickness)
    refs = []
    for o in openings:
        if o.kind != "door" or o.confidence < 0.6:
            continue
        width = float(np.hypot(o.end[0] - o.start[0], o.end[1] - o.start[1]))
        if width <= 4:
            continue
        mid = Point((o.start[0] + o.end[0]) / 2, (o.start[1] + o.end[1]) / 2)
        hit = min(llm_doors, key=lambda pd: pd[0].distance(mid), default=None)
        if hit and hit[0].distance(mid) <= reach:
            if hit[1].swing in ("sliding", "double"):
                continue  # no standard width: a balcony slider or double door would skew the scale
            real = min(max(hit[1].width_m or DOOR_WIDTH_M, 0.7), 0.9)  # the model's width, held to the standard range
            refs.append({"source": "door", "label": f"Door {hit[1].id} ({width:.0f} px ≈ {real:.2f} m)", "mpp": real / width, "weight": 1.2 * o.confidence})
        else:
            refs.append({"source": "door", "label": f"Door ({width:.0f} px ≈ {DOOR_WIDTH_M} m)", "mpp": DOOR_WIDTH_M / width, "weight": 0.6 * o.confidence})
    return refs


def object_references(analysis: PlanAnalysis | None, gray: np.ndarray, size: tuple[int, int]) -> list[dict]:
    """Written dimension lines and standard objects the model pointed at, measured by OpenCV."""
    if not analysis:
        return []
    refs = []
    for r in analysis.scale_references:
        a, b = _px(r.start, size), _px(r.end, size)
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if length < 8:
            continue
        if r.kind == "dimension_annotation":
            if r.dimension_source == "annotated":
                refs.append({"source": "annotation", "label": f"Dimension {r.text or ''} = {r.real_m:.2f} m".replace("  ", " "), "mpp": r.real_m / length, "weight": 1.5})
            continue
        if r.kind == "door":
            continue  # door_references measures doors on the detected openings
        measured = _outline_side(gray, a, b, length)
        real = STANDARD_M.get(r.kind, r.real_m)
        if measured is None:
            refs.append({"source": "furniture", "label": f"{r.kind} (model's outline) ≈ {real} m", "mpp": real / length, "weight": 0.25})
        else:
            refs.append({"source": "furniture", "label": f"{r.kind} ({measured:.0f} px ≈ {real} m)", "mpp": real / measured, "weight": 0.6 if r.kind in STANDARD_M else 0.3})
    return refs


def room_estimate_references(analysis: PlanAnalysis | None, zones: list[dict]) -> list[dict]:
    """The model's estimated (not written) room sizes, matched to detected zones of the same type and shape."""
    if not analysis:
        return []
    refs, used = [], set()
    for room in analysis.rooms:
        if room.dimension_source != "inferred" or not room.estimated_width_m or not room.estimated_length_m:
            continue
        best = None
        for z in zones:
            m = z.get("measure_px") or {}
            if z["id"] in used or not m.get("width") or not m.get("depth") or z.get("kind") != KIND[room.type]:
                continue
            for w_m, l_m in ((room.estimated_width_m, room.estimated_length_m), (room.estimated_length_m, room.estimated_width_m)):
                mx, my = w_m / m["width"], l_m / m["depth"]
                spread = abs(mx - my) / max(mx, my)
                if spread < 0.2 and (best is None or spread < best[0]):
                    best = (spread, z["id"], (mx + my) / 2)
        if best:
            used.add(best[1])
            refs.append({"source": "room_estimate", "label": f"{room.name} ≈ {room.estimated_width_m:.1f} × {room.estimated_length_m:.1f} m (estimated)", "mpp": best[2], "weight": 0.5})
    return refs


def fixtures_px(analysis: PlanAnalysis | None, gray: np.ndarray, size: tuple[int, int]) -> list[dict]:
    """Furniture boxes snapped to their drawn outline when one fills the box; otherwise the model's box."""
    if not analysis:
        return []
    out = []
    for f in analysis.furniture:
        x0, y0 = _px(f.bbox[:2], size)
        x1, y1 = _px(f.bbox[2:], size)
        w, h = x1 - x0, y1 - y0
        if w < 4 or h < 4:
            continue
        center, dims, angle, refined = [(x0 + x1) / 2, (y0 + y1) / 2], [w, h], float(f.orientation_deg), False
        pad = 0.1 * max(w, h)
        ink, ox, oy = _ink(gray, x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        if ink is not None:
            contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                (cx, cy), (bw, bh), a = cv2.minAreaRect(c)
                if 0.4 * w * h <= bw * bh <= 1.6 * w * h:
                    center, dims, angle, refined = [cx + ox, cy + oy], [float(bw), float(bh)], float(a), True
        out.append({"id": f.id, "type": f.type, "room_id": f.room_id, "center": center, "size": dims, "angle": angle,
                    "height_m": FIXTURE_HEIGHT_M[f.type], "confidence": f.confidence, "refined": refined})
    return out


def _ink(gray: np.ndarray, x0, y0, x1, y1):
    h, w = gray.shape[:2]
    ox, oy = max(int(x0), 0), max(int(y0), 0)
    crop = gray[oy : min(int(y1) + 1, h), ox : min(int(x1) + 1, w)]
    if crop.size == 0:
        return None, ox, oy
    ink = (crop < 128).astype(np.uint8) * 255
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), ox, oy


def _outline_side(gray: np.ndarray, a, b, length: float) -> float | None:
    """The side of an ink outline near the segment that best matches its length (±35%)."""
    pad = 0.6 * length
    ink, _, _ = _ink(gray, min(a[0], b[0]) - pad, min(a[1], b[1]) - pad, max(a[0], b[0]) + pad, max(a[1], b[1]) + pad)
    if ink is None:
        return None
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sides = [s for c in contours for s in cv2.minAreaRect(c)[1] if s >= 6]
    best = min(sides, key=lambda s: abs(s - length), default=None)
    return float(best) if best is not None and abs(best - length) <= 0.35 * length else None
