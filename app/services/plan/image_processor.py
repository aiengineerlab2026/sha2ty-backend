"""Floor-plan image → validated 2D plan (pixel space) → metric 3D geometry.

Pipeline (each stage writes a debug image):
  0. Vision-LLM analysis (vision_analysis.py, done by the API before this): semantic guidance only
  1. OCR on the original image (names, printed dimensions)       → 10_ocr
  2. Structural walls by thickness, H/V/diagonal segments          → 01..07, 20_walls
  3. Verified openings (door arcs, window lines, exterior gaps)    → 21_openings
  4. Free-space rooms + open-plan zones, topology checks           → 30_sealed, 31_rooms
  5. Scale: printed dimensions, else doors / furniture / heuristic → 40_final (scale_estimator.py)
The correction UI edits the pixel-space state (walls/openings/labels/zone lines/scale) and sends it
back to `plan_from_state`, which re-runs steps 4–5 on the edited walls, so preview and 3D agree.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from functools import partial
from dataclasses import asdict

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from app.services.plan.dimension_ocr import calibrate, measurements_from_boxes, parse_dimension, room_labels, run_ocr
from app.services.plan.geometry_builder import DEFAULT_HEIGHT, DEFAULT_THICKNESS, geometry_from_plan
from app.services.plan.room_segmenter import RoomLabel, segment
from app.schemas.plan_analysis import PlanAnalysis
from app.services.plan.analysis_guidance import (
    door_references,
    drop_furniture_walls,
    fixtures_px,
    labels_from_analysis,
    object_references,
    room_estimate_references,
)
from app.services.plan.scale_estimator import estimate_scale, furniture_references
from app.services.plan.vision_geometry import draw_analysis, opencv_agreement, segment_from_analysis
from app.services.plan.wall_detector import Opening, WallSegment, detect_walls

logger = logging.getLogger(__name__)
MAX_SIDE = 2400  # larger scans are downscaled for speed; `image_scale` maps back to the upload


def plan_from_image(
    image_bytes: bytes,
    wall_height: float = DEFAULT_HEIGHT,
    wall_thickness: float = DEFAULT_THICKNESS,
    width_m: float | None = None,
    depth_m: float | None = None,
    debug: bool = False,
    analysis: PlanAnalysis | None = None,
    on_stage=lambda _name, _percent: None,
) -> dict:
    original = _decode_image(image_bytes)
    image, image_scale = _working_copy(original)
    debug_images: dict[str, np.ndarray] = {"00_original": image}

    boxes, ocr_notes = run_ocr(image)
    labels, drawing_unit = room_labels(boxes)
    size = (int(image.shape[1]), int(image.shape[0]))
    labels += labels_from_analysis(analysis, labels, size)  # the model names rooms OCR could not read
    measurements, ocr_warn = measurements_from_boxes(boxes)
    debug_images["10_ocr"] = _draw_ocr(image, boxes, labels)

    on_stage("Extracting rooms", 72)
    text_boxes = [(b.x, b.y, b.width, b.height) for b in boxes]
    walls = detect_walls(image, text_boxes)
    walls.walls, dropped = drop_furniture_walls(walls.walls, analysis, size)
    if dropped:
        walls.warnings.append(f"Ignored {dropped} furniture outline(s) that looked like walls.")
    debug_images.update(walls.debug)
    debug_images["20_walls"] = _draw_walls(image, walls.walls, [])
    debug_images["21_openings"] = _draw_walls(image, walls.walls, walls.openings)

    state = {
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "image_scale": image_scale,
        "thickness_px": walls.thickness,
        "walls_px": [w.as_dict(i) for i, w in enumerate(walls.walls)],
        "openings_px": [o.as_dict(i) for i, o in enumerate(walls.openings)],
        "labels": [_label_dict(l) for l in labels],
        "zone_lines": [],
        "drawing_unit": drawing_unit,
        "straight_lines": _encode_mask(walls.debug["07_straight_thin"]),
        # The validated model output, kept with the plan for later editing (None without a vision model).
        "analysis": analysis.model_dump() if analysis else None,
    }
    if width_m and depth_m and width_m > 0 and depth_m > 0:
        state["manual_extent_m"] = [width_m, depth_m]

    run = partial(plan_from_state, wall_height=wall_height, wall_thickness=wall_thickness, extra_warnings=ocr_notes + ocr_warn + walls.warnings, image=image)
    on_stage("Estimating dimensions", 84)
    payload = run(state, debug_images=debug_images)
    payload["geometry_source"] = "opencv"
    if analysis and analysis.rooms:
        debug_images["25_vision_rooms"] = draw_analysis(image, analysis)
        agreement = opencv_agreement(analysis, payload["zones_px"], size)
        payload["opencv_agreement"] = round(agreement, 2)
        # OpenCV geometry wins whenever it validates: the model's coordinates are approximate. Every
        # room the model names is also a label, so a room OpenCV missed fails validation and only
        # then is the model's layout refined on the detected walls instead.
        if not payload["validation"]["ok"] or len(payload["rooms_px"]) <= 1:
            vision_debug: dict = {}
            vision = run({**state, "geometry_source": "vision"}, debug_images=vision_debug)
            logger.info("OpenCV found %d rooms (agreement %.0f%%, ok=%s); vision layout: %d rooms, ok=%s",
                        len(payload["rooms_px"]), agreement * 100, payload["validation"]["ok"], len(vision["rooms_px"]), vision["validation"]["ok"])
            if vision["validation"]["ok"] or not payload["validation"]["ok"]:
                payload = {**vision, "geometry_source": "vision", "opencv_agreement": round(agreement, 2)}
                debug_images.update(vision_debug)
    on_stage("Generating 3D", 93)
    debug_images["22_refined_walls"] = _draw_walls(image, [_wall(w) for w in payload["walls_px"]], [_opening(o) for o in payload["openings_px"]])
    logger.info(
        "Plan: %s layout, %d rooms %s, scale %s px/m (%s, confidence %.2f), errors %s",
        payload["geometry_source"], len(payload["rooms"]), [r["name"] for r in payload["rooms"]],
        round(1 / payload["meters_per_pixel"], 1) if payload.get("meters_per_pixel") else None,
        payload["scale"]["source"], payload["scale"]["confidence"], payload["validation"]["errors"],
    )
    payload["preview_image"] = _png_data_url(image)
    payload["ocr_measurements"] = [
        {"text": m.text, "meters": m.meters, "unit": m.unit, "pair": m.pair, "label": m.label, "uncertain": m.uncertain,
         "box": [round(m.x, 1), round(m.y, 1), round(m.width, 1), round(m.height, 1)]}
        for m in measurements
    ]
    payload["ocr_text"] = [{"text": b.text, "score": round(b.score, 2), "box": [round(b.x, 1), round(b.y, 1), round(b.width, 1), round(b.height, 1)]} for b in boxes]
    _write_debug_dir(debug_images, hashlib.sha256(image_bytes).hexdigest()[:12])
    if debug:
        payload["debug_images"] = {k: _png_data_url(v, max_side=720) for k, v in debug_images.items() if k[:2] in ("00", "05", "10", "21", "22", "25", "30", "31", "40")}
    return payload


def plan_from_state(
    state: dict,
    wall_height: float = DEFAULT_HEIGHT,
    wall_thickness: float = DEFAULT_THICKNESS,
    meters_per_pixel: float | None = None,
    scale_confirmed: bool = False,
    extra_warnings: list[str] | None = None,
    debug_images: dict | None = None,
    image: np.ndarray | None = None,
) -> dict:
    """Segment rooms from (possibly user-edited) walls/openings/labels and build metric geometry."""
    size = state["image_size"]
    shape = (int(size["height"]), int(size["width"]))
    thickness = float(state.get("thickness_px") or _median_thickness(state["walls_px"]))
    walls = [_wall(w) for w in state["walls_px"]]
    openings = [_opening(o) for o in state.get("openings_px") or []]
    labels = [_label(l, state.get("drawing_unit")) for l in state.get("labels") or []]
    analysis = PlanAnalysis.model_validate(state["analysis"]) if state.get("analysis") else None
    size = (int(size["width"]), int(size["height"]))
    if state.get("geometry_source") == "vision" and analysis:
        # The model's layout, refined on the detected walls/openings (vision_geometry.py).
        seg, walls, openings, thickness = segment_from_analysis(analysis, size, walls, openings, thickness)
        state = {
            **state, "thickness_px": thickness,
            "walls_px": [w.as_dict(i) for i, w in enumerate(walls)], "openings_px": [o.as_dict(i) for i, o in enumerate(openings)],
        }
    else:
        straight = _decode_mask(state.get("straight_lines"), shape)
        seg = segment(shape, walls, openings, thickness, labels, (straight, straight) if straight is not None else None, state.get("zone_lines"))
        state = {**state, "openings_px": [o.as_dict(i) for i, o in enumerate(openings)]}  # balcony windows became doors
    if image is not None and "scale_refs" not in state:
        # Object references and fixtures need the drawing: measured once on upload, carried in the state.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        objects = object_references(analysis, gray, size)
        if not any(r["source"] == "furniture" for r in objects):
            objects += furniture_references(gray, seg.zones)  # no model, or it pointed at no furniture
        state = {**state, "scale_refs": objects + room_estimate_references(analysis, seg.zones), "fixtures_px": fixtures_px(analysis, gray, size)}
    # Doors are re-measured on every run: the correction view can edit openings.
    refs = door_references(openings, analysis, size, thickness) + [dict(r) for r in state.get("scale_refs") or []]
    scale = estimate_scale(calibrate(seg.zones), refs, thickness)
    if meters_per_pixel and meters_per_pixel > 0:
        scale["detected_meters_per_pixel"] = scale.get("meters_per_pixel")
        scale["meters_per_pixel"] = float(meters_per_pixel)
        scale["source"] = "user"
        scale["estimated"] = False
    elif state.get("manual_extent_m") and seg.footprint:
        w_m, d_m = state["manual_extent_m"]
        xs = [p[0] for p in seg.footprint]
        ys = [p[1] for p in seg.footprint]
        scale["meters_per_pixel"] = min(w_m / max(max(xs) - min(xs), 1), d_m / max(max(ys) - min(ys), 1))
        scale["source"] = "overall size"
        scale["estimated"] = False
        scale_confirmed = True
    scale["confirmed"] = bool(scale_confirmed)
    mpp = scale["meters_per_pixel"]
    if mpp:
        scale["px_per_ft"] = round(0.3048 / mpp, 2)

    _check_zone_sizes(seg.zones, mpp)
    validation = validate(seg, walls, openings, thickness, scale, labels)
    _room_confidence(seg, validation)

    payload: dict = {
        **{k: v for k, v in state.items() if k != "manual_extent_m"},
        "rooms_px": seg.rooms,
        "zones_px": seg.zones,
        "adjacency": seg.adjacency,
        "footprint_px": seg.footprint,
        "scale": scale,
        "validation": validation,
        "warnings": list(dict.fromkeys((extra_warnings or []) + seg.warnings + scale["warnings"] + validation["warnings"])),
    }
    payload.update(
        geometry_from_plan(
            seg.rooms, seg.zones, state["walls_px"], state.get("openings_px") or [], seg.footprint, mpp or 0.0,
            height=wall_height, default_thickness=wall_thickness, fixtures_px=state.get("fixtures_px"),
        )
    )
    if payload.get("analysis") and mpp:
        payload["analysis"] = {**payload["analysis"], "scale": {"pixels_per_meter": round(1 / mpp, 2), "confidence": scale["confidence"], "source": scale["source"]}}
    payload["scale_confidence"] = 1.0 if scale_confirmed else scale["confidence"]
    payload["meters_per_pixel"] = mpp

    if debug_images is not None and image is not None:
        debug_images["30_sealed"] = (seg.sealed.astype(np.uint8) * 255)
        debug_images["31_rooms"] = _draw_rooms(image, seg, walls, openings)
        debug_images["40_final"] = _draw_final(image, seg, walls, openings, scale)
    return payload


def validate(seg, walls: list[WallSegment], openings: list[Opening], thickness: float, scale: dict, labels: list[RoomLabel] | None = None) -> dict:
    """Blocking errors stop 3D generation; warnings are shown but do not block."""
    errors = list(seg.errors)
    warnings: list[str] = []
    # A "wall" many times the drawing's stroke width is a filled area (grey room fill, furniture
    # block) read as ink: extruded, it is a solid slab over the rooms.
    for w in walls:
        if w.thickness > max(4 * thickness, 25) and w.length > w.thickness:
            errors.append(f"A {w.thickness:.0f} px thick wall at ({w.start[0]:.0f}, {w.start[1]:.0f}) is a filled area, not a wall.")
    # Every room named on the plan must be an enclosed space of its own (or a zone of an open plan).
    named = {z.get("label_index") for z in seg.zones}
    for i, label in enumerate(labels or []):
        if i not in named and not any(Polygon(z["polygon"]).buffer(thickness).contains(Point(label.x, label.y)) for z in seg.zones if len(z["polygon"]) >= 3):
            errors.append(f"'{label.text}' is labelled on the plan but no enclosed space was found for it.")
    if not seg.footprint:
        errors.append("Exterior footprint is not closed. Add the missing exterior wall or mark the opening.")
    if not seg.rooms:
        errors.append("No rooms were found.")
    for room in seg.rooms:
        for problem in room["invalid"]:
            errors.append(f"{room['name'] or room['id']}: {problem}.")
        if room["uncertain"]:
            warnings.append(f"{room['name']}: may combine two rooms; check for a missing wall.")
    # Scale never blocks: an estimated one is labelled approximate and can be corrected afterwards.
    if scale.get("estimated") and not scale.get("confirmed"):
        warnings.append("Dimensions automatically estimated from your floor plan. Measurements are approximate.")
    # Unexplained crossings: two wall bodies crossing away from their ends (a junction is fine).
    for i, a in enumerate(walls):
        for b in walls[i + 1 :]:
            if a.orientation == b.orientation and a.orientation != "d":
                continue
            if _crosses_interior(a, b, thickness):
                warnings.append(f"Walls cross without a junction near ({a.start[0]:.0f}, {a.start[1]:.0f}).")
    # Doors must sit in a wall line: both jambs touch a wall, or the 3D cut leaves floating pieces.
    for o in openings:
        if o.kind == "opening":
            continue
        if not (_touches_wall(o.start, walls, thickness) and _touches_wall(o.end, walls, thickness)):
            warnings.append(f"{o.kind.capitalize()} at ({o.start[0]:.0f}, {o.start[1]:.0f}) is not attached to walls on both sides.")
    for z in seg.zones:
        if z.get("size_check"):
            warnings.append(z["size_check"])
    return {"ok": not errors, "errors": list(dict.fromkeys(errors)), "warnings": list(dict.fromkeys(warnings))}


def _check_zone_sizes(zones: list[dict], mpp: float | None) -> None:
    """Compare printed dimensions with the measured polygon; report clear disagreements."""
    for z in zones:
        z["measured_m"] = None
        m = z.get("measure_px") or {}
        if not mpp or not m.get("width"):
            continue
        w, d = m["width"] * mpp, m["depth"] * mpp
        z["measured_m"] = [round(w, 2), round(d, 2)]
        if not z.get("dims_m"):
            continue
        a, b = z["dims_m"]
        err = min(abs(a - w) / a + abs(b - d) / b, abs(b - w) / b + abs(a - d) / a) / 2
        z["dims_error"] = round(err, 3)
        if err > 0.12:
            tol = "(boundary inferred)" if z.get("boundary") == "inferred" else ""
            z["size_check"] = f"{z['name']}: printed {z.get('dims_text')} but measures {w:.2f} × {d:.2f} m {tol}".strip()


def _room_confidence(seg, validation: dict) -> None:
    for room in seg.rooms:
        zones = [z for z in seg.zones if z["room_id"] == room["id"]]
        conf = 0.55
        if any(z.get("name") not in ("Room", "Closet") or z.get("dims_m") for z in zones):
            conf += 0.2
        errs = [z.get("dims_error") for z in zones if z.get("dims_error") is not None]
        if errs:
            conf += 0.15 if max(errs) < 0.08 else (-0.15 if max(errs) > 0.12 else 0.0)
        if room["uncertain"]:
            conf -= 0.25
        if room["invalid"]:
            conf -= 0.4
        if room["kind"] == "open_plan":
            conf -= 0.05
        room["confidence"] = round(max(0.05, min(0.95, conf)), 2)
        for z in zones:
            z["confidence"] = round(max(0.05, room["confidence"] - (0.2 if z["boundary"] == "inferred" else 0.0)), 2)


def _crosses_interior(a: WallSegment, b: WallSegment, thickness: float) -> bool:
    from shapely.geometry import LineString

    la, lb = LineString([a.start, a.end]), LineString([b.start, b.end])
    if not la.intersects(lb):
        return False
    p = la.intersection(lb)
    if p.geom_type != "Point":
        return False
    near_end = min(np.hypot(p.x - q[0], p.y - q[1]) for q in (a.start, a.end, b.start, b.end))
    # H×V crossings are ordinary cross junctions; a diagonal cutting through a wall body is not.
    return near_end > thickness * 1.5 and "d" in (a.orientation, b.orientation)


def _touches_wall(point, walls: list[WallSegment], thickness: float) -> bool:
    from shapely.geometry import LineString, Point

    p = Point(point)
    return any(LineString([w.start, w.end]).distance(p) <= w.thickness / 2 + thickness for w in walls)


# ── (de)serialisation of the editable state ──────────────────────────────────────────────


def _wall(w: dict) -> WallSegment:
    (x0, y0), (x1, y1) = w["start"], w["end"]
    orientation = w.get("orientation") or ("h" if abs(y1 - y0) < 0.5 else "v" if abs(x1 - x0) < 0.5 else "d")
    return WallSegment((float(x0), float(y0)), (float(x1), float(y1)), float(w.get("thickness") or 8), orientation, float(w.get("confidence", 1.0)))


def _opening(o: dict) -> Opening:
    return Opening(tuple(o["start"]), tuple(o["end"]), float(o.get("thickness") or 8), o.get("kind") or "door", o.get("evidence") or "user", float(o.get("confidence", 1.0)))


def _label(l: dict, drawing_unit: str | None = None) -> RoomLabel:
    # A size typed in the correction view arrives as text only.
    if l.get("dims_text") and not l.get("dims_m"):
        parsed = parse_dimension(l["dims_text"], drawing_unit)
        l = {**l, "dims_m": [parsed.a, parsed.b] if parsed else None, "dims_uncertain": bool(parsed and parsed.uncertain)}
    return RoomLabel(
        name=l["name"], kind=l.get("kind") or "room", x=float(l["x"]), y=float(l["y"]), box=tuple(l.get("box") or (l["x"], l["y"], 1, 1)),
        text=l.get("text") or l["name"], dims_m=tuple(l["dims_m"]) if l.get("dims_m") else None, dims_text=l.get("dims_text"),
        dims_uncertain=bool(l.get("dims_uncertain")),
    )


def _label_dict(l: RoomLabel) -> dict:
    d = asdict(l)
    d["box"] = [round(v, 1) for v in d["box"]]
    d["x"], d["y"] = round(d["x"], 1), round(d["y"], 1)
    return d


def _median_thickness(walls: list[dict]) -> float:
    values = [float(w.get("thickness") or 0) for w in walls if w.get("thickness")]
    return float(np.median(values)) if values else 8.0


def _encode_mask(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (mask > 0).astype(np.uint8) * 255)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _decode_mask(data: str | None, shape) -> np.ndarray | None:
    if not data:
        return None
    try:
        arr = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    return arr > 0 if arr is not None and arr.shape == tuple(shape) else None


# ── images ──────────────────────────────────────────────────────────────────────────────


def _decode_image(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))[:, :, ::-1].copy()
    return image


def _working_copy(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale only very large scans. Returned scale maps working pixels back to upload pixels."""
    h, w = image.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return image, 1.0
    s = MAX_SIDE / max(h, w)
    return cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), 1 / s


def _draw_ocr(image, boxes, labels) -> np.ndarray:
    vis = image.copy()
    for b in boxes:
        cv2.rectangle(vis, (int(b.x), int(b.y)), (int(b.x + b.width), int(b.y + b.height)), (40, 40, 220), 1)
        cv2.putText(vis, b.text[:24], (int(b.x), max(10, int(b.y) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 220), 1)
    for l in labels:
        cv2.circle(vis, (int(l.x), int(l.y)), 4, (0, 160, 0), -1)
    return vis


def _draw_walls(image, walls: list[WallSegment], openings: list[Opening]) -> np.ndarray:
    vis = image.copy()
    for w in walls:
        color = (0, 0, 230) if w.orientation == "d" else ((0, 140, 255) if w.confidence < 1 else (200, 40, 0))
        cv2.line(vis, _pt(w.start), _pt(w.end), color, 2)
    colors = {"door": (0, 170, 0), "window": (220, 0, 220), "opening": (0, 200, 255)}
    for o in openings:
        cv2.line(vis, _pt(o.start), _pt(o.end), colors[o.kind], 4)
    return vis


def _draw_rooms(image, seg, walls, openings) -> np.ndarray:
    overlay = image.copy()
    rng = np.random.default_rng(7)
    for room in seg.rooms:
        color = tuple(int(c) for c in rng.integers(70, 230, 3))
        cv2.fillPoly(overlay, [np.array(room["polygon"], np.int32)], color)
    vis = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
    for z in seg.zones:
        color = (0, 0, 0) if z["boundary"] == "wall" else (0, 0, 220)
        cv2.polylines(vis, [np.array(z["polygon"], np.int32)], True, color, 2)
    return vis


def _draw_final(image, seg, walls, openings, scale) -> np.ndarray:
    """Visual regression output: original drawing with final walls, openings, rooms and names."""
    vis = _draw_rooms(image, seg, walls, openings)
    for w in walls:
        cv2.line(vis, _pt(w.start), _pt(w.end), (20, 20, 20), max(2, int(w.thickness * 0.5)))
    colors = {"door": (0, 170, 0), "window": (220, 0, 220), "opening": (0, 200, 255)}
    for o in openings:
        cv2.line(vis, _pt(o.start), _pt(o.end), colors[o.kind], 3)
    for z in seg.zones:
        x, y = (int(v) for v in z["label_point"])
        text = z["name"]
        if z.get("measured_m"):
            text += f" {z['measured_m'][0]:.1f}x{z['measured_m'][1]:.1f}m"
        cv2.putText(vis, text, (x - 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 3)
        cv2.putText(vis, text, (x - 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
    if scale.get("px_per_ft"):
        cv2.putText(vis, f"{scale['px_per_ft']} px/ft  conf {scale['confidence']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return vis


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _write_debug_dir(images: dict[str, np.ndarray], key: str) -> None:
    """storage/plan-debug/<image hash>/: one folder per upload, so a bad conversion can be inspected later."""
    try:
        from app.core.config import get_settings

        path = get_settings().storage_dir / "plan-debug" / key
        path.mkdir(parents=True, exist_ok=True)
        for name, img in images.items():
            cv2.imwrite(str(path / f"{name}.png"), img)
    except Exception:
        pass


def _png_data_url(img: np.ndarray, max_side: int | None = None) -> str:
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
