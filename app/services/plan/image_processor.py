"""Floor-plan image → validated 2D plan (pixel space) → metric 3D geometry.

Pipeline (each stage writes a debug image):
  1. OCR on the original image (names, printed dimensions)       → 10_ocr
  2. Structural walls by thickness, H/V/diagonal segments          → 01..07, 20_walls
  3. Verified openings (door arcs, window lines, exterior gaps)    → 21_openings
  4. Free-space rooms + open-plan zones, topology checks           → 30_sealed, 31_rooms
  5. Scale from printed dimensions (robust median)                 → 40_final
The correction UI edits the pixel-space state (walls/openings/labels/zone lines/scale) and sends it
back to `plan_from_state`, which re-runs steps 4–5 on the edited walls, so preview and 3D agree.
"""

from __future__ import annotations

import base64
import io
from dataclasses import asdict

import cv2
import numpy as np
from PIL import Image

from app.services.plan.dimension_ocr import calibrate, measurements_from_boxes, parse_dimension, room_labels, run_ocr
from app.services.plan.geometry_builder import DEFAULT_HEIGHT, DEFAULT_THICKNESS, geometry_from_plan
from app.services.plan.room_segmenter import RoomLabel, segment
from app.services.plan.wall_detector import Opening, WallSegment, detect_walls

MAX_SIDE = 2400  # larger scans are downscaled for speed; `image_scale` maps back to the upload


def plan_from_image(
    image_bytes: bytes,
    wall_height: float = DEFAULT_HEIGHT,
    wall_thickness: float = DEFAULT_THICKNESS,
    width_m: float | None = None,
    depth_m: float | None = None,
    debug: bool = False,
) -> dict:
    original = _decode_image(image_bytes)
    image, image_scale = _working_copy(original)
    debug_images: dict[str, np.ndarray] = {"00_original": image}

    boxes, ocr_notes = run_ocr(image)
    labels, drawing_unit = room_labels(boxes)
    measurements, ocr_warn = measurements_from_boxes(boxes)
    debug_images["10_ocr"] = _draw_ocr(image, boxes, labels)

    text_boxes = [(b.x, b.y, b.width, b.height) for b in boxes]
    walls = detect_walls(image, text_boxes)
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
    }
    if width_m and depth_m and width_m > 0 and depth_m > 0:
        state["manual_extent_m"] = [width_m, depth_m]

    payload = plan_from_state(
        state,
        wall_height=wall_height,
        wall_thickness=wall_thickness,
        extra_warnings=ocr_notes + ocr_warn + walls.warnings,
        debug_images=debug_images,
        image=image,
    )
    payload["preview_image"] = _png_data_url(image)
    payload["ocr_measurements"] = [
        {"text": m.text, "meters": m.meters, "unit": m.unit, "pair": m.pair, "label": m.label, "uncertain": m.uncertain,
         "box": [round(m.x, 1), round(m.y, 1), round(m.width, 1), round(m.height, 1)]}
        for m in measurements
    ]
    payload["ocr_text"] = [{"text": b.text, "score": round(b.score, 2), "box": [round(b.x, 1), round(b.y, 1), round(b.width, 1), round(b.height, 1)]} for b in boxes]
    _write_debug_dir(debug_images)
    if debug:
        payload["debug_images"] = {k: _png_data_url(v, max_side=720) for k, v in debug_images.items() if k[:2] in ("00", "05", "10", "21", "30", "31", "40")}
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
    straight = _decode_mask(state.get("straight_lines"), shape)
    seg = segment(shape, walls, openings, thickness, labels, (straight, straight) if straight is not None else None, state.get("zone_lines"))

    scale = calibrate(seg.zones)
    if meters_per_pixel and meters_per_pixel > 0:
        scale["detected_meters_per_pixel"] = scale.get("meters_per_pixel")
        scale["meters_per_pixel"] = float(meters_per_pixel)
        scale["source"] = "user"
    elif state.get("manual_extent_m") and seg.footprint:
        w_m, d_m = state["manual_extent_m"]
        xs = [p[0] for p in seg.footprint]
        ys = [p[1] for p in seg.footprint]
        scale["meters_per_pixel"] = min(w_m / max(max(xs) - min(xs), 1), d_m / max(max(ys) - min(ys), 1))
        scale["source"] = "overall size"
        scale_confirmed = True
    else:
        scale["source"] = "printed dimensions" if scale["meters_per_pixel"] else "none"
    scale["confirmed"] = bool(scale_confirmed)
    mpp = scale["meters_per_pixel"]
    if mpp:
        scale["px_per_ft"] = round(0.3048 / mpp, 2)

    _check_zone_sizes(seg.zones, mpp)
    validation = validate(seg, walls, openings, thickness, scale)
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
            height=wall_height, default_thickness=wall_thickness,
        )
    )
    payload["scale_confidence"] = scale["confidence"] if scale["source"] == "printed dimensions" else (1.0 if scale_confirmed else 0.0)
    payload["meters_per_pixel"] = mpp

    if debug_images is not None and image is not None:
        debug_images["30_sealed"] = (seg.sealed.astype(np.uint8) * 255)
        debug_images["31_rooms"] = _draw_rooms(image, seg, walls, openings)
        debug_images["40_final"] = _draw_final(image, seg, walls, openings, scale)
    return payload


def validate(seg, walls: list[WallSegment], openings: list[Opening], thickness: float, scale: dict) -> dict:
    """Blocking errors stop 3D generation; warnings are shown but do not block."""
    errors = list(seg.errors)
    warnings: list[str] = []
    if not seg.footprint:
        errors.append("Exterior footprint is not closed. Add the missing exterior wall or mark the opening.")
    if not seg.rooms:
        errors.append("No rooms were found.")
    for room in seg.rooms:
        for problem in room["invalid"]:
            errors.append(f"{room['name'] or room['id']}: {problem}.")
        if room["uncertain"]:
            warnings.append(f"{room['name']}: may combine two rooms; check for a missing wall.")
    if not scale.get("meters_per_pixel"):
        errors.append("Scale is unknown. Enter the scale or a known room size, then confirm it.")
    elif not scale.get("confirmed") and (scale.get("confidence", 0) < 0.6 or not scale.get("uniform", True)):
        errors.append("Scale is not reliable enough. Check it against a printed dimension and confirm it.")
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


def _write_debug_dir(images: dict[str, np.ndarray]) -> None:
    try:
        from app.core.config import get_settings

        path = get_settings().storage_dir / "plan-debug"
        path.mkdir(parents=True, exist_ok=True)
        for old in path.glob("*.jpg"):
            old.unlink()  # earlier pipeline's stage images; names no longer match
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
