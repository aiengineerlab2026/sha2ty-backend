"""The finished model compared with the photo it was rebuilt from, part by part (deterministic, no AI call).

The photo's front face is straightened by the reading's front quad (a homography) onto the model's front rectangle;
the model is rendered there orthographically, unshaded (services/render.py). Then, for every part the photo shows:
  outline   overlap (IoU) of the part in the render and its corners in the straightened photo: right size and place
  colour    ΔE between the part's rendered colour and its clean pixels in the photo (reflections, shadows, objects in
            front and smaller parts on it excluded)
  mirrors   the render's mirror region against the photo's mirror / glass corners (what they reflect is never compared)
  count     doors and drawers in the model against the reading
A mismatch is reported per part. It never makes a model look better than it is; it can only flag it.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np

SIZE = 360
IOU_MIN = 0.6
DE_MAX = 22.0
THIN_PX = 6.0  # a part narrower than this in the photo is not judged (blur and anti-aliasing are its whole width)  # the photo's lighting and exposure stay in its pixels; a recoloured part is far beyond this


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)


def compare(glb: bytes, photo: bytes, doc: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    from app.services import render
    from app.services.assembly import FRONT_KINDS, front_quad
    from app.services.materials import quad_of

    analysis, details = doc.get("analysis") or {}, doc.get("specDetails") or {}
    size_m = doc.get("sizeM") or {}
    quad = front_quad(analysis, details)
    if not quad or not size_m:
        return {"checked": False, "reason": "No front face to compare with (the reading has no front corners)."}
    W, H = float(size_m["width"]), float(size_m["height"])
    sw, sh = (SIZE, max(8, int(SIZE * H / W))) if W >= H else (max(8, int(SIZE * W / H)), SIZE)
    rgb = np.asarray(Image.open(io.BytesIO(photo)).convert("RGB"))
    ih, iw = rgb.shape[:2]
    src = np.array([[p[0] * iw, p[1] * ih] for p in quad], np.float32)
    M = cv2.getPerspectiveTransform(src, np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], np.float32))
    flat = cv2.warpPerspective(rgb, M, (sw, sh))
    # the model's front, orthographic over exactly the front rectangle, then the same aspect as the straightened photo
    n = max(sw, sh)
    img, ids, mirror, part_ids = render.render(glb, ortho=(-W / 2, 0, W / 2, H) if W >= H else (-W / 2, 0, W / 2, H), size=n, shaded=False)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_NEAREST)
    ids = cv2.resize(ids.astype(np.int32), (sw, sh), interpolation=cv2.INTER_NEAREST)
    mirror = cv2.resize(mirror.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0

    def photo_mask(q):
        m = np.zeros((ih, iw), np.uint8)
        cv2.fillPoly(m, [np.array([[p[0] * iw, p[1] * ih] for p in q], np.int32)], 255)
        return cv2.warpPerspective(m, M, (sw, sh), flags=cv2.INTER_NEAREST) > 0

    excluded = np.zeros((sh, sw), bool)
    for e in analysis.get("exclusions") or []:
        excluded |= photo_mask(e["quad"])
    spec_by_id = {c["id"]: c for c in analysis.get("components") or []}
    index = {pid: i for i, pid in enumerate(part_ids)}
    parts, issues = [], []
    for comp in doc.get("components") or []:
        spec = spec_by_id.get(comp.get("analysisId") or "")
        q = quad_of(spec, details.get("bbox")) if spec else None
        if not q or comp.get("kind") in ("back_panel", "base_panel", "side_panel", "top_panel") or comp["id"] not in index or comp.get("hardware"):
            continue  # hardware is compared on its own door (check_hardware): an outline overlap means nothing for a bar
        in_photo = photo_mask(q)
        if comp.get("kind") == "door_frame":  # a frame is its border: what fills its opening (the mirror) is not it
            for other in doc.get("components") or []:
                o = spec_by_id.get(other.get("analysisId") or "")
                if o is not None and other.get("assembly") == comp.get("assembly") and other.get("kind") in ("mirror", "glass"):
                    in_photo &= ~photo_mask(quad_of(o, details.get("bbox")))
        in_render = ids == index[comp["id"]]
        # A thin part (a frame's border, a trim) seen in only a few pixels' width cannot be judged by overlap or colour.
        ys, xs = np.nonzero(in_photo)
        span = (xs.max() - xs.min() + ys.max() - ys.min() + 2) if len(xs) else 1
        # area ÷ (width + height): a strip's thickness, a frame ring's border × 2; in the photo's own pixels
        width_px = in_photo.sum() / max(span, 1) * (rgb.shape[1] / sw) / (2 if comp.get("kind") == "door_frame" else 1)
        if width_px < THIN_PX:
            parts.append({"component": comp["id"], "judged": False, "reason": f"only ~{width_px:.0f} px wide in the photo"})
            continue
        union = (in_photo | in_render).sum()
        iou = float((in_photo & in_render).sum() / union) if union else 0.0
        entry: dict[str, Any] = {"component": comp["id"], "iou": round(iou, 3)}
        if comp.get("kind") in ("mirror", "glass"):
            own = in_render & mirror  # this pane's own mirror pixels (another pane is another part)
            entry["mirrorIoU"] = round(float((in_photo & own).sum() / max((in_photo | own).sum(), 1)), 3)
        else:
            clean_photo = in_photo & ~excluded
            for other in doc.get("components") or []:  # smaller parts lying on it (handles, panes) are not its colour
                o = spec_by_id.get(other.get("analysisId") or "")
                oq = quad_of(o, details.get("bbox")) if o else None
                if o is not None and o is not spec and oq:
                    om = photo_mask(oq)
                    if 0 < om.sum() < 0.6 * in_photo.sum():
                        clean_photo &= ~om
            shown = in_render & ~mirror
            if clean_photo.sum() > 50 and shown.sum() > 50:
                de = float(np.linalg.norm(np.median(_lab(img[shown]), axis=0) - np.median(_lab(flat[clean_photo]), axis=0)))
                entry["colourDE"] = round(de, 1)
                if de > DE_MAX:
                    issues.append(f"{comp['name']}: its colour differs from the photo (ΔE {de:.0f}).")
        if iou < IOU_MIN:
            issues.append(f"{comp['name']}: its outline does not match the photo (overlap {iou:.0%}).")
        parts.append(entry)
    hardware = check_hardware(doc, analysis, details, size_m)
    issues += hardware["issues"]
    movers = len([a for a in (doc.get("animation") or {}).get("animations") or [] if a.get("enabled")])
    fronts = len([c for c in analysis.get("components") or [] if c.get("kind") in FRONT_KINDS])
    if fronts and movers != fronts:
        issues.append(f"The photo shows {fronts} doors / drawers; the model moves {movers}.")
    return {"checked": True, "view": "front (straightened photo vs orthographic render)", "parts": parts, "issues": issues,
            "meanIoU": round(float(np.mean([p["iou"] for p in parts if "iou" in p])), 3) if any("iou" in p for p in parts) else None, "hardware": hardware["parts"], "_images": (flat, img)}


DETAIL_PX = 20  # fewer of the handle's own pixels than this in the photo: too little detail to call a mismatch


def check_hardware(doc: dict[str, Any], analysis: dict[str, Any], details: dict[str, Any], size_m: dict[str, Any]) -> dict[str, Any]:
    """Each handle against the photo, measured on its own door (outline overlap is meaningless for a 1.6 cm bar):
    centre error and length / width ratios as shares of the door, and orientation. A mismatch is only reported when
    the photo shows the handle in enough pixels to judge it."""
    from app.services.assembly import front_quad, front_rect, homography

    quad = front_quad(analysis, details)
    Hm = homography(quad) if quad else None
    W, H = float(size_m["width"]), float(size_m["height"])
    spec_by_id = {c["id"]: c for c in analysis.get("components") or []}
    comps = {c["id"]: c for c in doc.get("components") or []}
    out, issues = [], []
    for comp in doc.get("components") or []:
        hw, spec = comp.get("hardware"), spec_by_id.get(comp.get("analysisId") or "")
        host = comps.get(comp.get("host") or "")
        if not hw or not spec or not host or Hm is None or not spec.get("quad"):
            continue
        r = front_rect(spec, Hm, details.get("bbox"))
        photo = np.array([-W / 2 + (r[0] + r[2]) / 2 * W, H - (r[1] + r[3]) / 2 * H, (r[2] - r[0]) * W, (r[3] - r[1]) * H])
        lo, hi = np.array(comp["box"][0]), np.array(comp["box"][1])
        model = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[0] - lo[0], hi[1] - lo[1]])
        hlo, hhi = np.array(host["box"][0]), np.array(host["box"][1])
        door = np.maximum(hhi[:2] - hlo[:2], 1e-6)
        centre_error = float(np.linalg.norm((model[:2] - photo[:2]) / door))
        axis = 1 if hw.get("orientation") == "vertical" else 0
        length_ratio = float(model[2 + axis] / max(photo[2 + axis], 1e-6))
        photo_orientation = "vertical" if photo[3] >= photo[2] else "horizontal"
        detail = int((spec.get("measured") or {}).get("pixels") or 0)
        entry = {"component": comp["id"], "centreError": round(centre_error, 3), "lengthRatio": round(length_ratio, 2),
                 "orientation": hw.get("orientation"), "photoOrientation": photo_orientation if hw.get("type") == "bar" else "none", "photoPixels": detail,
                 "widthEstimated": bool((hw.get("estimated") or {}).get("thickness"))}
        out.append(entry)
        if detail < DETAIL_PX:
            entry["judged"] = False  # too few pixels to call a mismatch
            continue
        entry["judged"] = True
        if centre_error > 0.15:
            issues.append(f"{comp['name']}: placed away from where the photo shows it.")
        if hw.get("type") == "bar" and not 0.7 <= length_ratio <= 1.4:
            issues.append(f"{comp['name']}: its length differs from the photo ({length_ratio:.1f}×).")
        if hw.get("type") == "bar" and photo_orientation != hw.get("orientation"):
            issues.append(f"{comp['name']}: its orientation differs from the photo.")
    return {"parts": out, "issues": issues}
