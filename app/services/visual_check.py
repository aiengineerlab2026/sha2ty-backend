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
DE_MAX = 22.0  # the photo's lighting and exposure stay in its pixels; a recoloured part is far beyond this


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
        if not q or comp.get("kind") in ("back_panel", "base_panel", "side_panel", "top_panel") or comp["id"] not in index:
            continue
        in_photo = photo_mask(q)
        if comp.get("kind") == "door_frame":  # a frame is its border: what fills its opening (the mirror) is not it
            for other in doc.get("components") or []:
                o = spec_by_id.get(other.get("analysisId") or "")
                if o is not None and other.get("assembly") == comp.get("assembly") and other.get("kind") in ("mirror", "glass"):
                    in_photo &= ~photo_mask(quad_of(o, details.get("bbox")))
        in_render = ids == index[comp["id"]]
        union = (in_photo | in_render).sum()
        iou = float((in_photo & in_render).sum() / union) if union else 0.0
        entry: dict[str, Any] = {"component": comp["id"], "iou": round(iou, 3)}
        if comp.get("kind") in ("mirror", "glass"):
            entry["mirrorIoU"] = round(float((in_photo & mirror).sum() / max((in_photo | mirror).sum(), 1)), 3)
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
    movers = len([a for a in (doc.get("animation") or {}).get("animations") or [] if a.get("enabled")])
    fronts = len([c for c in analysis.get("components") or [] if c.get("kind") in FRONT_KINDS])
    if fronts and movers != fronts:
        issues.append(f"The photo shows {fronts} doors / drawers; the model moves {movers}.")
    return {"checked": True, "view": "front (straightened photo vs orthographic render)", "parts": parts, "issues": issues,
            "meanIoU": round(float(np.mean([p["iou"] for p in parts])), 3) if parts else None, "_images": (flat, img)}
