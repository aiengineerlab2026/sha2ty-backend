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
            if comp.get("kind") in ("plinth", "crown_trim"):  # the same trim core as its sample (materials.trim_core)
                ys = np.nonzero(in_photo.any(1))[0]
                if len(ys) >= 3:
                    cut = ys[0] + int(np.ceil(0.6 * (ys[-1] - ys[0] + 1))) if comp["kind"] == "plinth" else ys[0] + int(0.4 * (ys[-1] - ys[0] + 1))
                    clean_photo[cut:] = False if comp["kind"] == "plinth" else clean_photo[cut:]
                    if comp["kind"] == "crown_trim":
                        clean_photo[:cut] = False
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
    design = check_design(doc, analysis, details, size_m, ids, part_ids, photo_mask)
    issues += design["issues"]
    try:
        silhouette = reference_silhouette(glb, analysis, details, size_m, rgb.shape[:2])
    except Exception as exc:  # a metric, never a failure of the check
        silhouette = {"judged": False, "reason": str(exc)[:120]}
    movers = len([a for a in (doc.get("animation") or {}).get("animations") or [] if a.get("enabled")])
    fronts = len([c for c in analysis.get("components") or [] if c.get("kind") in FRONT_KINDS])
    if fronts and movers != fronts:
        issues.append(f"The photo shows {fronts} doors / drawers; the model moves {movers}.")
    return {"checked": True, "view": "front (straightened photo vs orthographic render)", "parts": parts, "issues": issues,
            "meanIoU": round(float(np.mean([p["iou"] for p in parts if "iou" in p])), 3) if any("iou" in p for p in parts) else None, "hardware": hardware["parts"],
            "design": design["metrics"], "corrections": design["corrections"], "silhouette": silhouette, "_images": (flat, img)}


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


EDGE_TOL = 0.03  # a front's edge this far (share of the height / width) from where the photo puts it is a mismatch
UNEXPECTED = 0.3  # a part with no photo evidence covering this share of its front where the photo shows other parts


def check_design(doc: dict[str, Any], analysis: dict[str, Any], details: dict[str, Any], size_m: dict[str, Any], ids: np.ndarray,
                 part_ids: list[str], photo_mask) -> dict[str, Any]:
    """The whole visible design against the straightened photo, in both directions, by normalized edges and relative
    sizes (robust to a low-resolution photo, where pixel overlap is not):
      fronts      each door / drawer: top and bottom edge, height and width as shares of the front, against its corners
      unexpected  a part the photo gives no evidence for (no corners of its own) whose front covers what the photo
                  shows as other parts: geometry the reconstruction introduced (an invented crown over the doors)
      plinth      its height against the photo's
      mirrors     each pane's share of its door and the frame border, against the photo's
      materials   parts of one finish (material group) in one colour; a neutral reading built tinted
    Only measurements with evidence (the photo's own corners) are judged. `corrections`: structured, deterministic
    adjustments the assembly can apply (hybrid.build_corrected), never a regeneration."""
    from app.services.assembly import FRONT_KINDS, front_quad, front_rect, homography, valid_quad

    W, H = float(size_m["width"]), float(size_m["height"])
    quad = front_quad(analysis, details)
    Hm = homography(quad) if quad else None
    spec_by_id = {c["id"]: c for c in analysis.get("components") or []}
    comps = doc.get("components") or []
    metrics: dict[str, Any] = {"fronts": [], "unexpected": [], "mirrors": []}
    issues: list[str] = []
    corrections: list[dict[str, Any]] = []

    def photo_rect(spec):
        return front_rect(spec, Hm, details.get("bbox")) if spec and Hm is not None and valid_quad(spec.get("quad")) else None

    def model_rect(c):  # the part's front as [u0, v0, u1, v1] shares of the front (v down)
        (x0, y0, _), (x1, y1, _) = c["box"]
        return [(x0 + W / 2) / W, (H - y1) / H, (x1 + W / 2) / W, (H - y0) / H]

    fronts = [c for c in comps if c.get("kind") in FRONT_KINDS | {"door_frame"}]
    for c in fronts:
        spec = spec_by_id.get(c.get("analysisId") or "")
        pr = photo_rect(spec)
        if pr is None:
            continue
        mr = model_rect(c)
        m = {"component": c["id"], "top": round(mr[1] - pr[1], 3), "bottom": round(mr[3] - pr[3], 3),
             "heightRatio": round((mr[3] - mr[1]) / max(pr[3] - pr[1], 1e-6), 3), "widthRatio": round((mr[2] - mr[0]) / max(pr[2] - pr[0], 1e-6), 3)}
        metrics["fronts"].append(m)
        if abs(m["top"]) > EDGE_TOL or abs(m["bottom"]) > EDGE_TOL:
            issues.append(f"{c['name']}: {'shorter' if m['heightRatio'] < 1 else 'taller'} than the photo shows (top {m['top'] * H * 100:+.0f} cm, bottom {m['bottom'] * H * 100:+.0f} cm).")
            corrections.append({"op": "front_extent", "component": spec["id"], "rect": [round(v, 4) for v in pr]})
    tops = [m["top"] for m in metrics["fronts"]]
    metrics["topsAligned"] = bool(not tops or max(tops) - min(tops) <= 0.01)

    # Geometry the photo gives no evidence for, lying over what the photo shows as other parts.
    shown = {}
    for c in comps:
        spec = spec_by_id.get(c.get("analysisId") or c.get("evidenceId") or "")
        if spec is not None and valid_quad(spec.get("quad")) and c.get("kind") not in ("side_panel", "back_panel", "top_panel", "base_panel"):
            shown[c["id"]] = photo_mask(spec["quad"])
    index = {pid: i for i, pid in enumerate(part_ids)}
    for c in comps:
        if c["id"] in shown or c["id"] not in index or c.get("hardware") or c.get("kind") in ("side_panel", "back_panel", "top_panel", "base_panel"):
            continue
        mine = ids == index[c["id"]]
        if mine.sum() < 20:
            continue
        others = np.zeros_like(mine)
        for oid, m in shown.items():
            if oid != c["id"]:
                others |= m
        share = float((mine & others).sum() / mine.sum())
        if share >= UNEXPECTED:
            metrics["unexpected"].append({"component": c["id"], "kind": c.get("kind"), "coversShown": round(share, 2)})
            issues.append(f"{c['name']}: the photo does not show it; it covers {share:.0%} of its area where the photo shows other parts.")
            corrections.append({"op": "remove", "component": c["id"], "kind": c.get("kind")})

    plinth = next((c for c in comps if c.get("kind") == "plinth"), None)
    pspec = spec_by_id.get((plinth or {}).get("analysisId") or "")
    pr = photo_rect(pspec)
    if plinth and pr:
        model_h, photo_h = (plinth["box"][1][1] - plinth["box"][0][1]), (pr[3] - pr[1]) * H
        metrics["plinth"] = {"modelM": round(model_h, 3), "photoM": round(photo_h, 3)}
        if abs(model_h - photo_h) > max(0.015, 0.4 * photo_h):
            issues.append(f"{plinth['name']}: {model_h * 100:.0f} cm high, the photo shows about {photo_h * 100:.0f} cm.")
            corrections.append({"op": "plinth_height", "component": pspec["id"], "height": round(pr[3] - pr[1], 4)})

    for pane in (c for c in comps if c.get("kind") in ("mirror", "glass")):
        door = next((c for c in comps if c.get("assembly") == pane.get("assembly") and c.get("kind") == "door_frame"), None)
        ps, ds = spec_by_id.get(pane.get("analysisId") or ""), spec_by_id.get((door or {}).get("analysisId") or "")
        ppr, dpr = photo_rect(ps), photo_rect(ds)
        if not door or not ppr or not dpr:
            continue
        pm, dm = model_rect(pane), model_rect(door)
        area = lambda r: max(r[2] - r[0], 0) * max(r[3] - r[1], 0)  # noqa: E731
        border = lambda p, d: min(p[0] - d[0], d[2] - p[2]) * W  # noqa: E731  the side borders, metres
        m = {"component": pane["id"], "areaShare": round(area(pm) / max(area(dm), 1e-9), 3), "photoAreaShare": round(area(ppr) / max(area(dpr), 1e-9), 3),
             "borderM": round(border(pm, dm), 3), "photoBorderM": round(max(border(ppr, dpr), 0), 3)}
        metrics["mirrors"].append(m)
        if m["borderM"] > m["photoBorderM"] + 0.015 and m["photoAreaShare"] - m["areaShare"] > 0.03:
            issues.append(f"{door['name']}: its border is {m['borderM'] * 100:.1f} cm, the photo shows about {m['photoBorderM'] * 100:.1f} cm.")
            corrections.append({"op": "frame_border", "component": ds["id"], "border": round(max(m["photoBorderM"], 0.0), 4)})

    groups: dict[str, list[str]] = {}
    for c in comps:
        g = (c.get("material") or {}).get("group")
        if g and g != c["id"]:
            groups.setdefault(g, [g]).append(c["id"])
    by = {c["id"]: c for c in comps}
    metrics["materialGroups"] = {g: sorted(set(m)) for g, m in groups.items()}
    for g, members in groups.items():
        cols = {(by.get(m) or {}).get("material", {}).get("color") for m in members if m in by}
        if len(cols) > 1:
            issues.append(f"Parts of one finish ({', '.join(by[m]['name'] for m in members if m in by)}) have different colours.")
    return {"metrics": metrics, "issues": issues, "corrections": corrections}


def camera_of(analysis: dict[str, Any], size_m: dict[str, Any]) -> dict[str, float]:
    """The photo's camera, estimated from its corners: yaw from how wide the side face looks against the front (its
    real depth against the real width), else the reading's viewYaw; pitch from the front's keystone (its top edge
    against its bottom edge). Degrees; used to render the model as the photo saw it, never to change the model."""
    W, D = float(size_m["width"]), float(size_m["depth"])
    q = analysis.get("frontQuad")
    yaw = float(analysis.get("viewYaw") or 0.0)
    side = next((c.get("sideQuad") for c in analysis.get("components") or [] if c.get("sideQuad")), None)
    pitch = 5.0
    if q:
        q = np.asarray(q, float)
        front_w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        if side:
            s_ = np.asarray(side, float)
            side_w = (np.linalg.norm(s_[1] - s_[0]) + np.linalg.norm(s_[2] - s_[3])) / 2
            est = float(np.degrees(np.arctan2(side_w * W, max(front_w * D, 1e-6))))
            yaw = est if (s_[:, 0].mean() > q[:, 0].mean()) else -est  # the right side seen: positive
        top, bottom = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
        pitch = float(np.clip(5.0 + 60.0 * (bottom - top) / max(top + bottom, 1e-6), -20, 30))
    return {"yaw": round(float(np.clip(yaw, -70, 70)), 1), "pitch": round(pitch, 1)}


def reference_silhouette(glb: bytes, analysis: dict[str, Any], details: dict[str, Any], size_m: dict[str, Any], shape) -> dict[str, Any]:
    """The model's outline rendered from the photo's estimated camera against the photo's (its front and side corners),
    both cropped to their own box and compared at one size (IoU). Judged only when the photo gives a side face or is
    nearly straight on (else its outline is the front alone and the comparison would mislead)."""
    from app.services import render

    cam = camera_of(analysis, size_m)
    q = analysis.get("frontQuad")
    side = [c["sideQuad"] for c in analysis.get("components") or [] if c.get("sideQuad")]
    if not q or (not side and abs(cam["yaw"]) > 5):
        return {"judged": False, "camera": cam, "reason": "the photo shows no side face to outline"}
    ih, iw = shape
    pts = np.array([[p[0] * iw, p[1] * ih] for quad in [q, *side] for p in quad], np.float32)
    photo = np.zeros((ih, iw), np.uint8)
    cv2.fillConvexPoly(photo, cv2.convexHull(pts).astype(np.int32), 255)
    _, ids, _, _ = render.render(glb, cam["yaw"], cam["pitch"], size=240, shaded=False)

    def crop(m):
        ys, xs = np.nonzero(m)
        c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
        return cv2.resize(c, (120, 120), interpolation=cv2.INTER_NEAREST) > 0, (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)

    a, ra = crop(photo > 0)
    b, rb = crop(ids >= 0)
    iou = float((a & b).sum() / max((a | b).sum(), 1))
    return {"judged": True, "camera": cam, "iou": round(iou, 3), "aspectPhoto": round(float(ra), 3), "aspectModel": round(float(rb), 3)}


def views(glb: bytes, analysis: dict[str, Any], size_m: dict[str, Any], size: int = 320):
    """The comparison renders (PIL images): straight orthographic front, the photo's estimated camera, a three-quarter
    view and the side."""
    from PIL import Image

    from app.services import render

    W, H = float(size_m["width"]), float(size_m["height"])
    span = max(W, H) / 2
    cam = camera_of(analysis, size_m)
    front = render.render(glb, ortho=(-span, H / 2 - span, span, H / 2 + span), size=size)[0]
    return {"front": Image.fromarray(front), "reference": Image.fromarray(render.render(glb, cam["yaw"], cam["pitch"], size)[0]),
            "threeQuarter": Image.fromarray(render.render(glb, 35, 10, size)[0]), "side": Image.fromarray(render.render(glb, 90, 5, size)[0])}
