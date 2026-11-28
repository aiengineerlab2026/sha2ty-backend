"""A part's MATERIAL from the original photo, separated from its geometry: never the photographed part pasted on.

For one component:
  1. its surface    the reading's corners of its visible face (or a side's face), warped flat by a homography; every
                    mask (reflections, shadows, objects in front, the parts mounted on it, its moulding lines, the
                    furniture's silhouette) warped by the SAME homography
  2. clean pixels   inside the surface, outside every mask, without the brightest and darkest tenth (highlights, cast
                    shadows)
  3. colour         their median in Lab → the base colour
  4. texture        only for wood and patterned fabric whose clean pixels show a real directional grain / pattern: the
                    largest clean square, its lighting flattened, made tileable (mirrored), its grain kept in the
                    face's own directions (UVs are real-size: solids.face_uv)
  5. fallbacks      too few clean pixels → the reading's colour; mirrors and glass → fixed physical materials
Each result says where it came from ("texture sample", "colour sample", "reading colour", "physical") and how sure.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np

MIN_CLEAN = 0.1  # clean pixels below this share of the surface (a frame's border around its mirror is ~0.25)…
MIN_PIXELS = 300  # …or fewer than this many: the sample is not trusted
TEXTURE_PX = 256
WOOD = ("solid_wood", "wood_veneer")
PATTERNED = ("fabric",)
MIRROR = {"metallic": 1.0, "roughness": 0.02, "color": [0.92, 0.93, 0.94]}
GLASS = {"metallic": 0.0, "roughness": 0.05, "color": [0.85, 0.9, 0.92], "alpha": 0.3}


def _px(quad, w: int, h: int) -> np.ndarray:
    return np.array([[p[0] * w, p[1] * h] for p in quad], np.float32)


def _rect_size(q: np.ndarray) -> tuple[int, int]:
    width = int(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))
    height = int(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))
    return max(width, 2), max(height, 2)


def rectify(rgb: np.ndarray, mask: np.ndarray, quad) -> tuple[np.ndarray, np.ndarray]:
    """The face's pixels and its mask, both warped by the same homography onto an upright rectangle."""
    h, w = rgb.shape[:2]
    src = _px(quad, w, h)
    tw, th = _rect_size(src)
    dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    face = cv2.warpPerspective(rgb, M, (tw, th), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(mask, M, (tw, th), flags=cv2.INTER_NEAREST, borderValue=0)
    return face, valid


def quad_of(c: dict[str, Any], obj_box: list[float] | None) -> list[list[float]] | None:
    """The part's face corners in the image: the reading's, else its box in the object's box (as photographed, upright)."""
    if c.get("quad") or c.get("sideQuad"):
        return c.get("quad") or c.get("sideQuad")
    b, o = c.get("box"), obj_box or [0.0, 0.0, 1.0, 1.0]
    if not b:
        return None
    x0, y0 = o[0] + b[0] * (o[2] - o[0]), o[1] + b[1] * (o[3] - o[1])
    x1, y1 = o[0] + b[2] * (o[2] - o[0]), o[1] + b[3] * (o[3] - o[1])
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def masks(rgb: np.ndarray, component: dict[str, Any], analysis: dict[str, Any], silhouette: np.ndarray | None, obj_box: list[float] | None = None) -> np.ndarray:
    """255 where the photo shows this part's own material: inside the furniture, outside reflections, shadows,
    objects in front and the smaller parts lying on it."""
    h, w = rgb.shape[:2]
    # The part's own face corners already bound its surface: the silhouette (which can drop a dark trim the colour of
    # the floor) is only for a part read without corners.
    own = quad_of(component, obj_box)
    exact = component.get("quad") or component.get("sideQuad")
    keep = np.full((h, w), 255, np.uint8) if silhouette is None or exact else silhouette.copy()
    for e in analysis.get("exclusions") or []:
        cv2.fillPoly(keep, [_px(e["quad"], w, h).astype(np.int32)], 0)
    own_area = cv2.contourArea(_px(own, w, h)) if own else 0
    for other in analysis.get("components") or []:
        q = quad_of(other, obj_box)
        if other is component or not q or other.get("logical"):
            continue
        poly = _px(q, w, h)
        if 0 < cv2.contourArea(poly) < 0.6 * own_area:  # a handle, a pane, a smaller part lying on it
            cv2.fillPoly(keep, [poly.astype(np.int32)], 0)
    return keep


def _moulding_lines(valid: np.ndarray, mouldings: list[dict[str, Any]]) -> np.ndarray:
    """The moulding rings' bands (with their shading) taken out of the face's mask."""
    th, tw = valid.shape
    out = valid.copy()
    for m in mouldings or []:
        x0, y0, x1, y1 = m["rect"]
        b = max(2, int((m.get("border") or 0.04) * min(tw, th) * 1.6))
        X0, Y0, X1, Y1 = int(x0 * tw), int(y0 * th), int(x1 * tw), int(y1 * th)
        cv2.rectangle(out, (X0, Y0), (X1, Y1), 0, thickness=b)
    return out


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)


def _grain(face: np.ndarray, valid: np.ndarray) -> float:
    """How directional the surface's fine detail is (0 none … 1 one direction everywhere): wood grain, weave."""
    g = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 6)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = valid > 0
    if m.sum() < 400 or g[m].std() < 2.0:
        return 0.0
    jxx, jyy, jxy = (gx * gx)[m].mean(), (gy * gy)[m].mean(), (gx * gy)[m].mean()
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    return float(lam / max(jxx + jyy, 1e-6))


def _clean_square(valid: np.ndarray) -> tuple[int, int, int] | None:
    """The largest square of the face that is all clean (x, y, side), searched on a coarse grid."""
    th, tw = valid.shape
    integral = cv2.integral((valid > 0).astype(np.uint8))
    for side in range(min(th, tw), 24, -max(4, min(th, tw) // 20)):
        step = max(4, side // 6)
        for y in range(0, th - side + 1, step):
            for x in range(0, tw - side + 1, step):
                total = integral[y + side, x + side] - integral[y, x + side] - integral[y + side, x] + integral[y, x]
                if total >= 0.98 * side * side:
                    return x, y, side
    return None


def _tile(patch: np.ndarray, colour: np.ndarray) -> bytes:
    """The patch with its lighting flattened (its own large-scale shading removed) around `colour`, mirrored into a
    seamless tile."""
    from PIL import Image

    p = patch.astype(np.float32)
    shading = cv2.GaussianBlur(p, (0, 0), max(4, patch.shape[0] / 6))
    detail = p - shading
    flat = np.clip(colour * 255 + detail * 0.8, 0, 255).astype(np.uint8)
    flat = cv2.resize(flat, (TEXTURE_PX // 2, TEXTURE_PX // 2), interpolation=cv2.INTER_AREA)
    seamless = np.vstack([np.hstack([flat, flat[:, ::-1]]), np.hstack([flat[::-1], flat[::-1, ::-1]])])
    buf = io.BytesIO()
    Image.fromarray(seamless).save(buf, "JPEG", quality=88)
    return buf.getvalue()


def _hex(value: str | None) -> list[float] | None:
    v = (value or "").strip().lstrip("#")
    return [int(v[i:i + 2], 16) / 255 for i in (0, 2, 4)] if len(v) == 6 else None


NEUTRAL = 8.0  # a reading colour within this chroma (Lab a*b*, 8-bit scale ≈ CIE) is a neutral finish (grey, white, black)
TINT = 24.0  # a sample within this chroma can be a neutral surface tinted by the room's light


def trim_core(valid: np.ndarray, kind: str | None) -> np.ndarray:
    """A trim against the floor or the ceiling (a plinth, a crown) is sampled away from that edge: its corners in a
    photo are seldom exact to the pixel, and the floor or ceiling beside it is never its finish."""
    if kind not in ("plinth", "crown_trim") or valid.shape[0] < 3:
        return valid
    out, h = valid.copy(), valid.shape[0]
    if kind == "plinth":
        out[int(np.ceil(0.6 * h)):] = 0
    else:
        out[: int(0.4 * h)] = 0
    return out


def scene_cast(rgb: np.ndarray) -> np.ndarray:
    """The photo's overall colour cast (median a*, b*): the room's light. Used only to tell a tinted neutral surface
    from a coloured one, never applied to the photo as a global white balance."""
    small = cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_AREA) if max(rgb.shape[:2]) > 96 else rgb
    return np.median(_lab(small.reshape(-1, 3)), axis=0)[1:] - 128.0


def fuse(lab: np.ndarray, said: list[float] | None, cast: np.ndarray | None) -> tuple[np.ndarray, str | None]:
    """The measured colour (8-bit Lab) and the reading's colour together. When the reading says the finish is neutral,
    the sample is only slightly tinted and its tint leans the way the room's light does, the sample's tint is the
    light's: its lightness is kept (measured), its hue and chroma are the reading's. A coloured finish (the reading
    says so) keeps its measured colour: intentionally warm furniture stays warm."""
    if said is None or cast is None or np.hypot(*cast) < 3:
        return lab, None
    ref = _lab(np.uint8([[int(round(v * 255)) for v in said]]))[0]
    ra, rb, ma, mb = ref[1] - 128, ref[2] - 128, lab[1] - 128, lab[2] - 128
    if np.hypot(ra, rb) <= NEUTRAL and np.hypot(ma, mb) <= TINT and (ma - ra) * cast[0] + (mb - rb) * cast[1] > 0 and np.hypot(ma - ra, mb - rb) > 2:
        return np.array([lab[0], ref[1], ref[2]]), "a neutral finish under the room's coloured light: its hue from the reading, its lightness measured"
    return lab, None


def _dominant(lab: np.ndarray, said: list[float] | None) -> np.ndarray:
    """The surface's own pixels when two clearly different colours share its region (a thin trim and the floor under
    it, a frame and what shows through it): the cluster nearest the reading's colour, else the larger one."""
    if len(lab) < 12:
        return lab
    cv2.setRNGSeed(0)
    _, lbl, ctr = cv2.kmeans(lab.astype(np.float32), 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    shares = [(lbl == k).mean() for k in (0, 1)]
    if np.linalg.norm(ctr[0] - ctr[1]) < 25 or min(shares) < 0.2:
        return lab
    pick = int(np.argmax(shares))
    if said is not None:  # by hue: the reading's lightness is the least reliable part of its colour
        ref = _lab(np.uint8([[int(round(v * 255)) for v in said]]))[0]
        d = [np.hypot(*(c[1:] - ref[1:])) for c in ctr]
        if abs(d[0] - d[1]) > 5:
            pick = int(np.argmin(d))
    return lab[lbl == pick]


def extract(rgb: np.ndarray, component: dict[str, Any] | None, analysis: dict[str, Any], silhouette: np.ndarray | None, finish_rough: float,
            obj_box: list[float] | None = None, default: list[float] | None = None, cast: np.ndarray | None = None) -> dict[str, Any]:
    """{color, texture (JPEG bytes or None), metallic, roughness, alpha, source, confidence} for one component."""
    c = component or {}
    cls = str(c.get("materialClass") or "other")
    if c.get("kind") == "mirror" or cls == "mirror":
        return {**MIRROR, "texture": None, "alpha": 1.0, "source": "physical", "confidence": 1.0, "class": "mirror"}
    if c.get("kind") == "glass" or cls == "glass":
        return {**GLASS, "texture": None, "source": "physical", "confidence": 1.0, "class": "glass"}
    metallic = 1.0 if cls == "metal" else 0.0
    # No usable pixels: the reading's colour, else the caller's context colour (the furniture's dominant finish), else
    # a neutral mid grey (never an invented brown).
    fallback = {"color": _hex(c.get("color")) or default or [0.6, 0.6, 0.6], "texture": None, "metallic": metallic, "roughness": finish_rough,
                "alpha": 1.0, "source": "reading colour", "confidence": 0.4 if c.get("color") else 0.1, "class": cls}
    quad = quad_of(c, obj_box)
    if not quad:
        return fallback
    face, valid = rectify(rgb, masks(rgb, c, analysis, silhouette, obj_box), quad)
    valid = trim_core(_moulding_lines(valid, c.get("mouldings") or []), c.get("kind"))
    clean = valid > 0
    share = float(clean.mean())
    small = c.get("kind") in ("handle", "knob") or str(c.get("geometry")) in ("hardware",)
    need_px = 6 if small else MIN_PIXELS  # hardware is small in any photo: a handful of its own pixels is a sample
    if share < MIN_CLEAN:
        return {**fallback, "reason": f"only {share:.0%} of its surface is free of reflections, shadows and other parts"}
    if clean.sum() < need_px:
        return {**fallback, "reason": f"too small in the photo ({int(clean.sum())} pixels)"}
    lab = _lab(face[clean])
    said = _hex(c.get("color"))
    if cls not in WOOD + PATTERNED and not small:
        lab = _dominant(lab, said)
    lo, hi = np.percentile(lab[:, 0], [10, 90])
    keep = (lab[:, 0] >= lo) & (lab[:, 0] <= hi)
    kept = lab[keep] if keep.sum() > 30 else lab
    # The surface's own colour, not the room's lighting on it: hue from the median, lightness from the well-lit
    # pixels (the upper quartile of the trimmed range: a shaded half of a door is not its paint).
    # Metal: the median (its highlights are reflections, not its colour); other surfaces: the well-lit quartile.
    light = np.median(kept[:, 0]) if cls == "metal" else np.percentile(kept[:, 0], 75)
    med = np.array([light, np.median(kept[:, 1]), np.median(kept[:, 2])])
    med, fused = fuse(med, said, scene_cast(rgb) if cast is None else cast) if cls != "metal" else (med, None)
    colour = cv2.cvtColor(np.uint8([[np.clip(med, 0, 255)]]), cv2.COLOR_LAB2RGB)[0, 0] / 255.0
    out = {"color": colour.tolist(), "texture": None, "metallic": metallic, "roughness": finish_rough, "alpha": 1.0, "source": "colour sample",
           "confidence": round(min(1.0, share * 1.2), 2), "class": cls, **({"fused": fused} if fused else {})}
    grain = _grain(face, valid)
    out["grain"] = round(grain, 3)
    if cls in WOOD + PATTERNED and grain > 0.25:
        sq = _clean_square(valid)
        if sq and sq[2] >= 32:
            x, y, side = sq
            out.update(texture=_tile(face[y:y + side, x:x + side], np.asarray(colour)), source="texture sample")
    return out
