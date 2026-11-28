"""Non-generative image processing on the ORIGINAL photo (OpenCV): nothing here invents or repaints a pixel.

  silhouette(img, box)       the furniture's mask (GrabCut seeded with the object's box): background removal that keeps
                             every furniture pixel as photographed (only alpha is added)
  rectified(img, rect)       a part's region straightened: its four corners found from its edges and warped to a
                             rectangle (perspective correction), else the plain crop
  pattern(crop)              repeated straight lines across a part (grooves, flutes, slats): their axis and count,
                             from the period of the edge profile
  cutout(img, rect, mask)    a part's pixels with the background transparent, for a mesh provider

ponytail: GrabCut (in OpenCV, CPU, ~0.3 s) instead of BiRefNet / SAM. Those need torch plus 0.2–1 GB of weights and
2–4 GB of RAM per instance; onnxruntime (rembg) has no build for Python 3.14. A learned segmenter is the upgrade if
GrabCut's masks fall short on busy rooms.
"""

from __future__ import annotations

import io

import cv2
import numpy as np


def to_array(img) -> np.ndarray:
    return np.asarray(img.convert("RGB")) if hasattr(img, "convert") else np.asarray(img)


def silhouette(rgb: np.ndarray, box: list[float] | None, iterations: int = 4) -> np.ndarray:
    """The object's mask (uint8 0/255): GrabCut started from its box in the photo (fractions). The whole box when
    GrabCut finds nothing usable."""
    full_h, full_w = rgb.shape[:2]
    scale = min(1.0, 320 / max(full_h, full_w))  # GrabCut on ≤ 320 px (~0.1 s), the mask scaled back up
    small = cv2.resize(rgb, (max(2, int(full_w * scale)), max(2, int(full_h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    out = _grabcut(small, box, iterations) | _unlike_surroundings(small, box)
    return cv2.resize(out, (full_w, full_h), interpolation=cv2.INTER_NEAREST) if scale < 1 else out


def _unlike_surroundings(rgb: np.ndarray, box: list[float] | None, distance: float = 32) -> np.ndarray:
    """Pixels inside the object's box whose colour is far from every colour of the ring just outside it (the wall,
    the floor): GrabCut alone drops furniture parts whose colour its background model also covers."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box if box and len(box) == 4 else (0.02, 0.02, 0.98, 0.98)
    X0, Y0, X1, Y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    mx, my = max(6, int(0.06 * (X1 - X0))), max(6, int(0.06 * (Y1 - Y0)))
    ring = np.zeros((h, w), bool)
    ring[max(0, Y0 - my):min(h, Y1 + my), max(0, X0 - mx):min(w, X1 + mx)] = True
    gap = 3  # px: the box's edge row is often still furniture (rounding, a soft outline): never sampled as background
    ring[max(0, Y0 - gap):min(h, Y1 + gap), max(0, X0 - gap):min(w, X1 + gap)] = False
    out = np.zeros((h, w), np.uint8)
    if ring.sum() < 50:
        return out
    samples = rgb[ring].astype(np.float32)
    k = min(8, len(samples))
    _, _, centres = cv2.kmeans(samples, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    inside = rgb[Y0:Y1, X0:X1].astype(np.float32).reshape(-1, 3)
    far = np.min(np.linalg.norm(inside[:, None, :] - centres[None], axis=2), axis=1) > distance
    out[Y0:Y1, X0:X1] = (far.reshape(Y1 - Y0, X1 - X0) * 255).astype(np.uint8)
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _grabcut(rgb: np.ndarray, box: list[float] | None, iterations: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box if box and len(box) == 4 else (0.02, 0.02, 0.98, 0.98)
    rect = (max(1, int(x0 * w)), max(1, int(y0 * h)), max(2, int((x1 - x0) * w) - 2), max(2, int((y1 - y0) * h) - 2))
    mask = np.zeros((h, w), np.uint8)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    iterations, cv2.GC_INIT_WITH_RECT)
        out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        out = np.zeros((h, w), np.uint8)
    inside = out[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    if inside.size == 0 or inside.mean() < 0.2 * 255:  # nothing sensible: the box itself
        out[:] = 0
        out[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 255
    return out


def _order(quad: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    s, d = quad.sum(1), np.diff(quad, axis=1).ravel()
    return np.array([quad[np.argmin(s)], quad[np.argmin(d)], quad[np.argmax(s)], quad[np.argmax(d)]], np.float32)


def rectified(rgb: np.ndarray, rect: tuple[int, int, int, int], pad: float = 0.08) -> tuple[np.ndarray, bool]:
    """(the part's pixels as an upright rectangle, whether a perspective was corrected). The part's own outline is
    searched in a slightly larger window; a convex quadrilateral covering most of it is warped flat."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    H, W = rgb.shape[:2]
    px, py = int(pad * w), int(pad * h)
    wx0, wy0, wx1, wy1 = max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)
    window = rgb[wy0:wy1, wx0:wx1]
    plain = rgb[y0:y1, x0:x1].copy()
    if window.size == 0 or min(w, h) < 24:
        return plain, False
    # The part's outline: pixels unlike the window's border (what surrounds it), closed over grooves and handles.
    border = np.concatenate([window[0], window[-1], window[:, 0], window[:, -1]]).astype(np.float32)
    differs = (np.linalg.norm(window.astype(np.float32) - np.median(border, axis=0), axis=2) > 25).astype(np.uint8) * 255
    k = max(3, int(0.04 * min(w, h)) | 1)
    differs = cv2.morphologyEx(differs, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(differs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) >= 0.6 * w * h:
            best = approx.reshape(4, 2).astype(np.float32)
            break
    if best is None:
        return plain, False
    src = _order(best) + np.float32([wx0, wy0])
    window = rgb
    # Only a real perspective: an axis-aligned rectangle (the usual front photo) is just the crop.
    skew = max(abs(src[0][1] - src[1][1]), abs(src[3][1] - src[2][1]), abs(src[0][0] - src[3][0]), abs(src[1][0] - src[2][0]))
    del window
    if skew < 0.02 * max(w, h):
        return plain, False
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(src, dst), (w, h)), True


def pattern(crop: np.ndarray, min_count: int = 3, max_count: int = 80) -> dict | None:
    """Repeated straight lines across the part ({"axis": "vertical"|"horizontal", "count", "regularity"}), or None.
    Grooves, flutes and slats show as lines darker or lighter than the surface: along each axis the line profile (the
    surface minus its smoothed self) is searched for peaks; evenly spaced peaks covering the part are the pattern."""
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    best = None
    for axis, mean_axis in (("vertical", 0), ("horizontal", 1)):
        # Only the middle 60% of the other axis: a handle or the part's own edge band is not a line pattern.
        other = gray.shape[mean_axis]
        band = gray[int(0.2 * other):int(0.8 * other), :] if mean_axis == 0 else gray[:, int(0.2 * other):int(0.8 * other)]
        profile = band.mean(axis=mean_axis)
        n = len(profile)
        if n < 16:
            continue
        k = max(5, (n // 10) | 1)
        detail = np.abs(profile - cv2.GaussianBlur(profile.reshape(1, -1), (k, 1), 0).ravel())
        if detail.max() < 4:  # a plain surface (less than 4 grey levels of structure)
            continue
        threshold = detail.mean() + 1.0 * detail.std()
        peaks = [i for i in range(1, n - 1) if detail[i] >= threshold and detail[i] >= detail[i - 1] and detail[i] > detail[i + 1]]
        merged: list[int] = []
        for i in peaks:  # one line, not its two edges
            if merged and i - merged[-1] <= max(2, n // (4 * max_count)):
                continue
            merged.append(i)
        if not min_count <= len(merged) <= max_count:
            continue
        gaps = np.diff(merged)
        regularity = float(1 - gaps.std() / max(gaps.mean(), 1e-6))
        cover = (merged[-1] - merged[0]) / n
        if regularity >= 0.75 and cover >= 0.5 and (best is None or regularity > best["regularity"]):
            best = {"axis": axis, "count": len(merged), "regularity": round(regularity, 3)}
    return best


def pattern_band(crop: np.ndarray, bands: int = 10) -> dict | None:
    """Where on the part the photo shows a regular VERTICAL line pattern: the contiguous horizontal band (fractions of
    the part's height, top = 0) of rows whose slices show it, with its line count. A door fluted only along its
    bottom third gives that third; a plain part None."""
    h = crop.shape[0]
    bands = max(1, min(bands, h // 24))  # each band at least 24 px tall: enough rows to see a line in
    if bands == 1:
        whole = pattern(crop)
        return {**whole, "from": 0.0, "to": 1.0} if whole and whole["axis"] == "vertical" else None
    hits = []
    for i in range(bands):
        found = pattern(crop[i * h // bands:(i + 1) * h // bands])
        hits.append(found if found and found["axis"] == "vertical" else None)
    best, run = None, []
    for i, hit in enumerate(hits + [None]):
        if hit is not None:
            run.append(i)
            continue
        if run and (best is None or len(run) > len(best)):
            best = run
        run = []
    if not best:
        return None
    counts = [hits[i]["count"] for i in best]
    return {"axis": "vertical", "count": int(np.median(counts)), "regularity": round(float(np.mean([hits[i]["regularity"] for i in best])), 3),
            "from": best[0] / bands, "to": (best[-1] + 1) / bands}


def cutout(rgb: np.ndarray, rect: tuple[int, int, int, int], mask: np.ndarray | None = None, margin: float = 0.06) -> bytes:
    """The part's region as a PNG, background transparent where the object's mask says so (pixels untouched)."""
    from PIL import Image

    x0, y0, x1, y1 = rect
    H, W = rgb.shape[:2]
    mx, my = int(margin * (x1 - x0)), int(margin * (y1 - y0))
    x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
    region = rgb[y0:y1, x0:x1]
    alpha = mask[y0:y1, x0:x1] if mask is not None else np.full(region.shape[:2], 255, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.dstack([region, alpha])).save(buf, "PNG")
    return buf.getvalue()
