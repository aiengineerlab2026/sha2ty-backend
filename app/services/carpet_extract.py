"""Cut a carpet out of a vendor photo: segment it from the floor, straighten it to a top-down
rectangle with the carpet's real aspect ratio, and make everything outside it transparent."""

from __future__ import annotations

import cv2
import numpy as np

WORK = 800  # segmentation runs on a downscaled copy; the warp samples the full-resolution photo
OUT_LONG = 1024


class CarpetNotFound(ValueError):
    pass


def extract_carpet(data: bytes, width_m: float | None = None, length_m: float | None = None) -> bytes:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise CarpetNotFound("The file is not a readable image.")
    h, w = image.shape[:2]
    k = min(1.0, WORK / max(h, w))
    small = cv2.resize(image, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else image.copy()

    mask = _segment(small)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise CarpetNotFound("Could not find a carpet in the photo. Use a photo where the carpet fills most of the frame.")
    carpet = max(contours, key=cv2.contourArea)
    if cv2.contourArea(carpet) < 0.05 * mask.size:
        raise CarpetNotFound("The carpet is too small in the photo. Take it from closer, from above.")

    quad = _quad(carpet) / k  # back to full-resolution pixels
    full_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(full_mask, [np.round(carpet / k).astype(np.int32)], -1, 255, cv2.FILLED)

    out_w, out_h, quad = _output_size(quad, width_m, length_m)
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    rgb = cv2.warpPerspective(image, m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpPerspective(full_mask, m, (out_w, out_h), flags=cv2.INTER_LINEAR)
    ok, png = cv2.imencode(".png", np.dstack([rgb, alpha]))
    if not ok:
        raise CarpetNotFound("Could not encode the extracted carpet.")
    return png.tobytes()


def _segment(img: np.ndarray) -> np.ndarray:
    """GrabCut seeded with "the carpet is inside, a thin border band is floor"."""
    h, w = img.shape[:2]
    mx, my = max(2, round(w * 0.03)), max(2, round(h * 0.03))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, (mx, my, w - 2 * mx, h - 2 * my), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)  # fill pattern holes
    return cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)  # drop specks


def _quad(contour: np.ndarray) -> np.ndarray:
    """Four corners of the carpet: a 4-point simplification of its outline if one exists (keeps
    perspective), else its minimum-area rectangle (round or irregular rugs)."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return _order(approx.reshape(4, 2).astype(np.float64))
    return _order(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64))


def _order(pts: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left."""
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def _output_size(quad: np.ndarray, width_m: float | None, length_m: float | None) -> tuple[int, int, np.ndarray]:
    """Output keeps the carpet's real aspect (image x = width, y = length) when given, else the
    photo's. The quad is rotated so the carpet's long side lands on the output's long side."""
    top = np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])
    side = np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])
    aspect = (width_m / length_m) if width_m and length_m else top / max(side, 1e-9)  # out_w / out_h
    if (top >= side) != (aspect >= 1):
        quad = np.roll(quad, -1, axis=0)  # turn 90°: photo's long side → carpet's long side
    if aspect >= 1:
        return OUT_LONG, max(1, round(OUT_LONG / aspect)), quad
    return max(1, round(OUT_LONG * aspect)), OUT_LONG, quad
