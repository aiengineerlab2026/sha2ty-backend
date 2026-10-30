import cv2
import numpy as np

from app.services.carpet_extract import extract_carpet


def _photo() -> bytes:
    """Wooden-ish floor with a tilted red carpet (2:3) seen slightly in perspective."""
    img = np.full((600, 800, 3), (120, 150, 170), np.uint8)
    img[::12] = (100, 130, 150)  # floor boards
    carpet = np.float32([[230, 110], [560, 140], [600, 520], [190, 480]])
    cv2.fillPoly(img, [carpet.astype(np.int32)], (40, 40, 190))
    for i in range(0, 600, 25):  # carpet pattern
        cv2.line(img, (0, i), (800, i), (60, 60, 210), 3)
    outside = np.ones((600, 800), np.uint8)
    cv2.fillPoly(outside, [carpet.astype(np.int32)], 0)
    img[outside.astype(bool)] = np.full((600, 800, 3), (120, 150, 170), np.uint8)[outside.astype(bool)]
    img[::12][outside[::12].astype(bool)] = (100, 130, 150)
    return cv2.imencode(".png", img)[1].tobytes()


def test_extracts_carpet_at_real_aspect():
    out = cv2.imdecode(np.frombuffer(extract_carpet(_photo(), 2.0, 3.0), np.uint8), cv2.IMREAD_UNCHANGED)
    h, w, channels = out.shape
    assert channels == 4
    assert abs(w / h - 2 / 3) < 0.01, (w, h)
    inner = out[h // 5 : -h // 5, w // 5 : -w // 5]
    assert inner[..., 3].mean() > 250  # carpet opaque
    b, g, r = inner[..., 0].mean(), inner[..., 1].mean(), inner[..., 2].mean()
    assert r > 150 and b < 90, (b, g, r)  # it is the red carpet, not the floor
