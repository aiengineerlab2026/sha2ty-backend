"""Plan scale from every reference available; never returns without one.

Priority: printed dimensions (OCR, dimension_ocr.calibrate, and GPT's reading, plan_reader.py) →
door openings → standard furniture → a labelled low-confidence heuristic (wall thickness). Every reference gives one metres-per-pixel
candidate; outliers are dropped with a weighted median/MAD test and the weighted median of the
rest is the scale. Doors outweigh furniture (architecture is drawn more consistently).
"""

from __future__ import annotations

import cv2
import numpy as np

DOOR_WIDTH_M = 0.8  # interior doors are 0.7–0.9 m; one door is never trusted alone
DOOR_WEIGHT = 1.0
FURNITURE_WEIGHT = 0.5
HEURISTIC_WALL_M = 0.2  # last resort: a typical wall body
PRINTED_MIN_CONFIDENCE = 0.5

# Furniture references: (zone kinds, how a blob matches, metres it measures). Approximate on purpose.
BATHTUB_M = 1.6  # 1.5–1.7 m long
BED_AREA_M2 = 1.6 * 2.0  # double bed
COUNTER_DEPTH_M = 0.6


def door_references(openings: list, min_confidence: float = 0.6) -> list[dict]:
    refs = []
    for o in openings:
        kind, conf = (o["kind"], o.get("confidence", 1.0)) if isinstance(o, dict) else (o.kind, o.confidence)
        start, end = (o["start"], o["end"]) if isinstance(o, dict) else (o.start, o.end)
        width = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        if kind == "door" and conf >= min_confidence and width > 4:
            refs.append({"source": "door", "label": f"Door ({width:.0f} px ≈ {DOOR_WIDTH_M} m)", "mpp": DOOR_WIDTH_M / width, "weight": DOOR_WEIGHT * conf})
    return refs


def furniture_references(gray: np.ndarray, zones: list[dict]) -> list[dict]:
    """Bathtubs, double beds and kitchen counters drawn inside their rooms (outline blobs of dark ink)."""
    refs = []
    for z in zones:
        kind = z.get("kind")
        if kind not in ("bathroom", "bedroom", "kitchen") or len(z.get("polygon") or []) < 3:
            continue
        mask = np.zeros(gray.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.array(z["polygon"], np.int32)], 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))  # keep the walls out
        zone_area = float(cv2.countNonZero(mask))
        if zone_area < 400:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        zone_long = max(w, h)
        ink = ((gray < 128) & (mask > 0)).astype(np.uint8) * 255
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            _, (bw, bh), _ = cv2.minAreaRect(c)
            short, long = sorted((float(bw), float(bh)))
            if long < 20 or short < 4:  # glyphs and single lines
                continue
            aspect = short / long
            if kind == "bathroom" and 0.3 <= aspect <= 0.55 and 0.3 * zone_long <= long <= 0.95 * zone_long:
                ref = {"source": "furniture", "label": f"Bathtub ({long:.0f} px ≈ {BATHTUB_M} m)", "mpp": BATHTUB_M / long, "size": long}
            elif kind == "bedroom" and 0.7 <= aspect <= 0.9 and 0.08 <= short * long / zone_area <= 0.45:
                ref = {"source": "furniture", "label": "Double bed (≈ 1.6 × 2.0 m)", "mpp": float(np.sqrt(BED_AREA_M2 / (short * long))), "size": long}
            elif kind == "kitchen" and aspect <= 0.35 and long >= 0.4 * zone_long:
                ref = {"source": "furniture", "label": f"Kitchen counter ({short:.0f} px deep ≈ {COUNTER_DEPTH_M} m)", "mpp": COUNTER_DEPTH_M / short, "size": long}
            else:
                continue
            if best is None or ref["size"] > best["size"]:
                best = ref  # ponytail: the largest match per room; no symbol classifier
        if best:
            best.pop("size")
            refs.append({**best, "weight": FURNITURE_WEIGHT})
    return refs


def robust_scale(refs: list[dict]) -> tuple[float, list[dict]] | None:
    """Weighted median after rejecting candidates far from it (MAD, at least ±10%)."""
    if not refs:
        return None
    values = np.array([r["mpp"] for r in refs])
    weights = np.array([r["weight"] for r in refs])
    med = _weighted_median(values, weights)
    mad = float(np.median(np.abs(values - med)))
    keep = np.abs(values - med) <= max(3 * 1.4826 * mad, 0.1 * med)
    for r, k in zip(refs, keep):
        r["inlier"] = bool(k)
    return _weighted_median(values[keep], weights[keep]), [r for r in refs if r["inlier"]]


def estimate_scale(printed: dict, refs: list[dict], thickness_px: float) -> dict:
    """`printed` is calibrate()'s result; refs = GPT reading + doors + furniture."""
    scale = {**printed, "references": [], "estimated": False}
    strong_ocr = bool(printed.get("meters_per_pixel")) and printed.get("confidence", 0) >= PRINTED_MIN_CONFIDENCE
    has_gpt = any(r["source"].startswith("gpt") for r in refs)
    if strong_ocr and not has_gpt:
        scale["source"] = "printed dimensions"
        return scale
    if not strong_ocr:
        scale["uniform"] = True  # one isotropic estimate; a stretched scan can't be told from references
    if printed.get("meters_per_pixel"):  # OCR'd printed sizes: the strongest single reference
        refs = [*refs, {"source": "printed", "label": "Printed room dimensions (OCR)", "mpp": printed["meters_per_pixel"], "weight": 3.0 if strong_ocr else 2.0}]
    found = robust_scale(refs)
    if found:
        mpp, inliers = found
        count = lambda *sources: sum(r["source"] in sources for r in inliers)  # noqa: E731
        printed_n, gpt_printed, gpt_guess = count("printed"), count("gpt_printed"), count("gpt")
        doors, furniture = count("door"), count("furniture")
        strong = printed_n + gpt_printed
        conf = 0.2 + 0.25 * min(strong, 3) + 0.08 * min(doors, 5) + 0.06 * min(furniture, 3) + 0.04 * min(gpt_guess, 3) - (0.15 if len(inliers) == 1 else 0.0)
        scale.update(
            meters_per_pixel=mpp,
            # Only written measurements (OCR or read by GPT) may approach certainty; estimates stay below.
            confidence=round(min(max(conf, 0.15), 0.9 if strong else 0.75), 2),
            estimated=not strong,
            source=" + ".join(
                s for s, n in (("printed dimensions", printed_n), ("GPT reading", gpt_printed + gpt_guess), ("doors", doors), ("furniture", furniture)) if n
            ),
            references=refs,
        )
        if doors == 1 and not furniture:
            scale["warnings"] = [*scale.get("warnings", []), "Scale estimated from a single door; measurements are rough."]
    else:
        scale.update(
            meters_per_pixel=HEURISTIC_WALL_M / max(thickness_px, 1.0),
            confidence=0.1,
            estimated=True,
            source="heuristic",
            references=[{"source": "heuristic", "label": f"Walls assumed {HEURISTIC_WALL_M} m thick", "mpp": HEURISTIC_WALL_M / max(thickness_px, 1.0), "weight": 0.1, "inlier": True}],
        )
        scale["warnings"] = [*scale.get("warnings", []), "No door or furniture to measure; scale is a rough guess from wall thickness."]
    return scale


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, cum[-1] / 2)])
