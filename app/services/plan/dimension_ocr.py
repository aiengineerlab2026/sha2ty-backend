"""Printed room names and dimensions from a floor plan, plus scale calibration. No LLMs.

OCR always runs on the ORIGINAL image (independent of wall detection), on a few variants
(original, 2x upscale, rotated 90° for vertical labels); results are merged by overlap,
keeping the higher-confidence reading, and mapped back to original pixel coordinates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

FT = 0.3048
INCH = 0.0254

ROOM_NAMES = (
    ("living room", "Living Room", "living"),
    ("living", "Living Room", "living"),
    ("family", "Family Room", "living"),
    ("dining", "Dining", "dining"),
    ("kitchen", "Kitchen", "kitchen"),
    ("master bedroom", "Master Bedroom", "bedroom"),
    ("bedroom", "Bedroom", "bedroom"),
    ("bed room", "Bedroom", "bedroom"),
    ("bathroom", "Bathroom", "bathroom"),
    ("bath", "Bathroom", "bathroom"),
    ("wc", "Bathroom", "bathroom"),
    ("powder", "Bathroom", "bathroom"),
    ("patio", "Patio", "patio"),
    ("deck", "Patio", "patio"),
    ("balcony", "Balcony", "patio"),
    ("terrace", "Terrace", "patio"),
    ("closet", "Closet", "closet"),
    ("wardrobe", "Closet", "closet"),
    ("laundry", "Laundry", "laundry"),
    ("hall", "Hall", "hall"),
    ("foyer", "Foyer", "hall"),
    ("entry", "Entry", "hall"),
    ("garage", "Garage", "garage"),
    ("office", "Office", "room"),
    ("study", "Study", "room"),
)

# Normalise the many glyphs OCR and drawings use for feet, inches and "by".
_GLYPHS = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "′": "'", "”": '"', "“": '"', "″": '"', "×": "x", "✕": "x", "*": "x", "X": "x", "–": "-", "—": "-"})

_FT_IN = re.compile(
    r"""^\s*
    (?P<ft>\d+(?:\.\d+)?)\s*(?:ft\.?|feet|foot|')
    \s*-?\s*
    (?:(?P<inch>\d+(?:\.\d+)?)\s*(?:in\.?|inch(?:es)?|"|'')?)?
    \s*$""",
    re.IGNORECASE | re.VERBOSE,
)
_INCH = re.compile(r'^\s*(?P<v>\d+(?:\.\d+)?)\s*(?:in\.?|inch(?:es)?|"|\'\')\s*$', re.IGNORECASE)
_METRIC = re.compile(r"^\s*(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>mm|cm|m|meters?|metres?|متر|سم)\s*$", re.IGNORECASE)
_BARE = re.compile(r"^\s*(?P<v>\d+(?:\.\d+)?)\s*$")
# "12 6" — OCR lost both marks of 12'6"
_FT_IN_BARE = re.compile(r"^\s*(?P<ft>\d{1,2})\s+(?P<inch>\d{1,2})\s*$")
_PAIR = re.compile(r"\s*(?:x|by)\s*", re.IGNORECASE)


@dataclass
class OcrBox:
    text: str
    x: float
    y: float
    width: float
    height: float
    score: float = 1.0
    rotated: bool = False


@dataclass
class Dimension:
    a: float  # metres
    b: float
    unit: str  # "ft" | "m" | "unknown"
    uncertain: bool
    text: str


@dataclass
class OcrMeasurement:
    text: str
    meters: float
    unit: str
    x: float
    y: float
    width: float
    height: float
    pair: tuple[float, float] | None = None
    label: str | None = None
    uncertain: bool = False


def normalize(text: str) -> str:
    return text.translate(_GLYPHS).replace("''", '"')


def parse_length(token: str) -> tuple[float, str, bool] | None:
    """One length → (metres, unit, uncertain). None when it is not a length."""
    t = normalize(token).strip().strip(",;")
    if not t:
        return None
    if m := _FT_IN.match(t):
        return float(m["ft"]) * FT + float(m["inch"] or 0) * INCH, "ft", False
    if m := _METRIC.match(t):
        v = float(m["v"].replace(",", "."))
        u = m["u"].lower()
        if u == "mm":
            return v / 1000, "m", False
        if u in ("cm", "سم"):
            return v / 100, "m", False
        return v, "m", False
    if m := _FT_IN_BARE.match(t):
        if float(m["inch"]) < 12:
            return float(m["ft"]) * FT + float(m["inch"]) * INCH, "ft", True
    if m := _INCH.match(t):
        return float(m["v"]) * INCH, "in", False
    if m := _BARE.match(t):
        return float(m["v"]), "unknown", True
    return None


def parse_dimension(text: str, drawing_unit: str | None = None) -> Dimension | None:
    """'13'x 12'', '12'6" x 10'', '13 × 12', '4m x 3m', '12 6 × 10"' (OCR-mangled) → Dimension."""
    t = normalize(text)
    # Keep only the dimension part of "Bedroom 13'x12'"
    m = re.search(r"\d[\d\s'\".,\-]*(?:ft|feet|in|m|cm|mm)?\s*(?:x|by)\s*\d[\d\s'\".,\-]*(?:ft|feet|in|m|cm|mm)?", t, re.IGNORECASE)
    if not m:
        return None
    parts = _PAIR.split(m.group(0).strip())
    if len(parts) != 2:
        return None
    pa, pb = parse_length(parts[0]), parse_length(parts[1])
    if pa is None or pb is None:
        return None
    (va, ua, xa), (vb, ub, xb) = pa, pb
    # Uncertain = OCR evidently mangled the marks, or no unit is printed on either side.
    uncertain = (xa and ua != "unknown") or (xb and ub != "unknown")
    # A lone inch mark next to a feet value is a misread foot mark: 12'6" x 10" means 10'.
    if ua == "ft" and ub == "in" and vb / INCH >= 4:
        vb, ub, uncertain = vb / INCH * FT, "ft", True
    if ub == "ft" and ua == "in" and va / INCH >= 4:
        va, ua, uncertain = va / INCH * FT, "ft", True
    units = {u for u in (ua, ub) if u != "unknown"}
    if len(units) > 1:
        return None
    if not units:
        uncertain = True
    unit = next(iter(units)) if units else (drawing_unit or "unknown")
    if ua == "unknown":
        va = _apply_unit(va, unit)
    if ub == "unknown":
        vb = _apply_unit(vb, unit)
    if unit == "unknown" or va <= 0 or vb <= 0 or va > 60 or vb > 60:
        return None
    return Dimension(va, vb, unit, bool(uncertain), m.group(0).strip())


def _apply_unit(value: float, unit: str) -> float:
    return value * FT if unit == "ft" else value if unit == "m" else 0.0


def parse_length_to_meters(token: str) -> float | None:
    parsed = parse_length(token)
    if parsed is None or parsed[1] == "unknown":
        return None
    return parsed[0]


def parse_dimension_pair(text: str) -> tuple[float, float] | None:
    """Back-compat helper. Bare numbers ('13 x 12') are read as feet only here; the pipeline instead
    infers the unit from the other labels on the drawing (see room_labels)."""
    dim = parse_dimension(text) or parse_dimension(text, drawing_unit="ft")
    return (dim.a, dim.b) if dim else None


def parse_room_label(text: str) -> tuple[str, str] | None:
    lower = re.sub(r"[^a-z ]", " ", normalize(text).lower())
    lower = re.sub(r"\s+", " ", lower).strip()
    hits = [(name, kind) for needle, name, kind in ROOM_NAMES if re.search(rf"\b{needle}", lower)]
    if not hits:
        return None
    kinds = list(dict.fromkeys(k for _, k in hits))
    words = [w for w in lower.split() if len(w) > 2 and w not in {"by", "ft", "feet", "inch", "sqm"}]  # drop "x", units
    if len(kinds) > 1 and len(words) <= 6:  # "Living Room and Kitchen": one open space, its own name
        return " ".join(words).title().replace(" And ", " and "), kinds[0]
    return hits[0]


# ── OCR engines ────────────────────────────────────────────────────────────────────────────


def run_ocr(image_bgr: np.ndarray) -> tuple[list[OcrBox], list[str]]:
    """All text on the original drawing, merged across preprocessing variants."""
    notes: list[str] = []
    engine = _engine()
    if engine is None:
        notes.append(
            "No OCR engine is available (macOS Vision via ocrmac, PaddleOCR, RapidOCR or Tesseract). "
            "Room names and printed dimensions could not be read; confirm the scale manually."
        )
        return [], notes
    name, fn = engine
    boxes: list[OcrBox] = []
    for variant, scale, rotated in _variants(image_bgr):
        try:
            found = fn(variant)
        except Exception as exc:  # engine failure is reported, never silently swallowed
            notes.append(f"OCR ({name}) failed on a variant: {exc}")
            continue
        h, w = image_bgr.shape[:2]
        for b in found:
            if rotated:
                # variant = original rotated 90° clockwise: (x', y') = (h - 1 - y, x)
                x, y = b.y / scale, h - (b.x + b.width) / scale
                box = OcrBox(b.text, x, y, b.height / scale, b.width / scale, b.score, True)
                if box.height < box.width * 1.5:
                    continue  # only keep text that really runs vertically on the drawing
            else:
                box = OcrBox(b.text, b.x / scale, b.y / scale, b.width / scale, b.height / scale, b.score)
            boxes.append(box)
    merged = _merge_variants(boxes)
    logger.info("OCR %s: %d boxes", name, len(merged))
    return merged, notes


def _variants(image_bgr: np.ndarray):
    import cv2

    yield image_bgr, 1.0, False
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if max(gray.shape) < 2400:
        up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        yield cv2.cvtColor(up, cv2.COLOR_GRAY2BGR), 2.0, False
    yield cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), 1.0, True


def _merge_variants(boxes: list[OcrBox]) -> list[OcrBox]:
    """Same text region read by several variants: keep the reading that parses best / scores highest."""

    def quality(b: OcrBox) -> tuple:
        has_dim = parse_dimension(b.text) is not None
        has_name = parse_room_label(b.text) is not None
        return (has_dim or has_name, b.score, len(b.text.strip()))

    kept: list[OcrBox] = []
    for b in sorted(boxes, key=quality, reverse=True):
        if not b.text.strip():
            continue
        if any(_iou(b, k) > 0.3 for k in kept):
            continue
        kept.append(b)
    return sorted(kept, key=lambda b: (b.y, b.x))


def _iou(a: OcrBox, b: OcrBox) -> float:
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


@lru_cache(maxsize=1)
def _engine():
    """First engine that actually loads. Cached: PaddleOCR model init costs seconds."""
    for name, probe in (("macos-vision", _probe_vision), ("paddleocr", _probe_paddle), ("rapidocr", _probe_rapid), ("tesseract", _probe_tesseract)):
        try:
            fn = probe()
        except Exception as exc:
            logger.info("OCR engine %s unavailable: %s", name, exc)
            continue
        if fn is not None:
            return name, fn
    return None


def _probe_vision():
    import Vision  # noqa: F401  (pyobjc-framework-Vision)

    return _ocr_vision


def _ocr_vision(image_bgr: np.ndarray) -> list[OcrBox]:
    import Vision
    from Foundation import NSData
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.fromarray(image_bgr[:, :, ::-1]).save(buf, format="PNG")
    data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)  # language correction "fixes" 13'x12' into words
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    ok, err = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(str(err))
    h, w = image_bgr.shape[:2]
    out = []
    for obs in request.results() or []:
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        text = str(cand[0].string())
        bb = obs.boundingBox()  # normalised, origin bottom-left
        x, bw, bh = bb.origin.x * w, bb.size.width * w, bb.size.height * h
        y = (1 - bb.origin.y - bb.size.height) * h
        out.append(OcrBox(text, float(x), float(y), float(bw), float(bh), float(cand[0].confidence())))
    return out


def _probe_paddle():
    from paddleocr import PaddleOCR  # type: ignore

    try:  # PaddleOCR 3.x
        engine = PaddleOCR(lang="en", use_textline_orientation=True)
        v3 = True
    except (TypeError, ValueError):  # 2.x: `show_log` / `use_angle_cls`
        engine = PaddleOCR(lang="en", use_angle_cls=True)
        v3 = False

    def run(image_bgr: np.ndarray) -> list[OcrBox]:
        out = []
        if v3:
            for page in engine.predict(image_bgr) or []:
                for text, score, poly in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
                    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
                    out.append(OcrBox(str(text), min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), float(score)))
            return out
        result = engine.ocr(image_bgr, cls=True) or []
        for line in (result[0] if result and isinstance(result[0], list) else result) or []:
            poly, (text, score) = line[0], line[1]
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            out.append(OcrBox(str(text), min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), float(score)))
        return out

    return run


def _probe_rapid():
    try:
        from rapidocr import RapidOCR  # type: ignore
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    engine = RapidOCR()

    def run(image_bgr: np.ndarray) -> list[OcrBox]:
        result = engine(image_bgr)
        rows = result[0] if isinstance(result, tuple) else list(zip(result.boxes, result.txts, result.scores))
        out = []
        for poly, text, score in rows or []:
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            out.append(OcrBox(str(text), min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), float(score or 0)))
        return out

    return run


def _probe_tesseract():
    import pytesseract  # type: ignore

    pytesseract.get_tesseract_version()

    def run(image_bgr: np.ndarray) -> list[OcrBox]:
        from PIL import Image

        data = pytesseract.image_to_data(Image.fromarray(image_bgr[:, :, ::-1]), output_type=pytesseract.Output.DICT, config="--psm 11")
        words = [
            OcrBox(str(t), float(l), float(tp), float(w), float(h), max(0.0, float(c)) / 100)
            for t, l, tp, w, h, c in zip(data["text"], data["left"], data["top"], data["width"], data["height"], data["conf"])
            if str(t).strip()
        ]
        return _join_words(words)

    return run


def _join_words(words: list[OcrBox]) -> list[OcrBox]:
    """Word-level engines only: join words on one baseline separated by less than ~one glyph height."""
    words = sorted(words, key=lambda b: (b.y, b.x))
    lines: list[list[OcrBox]] = []
    for w in words:
        for line in lines:
            last = line[-1]
            same_row = abs((w.y + w.height / 2) - (last.y + last.height / 2)) < max(w.height, last.height) * 0.5
            if same_row and 0 <= w.x - (last.x + last.width) < max(w.height, last.height) * 1.2:
                line.append(w)
                break
        else:
            lines.append([w])
    out = []
    for line in lines:
        x0, y0 = min(b.x for b in line), min(b.y for b in line)
        x1, y1 = max(b.x + b.width for b in line), max(b.y + b.height for b in line)
        out.append(OcrBox(" ".join(b.text for b in line), x0, y0, x1 - x0, y1 - y0, float(np.mean([b.score for b in line]))))
    return out


# ── Labels ─────────────────────────────────────────────────────────────────────────────────


def room_labels(boxes: list[OcrBox]):
    """Room-name boxes paired with the dimension printed with them (same box, or just below)."""
    from app.services.plan.room_segmenter import RoomLabel

    explicit = [d.unit for b in boxes if (d := parse_dimension(b.text)) and d.unit != "unknown"]
    drawing_unit = max(set(explicit), key=explicit.count) if explicit else None
    dims = []
    for b in boxes:
        d = parse_dimension(b.text, drawing_unit)
        if d:
            dims.append((b, d))
    used: set[int] = set()
    labels = []
    for b in boxes:
        named = parse_room_label(b.text)
        if not named or _is_title(b, boxes):
            continue
        name, kind = named
        dim = parse_dimension(b.text, drawing_unit)
        dim_box = b if dim else None
        if dim is None:
            best = None
            for i, (db, d) in enumerate(dims):
                if i in used or db is b:
                    continue
                dy = db.y - (b.y + b.height)
                overlap = min(b.x + b.width, db.x + db.width) - max(b.x, db.x)
                if -b.height * 0.3 <= dy <= b.height * 2.5 and overlap > -b.height:
                    if best is None or dy < best[0]:
                        best = (dy, i, db, d)
            if best:
                used.add(best[1])
                dim_box, dim = best[2], best[3]
        x, y = b.x + b.width / 2, b.y + b.height / 2
        if dim_box is not None and dim_box is not b:
            # Anchor between name and dimension: the label block's centre.
            x = (min(b.x, dim_box.x) + max(b.x + b.width, dim_box.x + dim_box.width)) / 2
            y = (b.y + dim_box.y + dim_box.height) / 2
        labels.append(
            RoomLabel(
                name=name,
                kind=kind,
                x=x,
                y=y,
                box=(b.x, b.y, b.width, b.height),
                text=b.text,
                dims_m=(dim.a, dim.b) if dim else None,
                dims_text=dim.text if dim else None,
                dims_uncertain=dim.uncertain if dim else False,
            )
        )
    return labels, drawing_unit


def _is_title(b: OcrBox, boxes: list[OcrBox]) -> bool:
    """Drawing titles ('3 Bedroom Apartment Floor Plan') are larger text than room labels."""
    heights = sorted(x.height for x in boxes if not x.rotated)
    if len(heights) < 3:
        return False
    median = heights[len(heights) // 2]
    # Larger text alone is not enough: "Living Room and Kitchen" is a room label set a bit bigger.
    titled = re.search(r"\b(plan|floor|apartment|flat|layout|level|unit|house|residence)\b", b.text, re.IGNORECASE)
    return b.height > median * 1.5 and len(b.text.split()) >= 3 and bool(titled)


def measurements_from_boxes(boxes: list[OcrBox], warnings: list[str] | None = None):
    """Flat list of names/dimensions per OCR box (used by the debug overlay and API payload)."""
    warnings = warnings if warnings is not None else []
    explicit = [d.unit for b in boxes if (d := parse_dimension(b.text)) and d.unit != "unknown"]
    unit = max(set(explicit), key=explicit.count) if explicit else None
    found = []
    for b in boxes:
        d = parse_dimension(b.text, unit)
        named = parse_room_label(b.text)
        if not d and not named:
            continue
        found.append(
            OcrMeasurement(
                text=b.text,
                meters=d.a if d else 0.0,
                unit=d.unit if d else "",
                x=b.x,
                y=b.y,
                width=b.width,
                height=b.height,
                pair=(d.a, d.b) if d else None,
                label=named[0] if named else None,
                uncertain=d.uncertain if d else False,
            )
        )
    if boxes and not any(m.pair for m in found):
        warnings.append("OCR read text but no printed room dimensions.")
    return found, warnings


# ── Scale ──────────────────────────────────────────────────────────────────────────────────


def calibrate(zones: list[dict]) -> dict:
    """Pixels-per-metre from printed room dimensions vs measured clear widths.

    Every labelled zone bounded by walls (or a drawn line) contributes one candidate per axis.
    The printed pair is matched to (width, depth) in the orientation that agrees best. Outliers
    are rejected with a median/MAD test; X and Y are compared to detect non-uniform scaling.
    """
    candidates = []
    for z in zones:
        dims = z.get("dims_m")
        m = z.get("measure_px") or {}
        if not dims or not m.get("width") or not m.get("depth"):
            continue
        if z.get("boundary") == "inferred":
            continue  # a cut we chose is not evidence of size
        wx, wy = m["width"], m["depth"]
        a, b = dims
        options = [((a / wx), (b / wy)), ((b / wx), (a / wy))]
        mx, my = min(options, key=lambda o: abs(o[0] - o[1]) / max(o[0], o[1]))
        swapped = options.index((mx, my)) == 1
        candidates.append(
            {
                "zone_id": z["id"],
                "name": z["name"],
                "printed": z.get("dims_text"),
                "printed_m": [b, a] if swapped else [a, b],
                "measured_px": [wx, wy],
                "mpp_x": mx,
                "mpp_y": my,
                "uncertain": bool(z.get("dims_uncertain")),
            }
        )
    result = {"meters_per_pixel": None, "confidence": 0.0, "candidates": candidates, "inliers": [], "warnings": [], "uniform": True}
    if not candidates:
        return result

    values = np.array([[c["mpp_x"], c["mpp_y"]] for c in candidates])
    flat = values.ravel()
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med))) or med * 0.02
    inlier_rows = [i for i, (vx, vy) in enumerate(values) if abs(vx - med) <= 3 * mad * 1.4826 + med * 0.04 and abs(vy - med) <= 3 * mad * 1.4826 + med * 0.04]
    for i, c in enumerate(candidates):
        c["inlier"] = i in inlier_rows
    if not inlier_rows:
        result["warnings"].append("Printed dimensions disagree with each other and with the drawing; the scale is estimated instead.")
        return result
    inl = values[inlier_rows]
    mpp = float(np.median(inl))
    mx, my = float(np.median(inl[:, 0])), float(np.median(inl[:, 1]))
    spread = float(np.std(inl) / mpp) if len(inl) > 1 else 0.0
    result.update(meters_per_pixel=mpp, inliers=[candidates[i]["zone_id"] for i in inlier_rows], spread=round(spread, 3))
    result["mpp_x"], result["mpp_y"] = mx, my
    if abs(mx - my) / mpp > 0.06:
        result["uniform"] = False
        result["warnings"].append(
            f"Horizontal and vertical scales differ by {abs(mx - my) / mpp:.0%}. The image may be stretched."
        )
    rejected = [c["name"] for i, c in enumerate(candidates) if i not in inlier_rows]
    if rejected:
        result["warnings"].append(f"Ignored inconsistent printed dimensions for: {', '.join(rejected)}.")
    n = len(inlier_rows)
    conf = 0.5 if n == 1 else min(0.95, 0.6 + 0.1 * n - spread)
    if not result["uniform"]:
        conf = min(conf, 0.5)
    if all(candidates[i]["uncertain"] for i in inlier_rows):
        conf = min(conf, 0.5)
    result["confidence"] = round(max(conf, 0.0), 2)
    if n == 1:
        result["warnings"].append("Scale comes from a single printed dimension.")
    return result
