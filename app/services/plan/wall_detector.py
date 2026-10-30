"""Structural walls and verified openings from a 2D floor-plan raster. OpenCV only, no ML.

Stage A  ink masks (all ink / dark ink), text boxes removed, original pixel coordinates kept.
Stage B  wall thickness from granulometry, wall core = morphological opening at ~0.7 T,
         decomposed into horizontal / vertical / genuine diagonal wall segments.
Stage C  thin geometry (door arcs, fixtures, furniture, tiling, text) never survives the
         thickness opening; leftover blobs are dropped by size / elongation / connectivity.
Openings each wall endpoint casts a ray along its own axis. A gap is closed only with
         geometric evidence: a door-swing arc, collinear thin window/slider lines, or a
         collinear wall endpoint of the same thickness (low confidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class WallSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    orientation: str  # "h" | "v" | "d"
    confidence: float = 1.0

    @property
    def length(self) -> float:
        return float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]))

    def as_dict(self, index: int) -> dict:
        return {
            "id": f"w{index}",
            "start": [round(self.start[0], 1), round(self.start[1], 1)],
            "end": [round(self.end[0], 1), round(self.end[1], 1)],
            "thickness": round(self.thickness, 1),
            "orientation": self.orientation,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class Opening:
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    kind: str  # "door" | "window" | "opening"
    evidence: str
    confidence: float

    def as_dict(self, index: int) -> dict:
        return {
            "id": f"o{index}",
            "kind": self.kind,
            "start": [round(self.start[0], 1), round(self.start[1], 1)],
            "end": [round(self.end[0], 1), round(self.end[1], 1)],
            "thickness": round(self.thickness, 1),
            "length": round(float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])), 1),
            "evidence": self.evidence,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class WallDetection:
    walls: list[WallSegment]
    openings: list[Opening]
    thickness: float
    wall_mask: np.ndarray
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, np.ndarray] = field(default_factory=dict)


def detect_walls(image_bgr: np.ndarray, text_boxes: list[tuple[float, float, float, float]] | None = None) -> WallDetection:
    warnings: list[str] = []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    text = _text_mask(gray.shape, text_boxes or [])
    ink = _all_ink(gray)
    dark = _dark_ink(gray)

    # Any non-background tone may be wall fill (gray walls, lighter junction squares);
    # thickness, not darkness, separates walls from hairline tiling, arcs and fixtures.
    thickness = estimate_wall_thickness(ink)
    core_k = max(3, int(round(thickness * 0.7)))
    core = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((core_k, core_k), np.uint8)) > 0
    core &= ~_inside_text(core, text)
    core = _drop_blobs(core, thickness)

    walls = _segments(core, thickness)
    if not walls:
        warnings.append("No structural walls survived the thickness filter.")
    skew = _skew_degrees(walls)
    if skew is not None and 0.4 < abs(skew) < 10:
        warnings.append(f"Drawing looks rotated by {skew:.1f}°. Straighten the scan for exact orthogonal walls.")

    # Evidence for openings is thin ink only: walls and text removed.
    not_wall = ~(cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~text
    thin = ink & not_wall
    straight_h, straight_v = _straight_lines(thin, int(thickness * 3))
    # Arcs survive; tile grids (H/V) go first, then straight diagonal hairlines such as closet hatching.
    # Order matters: Hough on a raw tile lattice finds phantom diagonals that would erase real arcs.
    curves = thin & ~(straight_h | straight_v)
    curves &= ~_straight_any(curves, int(thickness * 3))
    evidence = {"thin": thin, "curves": curves, "straight_h": straight_h, "straight_v": straight_v, "dark_thin": dark & not_wall}
    openings = find_openings(walls, rasterize_walls(gray.shape, walls), evidence, thickness)

    debug = {
        "01_gray": gray,
        "02_ink_all": _u8(ink),
        "03_ink_dark": _u8(dark),
        "04_text_mask": _u8(text),
        "05_wall_core": _u8(core),
        "06_arc_evidence": _u8(curves),
        "07_straight_thin": _u8(straight_h | straight_v),
    }
    return WallDetection(walls, openings, thickness, core, warnings, debug)


def estimate_wall_thickness(dark: np.ndarray) -> float:
    """Granulometry: wall thickness is where the opening area collapses after thin strokes are gone."""
    u8 = dark.astype(np.uint8)
    areas = [int(u8.sum())]
    ks = list(range(2, 41))
    for k in ks:
        areas.append(int(cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)).sum()))
        if areas[-1] == 0:
            break
    areas_np = np.array(areas, dtype=float)
    drops = areas_np[:-1] - areas_np[1:]  # drops[i]: area lost going from k=i+1 to k=i+2
    # Ignore the hairline drop (k <= 3); the largest later drop is the wall stroke width.
    if len(drops) <= 3 or areas_np[3] == 0:
        return 3.0
    best = int(np.argmax(drops[2:])) + 2
    return float(best + 1.5)


def rasterize_walls(shape: tuple[int, ...], walls: list[WallSegment]) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    for wall in walls:
        _draw_bar(mask, wall.start, wall.end, wall.thickness)
    return mask > 0


def seal_openings(wall_mask: np.ndarray, openings: list[Opening | dict]) -> np.ndarray:
    sealed = wall_mask.astype(np.uint8).copy()
    for o in openings:
        start, end, t = (o["start"], o["end"], o["thickness"]) if isinstance(o, dict) else (o.start, o.end, o.thickness)
        # Overlap both jambs by half a wall so rasterisation never leaves a 1-px leak.
        (x0, y0), (x1, y1) = start, end
        n = max(np.hypot(x1 - x0, y1 - y0), 1e-6)
        ux, uy = (x1 - x0) / n, (y1 - y0) / n
        ext = float(t) / 2 + 1
        _draw_bar(sealed, (x0 - ux * ext, y0 - uy * ext), (x1 + ux * ext, y1 + uy * ext), float(t) + 2)
    return sealed > 0


def find_openings(walls: list[WallSegment], wall_mask: np.ndarray, evidence: dict, thickness: float) -> list[Opening]:
    near_arc = cv2.dilate(evidence["curves"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    count, dark_labels, dark_stats, _ = cv2.connectedComponentsWithStats(evidence["dark_thin"].astype(np.uint8), connectivity=8)
    dark_h, dark_v = _straight_lines(evidence["dark_thin"], int(thickness * 3))
    evidence = {**evidence, "wall_mask": wall_mask, "near_arc": near_arc, "dark_labels": dark_labels, "dark_stats": dark_stats, "dark_h": dark_h, "dark_v": dark_v}
    max_gap = int(thickness * 40)
    found: list[Opening] = []
    ends = _endpoints(walls)
    for (x, y), (dx, dy), seg in ends:
        free = _free_end((x, y), seg, walls, thickness)
        t = seg.thickness
        half = max(1, int(t / 2) - 1)
        # Along the wall: the classic gap between two wall ends.
        gap = _ray_gap(wall_mask, x, y, dx, dy, half, max_gap)
        if gap is not None and gap >= 2 * thickness:
            a = (x, y)
            b = (x + dx * gap, y + dy * gap)
            kind, why, conf = _classify_gap(a, b, (dx, dy), t, thickness, gap, evidence, ends, seg)
            if kind:
                found.append(Opening(a, b, t, kind, why, conf))
        # Sideways from the jamb: a door set in a wall whose stub is no longer than the jamb itself.
        # Only free ends of confident walls: an L-corner already has its perpendicular wall, and
        # short isolated stubs (closet/slider frames) would hinge phantom doors on hatching.
        if not free or seg.confidence < 1:
            continue
        back = (x - dx * t / 2, y - dy * t / 2)
        for sx, sy in ((-dy, dx), (dy, -dx)):
            ox, oy = back[0] + sx * t / 2, back[1] + sy * t / 2
            gap = _ray_gap(wall_mask, ox, oy, sx, sy, half, int(thickness * 9))
            if gap is None or gap < 2.5 * thickness:
                continue
            a = (ox, oy)
            b = (ox + sx * gap, oy + sy * gap)
            kind, why, conf = _classify_gap(a, b, (sx, sy), t, thickness, gap, evidence, ends, seg, door_only=True)
            if kind:
                found.append(Opening(a, b, t, kind, why, conf))
    return _dedupe_openings(found, thickness)


def _classify_gap(a, b, axis, t, thickness, gap, evidence, ends, seg, door_only: bool = False):
    dx, dy = axis
    px, py = -dy, dx  # perpendicular
    # Door swing: a quarter arc hinged at either jamb, on either side, sweeping a clear quarter
    # disc. The leaf is often shorter than the clear gap (jamb/stop), so search radius 0.6–1.05 gap.
    if gap <= 10 * thickness:
        near = evidence["near_arc"]
        best = 0.0
        for r in np.arange(0.6, 1.06, 0.05) * gap:
            for side in (1, -1):
                perp = (px * side, py * side)
                for hinge, toward in ((a, (dx, dy)), (b, (-dx, -dy))):
                    arc = _arc_cover(near, hinge, toward, perp, r)
                    if arc > best and _swing_clear(evidence, hinge, toward, perp, r):
                        best = arc
                # Double door: two half-radius arcs from both jambs, same side.
                left = _arc_cover(near, a, (dx, dy), perp, r / 2)
                right = _arc_cover(near, b, (-dx, -dy), perp, r / 2)
                pair = min(left, right)
                if pair > best and _swing_clear(evidence, a, (dx, dy), perp, r / 2) and _swing_clear(evidence, b, (-dx, -dy), perp, r / 2):
                    best = pair
        if best >= 0.65:
            return "door", f"swing arc {best:.0%}", round(min(0.95, 0.4 + best / 2), 2)
    if door_only:
        return None, "", 0.0
    # Window / sliding door: straight thin lines along the gap inside the wall band, not floor
    # tiling (bands beside the wall must be emptier) and not the edge of a fixture (the dark
    # line must not belong to a shape extending away from the wall, e.g. a bathtub).
    lines = evidence["dark_h"] if dx else evidence["dark_v"]
    band = _band_cover(lines, a, b, t * 0.5)
    if band >= 0.75:
        beside = max(
            _band_cover(lines, _offset(a, px, py, s * t * 1.6), _offset(b, px, py, s * t * 1.6), t * 0.5) for s in (1, -1)
        )
        if beside < band * 0.5 and not _fixture_edge(evidence, a, b, t, horizontal=bool(dx)):
            # A window/slider symbol is two or more parallel lines in the wall band. A single line
            # (railing) only counts on the building edge: inside, it is a floor change or a fixture.
            n = _parallel_lines(lines, a, b, t, horizontal=bool(dx))
            if n >= 2:
                return "window", f"{n} parallel lines {band:.0%}", 0.8
            if _exterior_side(evidence["wall_mask"], a, b, (px, py)):
                return "window", f"railing line on exterior {band:.0%}", 0.65
    # The building envelope is never open: a door-to-window-sized gap whose outer side reaches the
    # image edge is closed as an exterior opening even without a readable symbol.
    if gap <= 15 * thickness and _exterior_side(evidence["wall_mask"], a, b, (px, py)):
        return "window", "gap in exterior wall (symbol unclear)", 0.55
    # Collinear wall end of the same thickness across a door-sized gap: likely opening, no symbol.
    if gap <= 7 * thickness:
        target = (b[0], b[1])
        for (ex, ey), (edx, edy), other in ends:
            if other is seg or (edx, edy) != (-dx, -dy):
                continue
            if abs(other.thickness - t) > thickness * 0.5:
                continue
            if np.hypot(ex - target[0], ey - target[1]) <= thickness * 1.2:
                return "opening", "aligned wall ends, no door symbol", 0.45
    return None, "", 0.0


def _fixture_edge(evidence: dict, a, b, t: float, horizontal: bool) -> bool:
    """True when the dark line in the gap belongs to a shape reaching well beyond the wall band
    across the middle of the gap (a bathtub edge), not merely at its jambs (door frames)."""
    labels = evidence["dark_labels"]
    h, w = labels.shape
    x0, x1 = sorted((int(a[0]), int(b[0])))
    y0, y1 = sorted((int(a[1]), int(b[1])))
    half = int(t / 2) + 1
    reach = int(t * 2.5)
    if horizontal:
        inner = slice(x0 + int(t), max(x0 + int(t) + 1, x1 - int(t)))
        band = labels[max(0, y0 - half) : y0 + half + 1, inner]
        outside = np.concatenate([labels[max(0, y0 - reach) : max(0, y0 - half), inner].ravel(), labels[y0 + half + 1 : y0 + reach, inner].ravel()])
    else:
        inner = slice(y0 + int(t), max(y0 + int(t) + 1, y1 - int(t)))
        band = labels[inner, max(0, x0 - half) : x0 + half + 1]
        outside = np.concatenate([labels[inner, max(0, x0 - reach) : max(0, x0 - half)].ravel(), labels[inner, x0 + half + 1 : x0 + reach].ravel()])
    ids = set(np.unique(band[band > 0]).tolist())
    if not ids:
        return False
    span = max(x1 - x0, y1 - y0, 1)
    shared = np.isin(outside, list(ids)).sum()
    return shared > span * 0.25


def _straight_any(thin: np.ndarray, length: int) -> np.ndarray:
    """Straight hairlines at any angle (closet hatching, counters). A door arc's chord never reaches `length` straight."""
    lines = cv2.HoughLinesP(thin.astype(np.uint8) * 255, 1, np.pi / 180, threshold=length, minLineLength=length, maxLineGap=1)
    out = np.zeros(thin.shape, np.uint8)
    for x1, y1, x2, y2 in np.asarray(lines if lines is not None else []).reshape(-1, 4):
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 1, 3)
    return out > 0


def _straight_lines(thin: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    u8 = thin.astype(np.uint8)
    h = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))) > 0
    v = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))) > 0
    return h, v


def _swing_clear(evidence: dict, hinge, along, perp, radius: float) -> bool:
    """The quarter disc a door sweeps holds no wall and little dark ink (closet hatching, fixtures).
    Sampled away from the hinge and the leaf, which are drawn thick and dark."""
    wall, dark = evidence["wall_mask"], evidence["dark_thin"]
    h, w = wall.shape
    hits = total = 0
    for f in (0.4, 0.55, 0.7):
        for theta in np.radians(np.linspace(15, 60, 5)):
            c, s = np.cos(theta), np.sin(theta)
            x = int(round(hinge[0] + radius * f * (along[0] * c + perp[0] * s)))
            y = int(round(hinge[1] + radius * f * (along[1] * c + perp[1] * s)))
            if not (0 <= x < w and 0 <= y < h) or wall[y, x]:
                return False
            total += 1
            hits += bool(dark[y, x])
    return hits / total <= 0.1


def _parallel_lines(lines: np.ndarray, a, b, t: float, horizontal: bool) -> int:
    """Distinct parallel strokes across the wall band over the middle of the gap."""
    x0, x1 = sorted((int(a[0]), int(b[0])))
    y0, y1 = sorted((int(a[1]), int(b[1])))
    half = int(t / 2) + 1
    if horizontal:
        mid = slice(x0 + int(t), max(x0 + int(t) + 1, x1 - int(t)))
        patch = lines[max(0, y0 - half) : y0 + half + 1, mid]
        profile = patch.mean(axis=1)
    else:
        mid = slice(y0 + int(t), max(y0 + int(t) + 1, y1 - int(t)))
        patch = lines[mid, max(0, x0 - half) : x0 + half + 1]
        profile = patch.mean(axis=0)
    # Each stroke must span the gap: a counter edge running beside a floor line is not a window.
    return len(_runs(np.where(profile >= 0.7)[0]))


def _exterior_side(wall_mask: np.ndarray, a, b, perp) -> bool:
    """Walking away from the gap on one side reaches the image edge without crossing a wall."""
    h, w = wall_mask.shape
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    for side in (1, -1):
        for frac in (0.25, 0.5, 0.75):
            x = a[0] + (b[0] - a[0]) * frac
            y = a[1] + (b[1] - a[1]) * frac
            blocked = False
            step = 0
            while True:
                step += 1
                xi = int(round(x + perp[0] * side * step))
                yi = int(round(y + perp[1] * side * step))
                if not (0 <= xi < w and 0 <= yi < h):
                    break
                if step > 3 and wall_mask[yi, xi]:
                    blocked = True
                    break
            if blocked:
                break
        else:
            return True
    return False


def _arc_cover(near_arc: np.ndarray, hinge, along, perp, radius: float) -> float:
    """Fraction of a quarter circle (from `along` to `perp` around `hinge`) lying on curved thin ink.
    The ends (where the arc meets the jamb and the leaf) are skipped."""
    h, w = near_arc.shape
    hits = 0
    angles = np.radians(np.linspace(8, 82, 24))
    for theta in angles:
        c, s = np.cos(theta), np.sin(theta)
        x = int(round(hinge[0] + radius * (along[0] * c + perp[0] * s)))
        y = int(round(hinge[1] + radius * (along[1] * c + perp[1] * s)))
        if not (0 <= x < w and 0 <= y < h):
            return 0.0
        hits += bool(near_arc[y, x])
    return hits / len(angles)


def _band_cover(mask: np.ndarray, a, b, half: float) -> float:
    h, w = mask.shape
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
    if n < 2:
        return 0.0
    horizontal = abs(b[0] - a[0]) >= abs(b[1] - a[1])
    hits = 0
    for i in range(n):
        x = a[0] + (b[0] - a[0]) * i / n
        y = a[1] + (b[1] - a[1]) * i / n
        if horizontal:
            y0, y1 = int(y - half), int(y + half) + 1
            col = int(x)
            ok = 0 <= col < w and mask[max(0, y0) : min(h, y1), col].any()
        else:
            x0, x1 = int(x - half), int(x + half) + 1
            row = int(y)
            ok = 0 <= row < h and mask[row, max(0, x0) : min(w, x1)].any()
        hits += bool(ok)
    return hits / n


def _ray_gap(wall_mask: np.ndarray, x: float, y: float, dx: int, dy: int, half: int, max_gap: int) -> int | None:
    """Distance from a wall end to the next wall pixel along the wall axis."""
    h, w = wall_mask.shape
    xi, yi = int(round(x)), int(round(y))
    for d in range(1, max_gap):
        cx, cy = xi + dx * d, yi + dy * d
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        if dx:
            strip = wall_mask[max(0, cy - half) : cy + half + 1, cx]
        else:
            strip = wall_mask[cy, max(0, cx - half) : cx + half + 1]
        if strip.size and strip.mean() >= 0.3:
            return d - 1
    return None


def _free_end(point, seg: WallSegment, walls: list[WallSegment], thickness: float) -> bool:
    """A wall end is a jamb only if no other wall touches it."""
    x, y = point
    for other in walls:
        if other is seg:
            continue
        x0, x1 = sorted((other.start[0], other.end[0]))
        y0, y1 = sorted((other.start[1], other.end[1]))
        pad = other.thickness / 2 + thickness * 0.6
        if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad:
            return False
    return True


def _endpoints(walls: list[WallSegment]):
    out = []
    for seg in walls:
        (x0, y0), (x1, y1) = seg.start, seg.end
        if seg.orientation == "h":
            out.append(((min(x0, x1), y0), (-1, 0), seg))
            out.append(((max(x0, x1), y0), (1, 0), seg))
        elif seg.orientation == "v":
            out.append(((x0, min(y0, y1)), (0, -1), seg))
            out.append(((x0, max(y0, y1)), (0, 1), seg))
    return out


def _dedupe_openings(openings: list[Opening], thickness: float) -> list[Opening]:
    """Both jambs see the same gap; keep the stronger reading once."""
    rank = {"door": 3, "window": 2, "opening": 1}
    kept: list[Opening] = []
    for o in sorted(openings, key=lambda o: (-rank[o.kind], -o.confidence)):
        mid = ((o.start[0] + o.end[0]) / 2, (o.start[1] + o.end[1]) / 2)
        if any(np.hypot(mid[0] - (k.start[0] + k.end[0]) / 2, mid[1] - (k.start[1] + k.end[1]) / 2) < thickness * 2 for k in kept):
            continue
        kept.append(o)
    return kept


def _segments(core: np.ndarray, thickness: float) -> list[WallSegment]:
    u8 = core.astype(np.uint8)
    run = max(int(thickness * 1.3) + 1, 5)
    horiz = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1)))
    vert = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, run)))
    out: list[WallSegment] = []
    out += _axis_segments(horiz, "h", thickness)
    out += _axis_segments(vert, "v", thickness)
    covered = cv2.dilate(((horiz | vert) > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out += _diagonal_segments(core & ~covered, thickness)
    return _score_segments(out, thickness)


def _axis_segments(mask: np.ndarray, orientation: str, thickness: float) -> list[WallSegment]:
    out = []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    for i in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[i, :5])
        along, across = (bw, bh) if orientation == "h" else (bh, bw)
        if along < thickness * 1.2:
            continue
        # A wide component is several parallel walls touching at a junction: split by profile.
        comp = labels[y : y + bh, x : x + bw] == i
        profile = comp.sum(axis=1 if orientation == "h" else 0)
        for lo, hi in _bands(profile, along * 0.35):
            t = float(hi - lo)
            if t < thickness * 0.5:
                continue
            sub = comp[lo:hi, :] if orientation == "h" else comp[:, lo:hi]
            idx = np.where(sub.any(axis=0 if orientation == "h" else 1))[0]
            for a, b in _runs(idx):
                if b - a < thickness * 1.2:
                    continue
                if orientation == "h":
                    cy = y + (lo + hi) / 2
                    out.append(WallSegment((float(x + a), cy), (float(x + b), cy), t, "h"))
                else:
                    cx = x + (lo + hi) / 2
                    out.append(WallSegment((cx, float(y + a)), (cx, float(y + b)), t, "v"))
    return out


def _bands(profile: np.ndarray, min_value: float) -> list[tuple[int, int]]:
    on = np.where(profile >= max(1, min_value))[0]
    if not len(on):
        on = np.where(profile > 0)[0]
    return [(a, b) for a, b in _runs(on)]


def _runs(idx: np.ndarray) -> list[tuple[int, int]]:
    if not len(idx):
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    stops = np.concatenate([idx[breaks], [idx[-1]]]) + 1
    return list(zip(starts.tolist(), stops.tolist()))


def _diagonal_segments(rest: np.ndarray, thickness: float) -> list[WallSegment]:
    """Genuinely diagonal walls: long, straight, wall-thick residue that is not H/V."""
    out = []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < thickness * thickness * 4:
            continue
        pts = np.column_stack(np.where(labels == i))[:, ::-1].astype(np.float32)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(pts)
        long_side, short_side = max(rw, rh), min(rw, rh)
        if long_side < thickness * 5 or short_side > thickness * 1.8:
            continue
        rad = np.radians(angle if rw >= rh else angle + 90)
        dx, dy = np.cos(rad) * long_side / 2, np.sin(rad) * long_side / 2
        deg = abs(np.degrees(np.arctan2(dy, dx))) % 90
        if min(deg, 90 - deg) < 8:
            continue
        out.append(WallSegment((cx - dx, cy - dy), (cx + dx, cy + dy), max(short_side, thickness * 0.5), "d"))
    return out


def _score_segments(walls: list[WallSegment], thickness: float) -> list[WallSegment]:
    """Short, unconnected pieces are low-confidence (shown orange in the correction preview)."""
    for seg in walls:
        touches = 0
        for other in walls:
            if other is seg:
                continue
            if _seg_distance(seg, other) <= thickness:
                touches += 1
        if seg.length < thickness * 3 or touches == 0:
            seg.confidence = 0.5
        if seg.orientation == "d":
            seg.confidence = min(seg.confidence, 0.7)
    return walls


def _seg_distance(a: WallSegment, b: WallSegment) -> float:
    ax0, ax1 = sorted((a.start[0], a.end[0]))
    ay0, ay1 = sorted((a.start[1], a.end[1]))
    bx0, bx1 = sorted((b.start[0], b.end[0]))
    by0, by1 = sorted((b.start[1], b.end[1]))
    dx = max(0.0, bx0 - ax1, ax0 - bx1)
    dy = max(0.0, by0 - ay1, ay0 - by1)
    return float(np.hypot(dx, dy)) - (a.thickness + b.thickness) / 2


def _skew_degrees(walls: list[WallSegment]) -> float | None:
    diag = [w for w in walls if w.orientation == "d"]
    if not diag or len(diag) < len(walls) * 0.5:
        return None
    angles = [np.degrees(np.arctan2(w.end[1] - w.start[1], w.end[0] - w.start[0])) % 90 for w in diag]
    angles = [a if a < 45 else a - 90 for a in angles]
    return float(np.median(angles))


def _dark_ink(gray: np.ndarray) -> np.ndarray:
    """Dark strokes: Otsu over non-background pixels separates light tiling/hatching from walls and lines."""
    nonwhite = gray[gray < 245]
    if nonwhite.size < 50:
        return gray < 128
    t, _ = cv2.threshold(nonwhite.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # A scan with no light texture has one dark mode; Otsu then splits inside it. Keep a sane floor.
    t = max(float(t), 100.0)
    mask = gray < t
    return _remove_specks(mask, 6)


def _all_ink(gray: np.ndarray) -> np.ndarray:
    """Everything visibly darker than the paper. Background level comes from the image, not a constant."""
    paper = float(np.percentile(gray, 90))
    return _remove_specks(gray < paper - 22, 4)


def _text_mask(shape: tuple[int, ...], boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    for x, y, bw, bh in boxes:
        pad = 2
        cv2.rectangle(mask, (int(x) - pad, int(y) - pad), (int(x + bw) + pad, int(y + bh) + pad), 1, -1)
    return mask > 0


def _inside_text(core: np.ndarray, text: np.ndarray) -> np.ndarray:
    """Core components lying entirely inside OCR text boxes (bold titles)."""
    if not text.any():
        return np.zeros_like(core)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    drop = np.zeros_like(core)
    for i in range(1, count):
        comp = labels == i
        if text[comp].mean() > 0.9:
            drop |= comp
    return drop


def _drop_blobs(core: np.ndarray, thickness: float) -> np.ndarray:
    """Thick but compact marks (filled symbols, bold dots) are not walls unless they touch walls."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(core)
    for i in range(1, count):
        bw, bh, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if max(bw, bh) >= thickness * 4 and area >= thickness * thickness * 3:
            keep |= labels == i
    return keep


def _remove_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    good = np.zeros(count, bool)
    good[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return good[labels]


def _draw_bar(mask: np.ndarray, start, end, thickness: float) -> None:
    (x0, y0), (x1, y1) = start, end
    half = thickness / 2
    if abs(y1 - y0) < 0.5:  # horizontal: exact rectangle, no rounded caps
        cv2.rectangle(mask, (int(round(min(x0, x1))), int(round(y0 - half))), (int(round(max(x0, x1))) - 1, int(round(y0 + half)) - 1), 1, -1)
    elif abs(x1 - x0) < 0.5:
        cv2.rectangle(mask, (int(round(x0 - half)), int(round(min(y0, y1)))), (int(round(x0 + half)) - 1, int(round(max(y0, y1))) - 1), 1, -1)
    else:
        cv2.line(mask, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 1, max(1, int(round(thickness))))


def _offset(p, px, py, d):
    return (p[0] + px * d, p[1] + py * d)


def _u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask
