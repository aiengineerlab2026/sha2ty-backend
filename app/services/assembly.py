"""Constraint-based furniture assembly: physical parts placed by their ROLE in a consistent structure, sized by the
overall measurements, positioned from the perspective-corrected front face. Not photographed rectangles made solid.

  1. front plane    the object's front-face quad in the photo → a homography onto a W × H rectangle (perspective
                    removed). Each part's front-face corners go through it: its true position on the front.
  2. carcass        (storage pieces) left and right sides, top, base, back, plinth and crown trim of ONE carcass:
                    each at its role's place, every board the board thickness (never the photographed width: a side
                    seen at an angle looks wide, it is not thick). Missing sides / top / base / back are inferred one by
                    one and marked estimated; a part already listed under another name is never added twice.
  3. fronts         doors, drawer fronts and fixed panels inside the opening the carcass leaves, edges snapped to the
                    opening and to their neighbours; sliding doors on two tracks (one in front of the other, the
                    same width, overlapping); hinged doors flush with the carcass front.
  4. assemblies     a logical assembly (a mirrored door) is a parent, never a solid: its frame is a frame with a real
                    opening and its pane sits in that opening; mouldings are raised / recessed geometry on their part.
  5. other pieces   (tables, beds, chairs…) parts placed from their front rectangle with their role's depth.

Coordinates: metres, x across (centre 0), y up (floor 0), z depth (front +z). Each placed part: {id, name, kind,
assembly, lo, hi, geometry, opening, mouldings, estimated, spec}.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

BOARD = 0.018
BACK = 0.006
DOOR = 0.018
SLIDING = 0.02
PANE = 0.004
TRACK_GAP = 0.006  # between two sliding tracks
OVERLAP = 0.04  # sliding doors overlap this much
SNAP = 0.05  # an edge this close (share of the size) to the opening or a neighbour meets it
HANDLE_PROUD = 0.025

FRONT_KINDS = {"door", "sliding_door", "drawer_front", "fixed_panel"}
CARCASS_KINDS = {"side_panel", "top_panel", "crown_trim", "base_panel", "plinth", "back_panel"}
INSIDE_KINDS = {"shelf", "divider"}
HARDWARE = {"handle", "knob"}
STORAGE = {"wardrobe", "cabinet", "bookshelf", "tv", "dresser", "desk"}


def _words(c: dict[str, Any]) -> str:
    return f"{c.get('id') or ''} {c.get('name') or ''}".lower().replace("_", " ")


# A reading without part kinds (made before they existed) is placed by its names. Order matters: "door handle" is a
# handle, "mirror door frame" a frame, "plinth drawer" a drawer.
_NAMED = (("handle", r"\b(handle|pull|grip)s?\b"), ("knob", r"\bknobs?\b"), ("door_frame", r"\bframe\b.*\bdoor\b|\bdoor\b.*\bframe\b"),
          ("mirror", r"^(?!.*\bdoor\b).*\bmirror\b"), ("glass", r"^(?!.*\bdoor\b).*\bglass\b"),
          ("drawer_front", r"\bdrawers?\b(?! (pedestal|unit|cabinet|chest))"),
          ("sliding_door", r"\bsliding\b"), ("door", r"\bdoors?\b"), ("shelf", r"\bshel(f|ves)\b"), ("divider", r"\b(divider|partition)\b"),
          ("plinth", r"\b(plinth|kick|skirting)\b"), ("crown_trim", r"\b(crown|cornice|top trim)\b"), ("side_panel", r"\bside\b"),
          ("top_panel", r"\btop panel\b|^top\b|\btop\b(?!.*\btable)"), ("base_panel", r"\b(base|bottom) panel\b"), ("back_panel", r"\bback panel\b"),
          ("tabletop", r"\b(tabletop|table top|desk top|worktop)\b"), ("leg", r"\blegs?\b"), ("headboard", r"\bheadboard\b"), ("footboard", r"\bfootboard\b"),
          ("seat", r"\bseat\b"), ("backrest", r"\bback ?rest\b"), ("armrest", r"\barm ?rest|\barms?\b"), ("rail", r"\b(rail|apron)\b"),
          ("carcass", r"\b(body|carcass|cabinet|case|frame)\b"))


def kind_of(c: dict[str, Any]) -> str:
    """The part's kind: the reading's, else from its name."""
    if c.get("kind") and c.get("kind") != "other":
        return c["kind"]
    words = _words(c)
    for kind, pattern in _NAMED:
        if re.search(pattern, words):
            if kind == "carcass" and not (c.get("box") and (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]) >= 0.6):
                continue  # only the whole body is the carcass; a "frame" elsewhere stays a part
            return kind
    return "other"


# ---- 1. perspective: the photo's front face onto the front plane ----


def homography(quad: list[list[float]]) -> np.ndarray:
    """Image points (fractions) → front-plane fractions (u right, v down, 0..1 over the object's front face)."""
    import cv2

    src = np.array(quad, np.float32)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_front(H: np.ndarray, pts) -> np.ndarray:
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def front_rect(c: dict[str, Any], H: np.ndarray | None, obj_box: list[float] | None) -> list[float] | None:
    """[u0, v0, u1, v1] of the part's front face on the front plane (0..1), from its corners when given (perspective
    corrected), else from its box in the object's box (as photographed)."""
    if H is not None and c.get("quad"):
        q = to_front(H, c["quad"])
        u0, u1 = (q[0, 0] + q[3, 0]) / 2, (q[1, 0] + q[2, 0]) / 2
        v0, v1 = (q[0, 1] + q[1, 1]) / 2, (q[3, 1] + q[2, 1]) / 2
        return [float(min(u0, u1)), float(min(v0, v1)), float(max(u0, u1)), float(max(v0, v1))]
    if c.get("box"):
        return list(c["box"])
    return None


def front_quad(analysis: dict[str, Any], details: dict[str, Any]) -> list[list[float]] | None:
    """The object's front face in the image: the reading's, else the bounding box of its parts' front faces."""
    if analysis.get("frontQuad"):
        return analysis["frontQuad"]
    quads = [p for c in analysis.get("components") or [] if c.get("quad") and c.get("kind") in FRONT_KINDS | CARCASS_KINDS for p in c["quad"]]
    if len(quads) >= 4:
        xs, ys = [p[0] for p in quads], [p[1] for p in quads]
        return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
    return None


# ---- 2–4. the assembly ----


def _part(c: dict[str, Any] | None, pid: str, name: str, kind: str, lo, hi, **extra) -> dict[str, Any]:
    return {"id": pid, "name": name, "kind": kind, "lo": np.asarray(lo, float), "hi": np.asarray(hi, float), "spec": c,
            "assembly": (c or {}).get("assembly") or extra.pop("assembly", None), "geometry": extra.pop("geometry", "panel"),
            "estimated": extra.pop("estimated", False), **extra}


def _snap(value: float, targets: list[float], tol: float) -> float:
    near = [t for t in targets if abs(t - value) <= tol]
    return min(near, key=lambda t: abs(t - value)) if near else value


def is_storage(family: str, comps: list[dict[str, Any]]) -> bool:
    kinds = {c.get("kind") for c in comps}
    return family in STORAGE or bool(kinds & (CARCASS_KINDS | {"door", "sliding_door", "drawer_front", "shelf"})) and "tabletop" not in kinds


def assemble(analysis: dict[str, Any], details: dict[str, Any], size: tuple[float, float, float], family: str) -> dict[str, Any]:
    """{"parts": physical parts placed, "assemblies": logical parents, "notes": what was inferred or uncertain,
    "frontPlane": the homography's quad}."""
    W, H, D = size
    comps = [{**c, "kind": kind_of(c)} for c in analysis.get("components") or []]
    by_id = {c["id"]: c for c in comps}
    # A part that other parts name as their assembly is logical (the door of a frame and a mirror), whatever its flag.
    parents = {c.get("assembly") for c in comps if c.get("assembly")}
    logical = {c["id"] for c in comps if c.get("logical") or c["id"] in parents or c.get("kind") == "carcass"}
    quad = front_quad(analysis, details)
    Hm = homography(quad) if quad else None
    obj_box = details.get("bbox")
    notes: list[str] = []

    def rect(c):
        r = front_rect(c, Hm, obj_box)
        return None if r is None else [min(max(v, -0.05), 1.05) for v in r]

    def x_of(u):
        return -W / 2 + u * W

    def y_of(v):
        return H - v * H

    physical = [c for c in comps if c["id"] not in logical and c.get("complexity") != "complex" and c.get("decoration") not in ("carved", "upholstered")]
    parts: list[dict[str, Any]] = []
    assemblies = [{"id": c["id"], "name": c.get("name") or c["id"], "kind": c.get("kind"), "motion": c.get("motion"), "assembly": c.get("assembly"), "spec": c}
                  for c in comps if c["id"] in logical]
    if not is_storage(family, comps):
        return {"parts": _loose(physical, rect, W, H, D), "assemblies": assemblies, "notes": notes, "frontPlane": quad}

    kinds: dict[str, list[dict[str, Any]]] = {}
    for c in physical:
        kinds.setdefault(c.get("kind") or "other", []).append(c)

    # ---- carcass: one structure, each board at its role's place ----
    plinth = (kinds.get("plinth") or [None])[0]
    crown = (kinds.get("crown_trim") or [None])[0]
    top_seen = next((c for c in kinds.get("top_panel", []) if c.get("visibility") != "inferred" and rect(c)), None)
    crown = crown or top_seen  # a visible top trim band is the crown's face
    plinth_h = float(np.clip((rect(plinth)[3] - rect(plinth)[1]) * H, 0.03, 0.2 * H)) if plinth and rect(plinth) else 0.0
    crown_h = float(np.clip((rect(crown)[3] - rect(crown)[1]) * H, BOARD, 0.15 * H)) if crown and rect(crown) else BOARD
    front_z = D / 2
    # The front zone: the depth the fronts occupy at the front of the carcass. Top, base, back and shelves stop behind it
    # so no door ever runs through them; the sides frame it, the crown and plinth close it above and below.
    two_tracks = sum(1 for c in comps if c.get("kind") == "sliding_door" or (c.get("motion") or {}).get("type") == "slide") >= 2
    zone = (2 * SLIDING + TRACK_GAP) if two_tracks else DOOR
    inner_front = front_z - zone
    sides = {"left": None, "right": None}
    for c in kinds.get("side_panel", []):
        word = _words(c)
        side = "left" if "left" in word else "right" if "right" in word else None
        if side is None and rect(c):
            side = "left" if (rect(c)[0] + rect(c)[2]) / 2 < 0.5 else "right"
        if side and sides[side] is None:
            sides[side] = c
    for side, sign in (("left", -1), ("right", 1)):
        c = sides[side]
        x0, x1 = (-W / 2, -W / 2 + BOARD) if sign < 0 else (W / 2 - BOARD, W / 2)
        inferred = c is None or c.get("visibility") == "inferred"
        parts.append(_part(c, c["id"] if c else f"side_{side}", (c or {}).get("name") or f"{side.title()} side panel", "side_panel",
                           [x0, 0, -D / 2], [x1, H, front_z], estimated=inferred, assembly="carcass"))
        if inferred:
            notes.append(f"{side.title()} side panel inferred (not seen).")
    top = (kinds.get("top_panel") or [None])[0]
    parts.append(_part(top, top["id"] if top else "top_panel", (top or {}).get("name") or "Top panel", "top_panel",
                       [-W / 2 + BOARD, H - BOARD, -D / 2], [W / 2 - BOARD, H, inner_front],
                       estimated=top is None or top.get("visibility") == "inferred", assembly="carcass"))
    if crown_h > BOARD:  # the visible trim band across the top front
        parts.append(_part(crown if crown is not top else None, (crown["id"] if crown is not top else "crown_trim"), "Crown trim", "crown_trim",
                           [-W / 2 + BOARD, H - crown_h, front_z - BOARD], [W / 2 - BOARD, H - (BOARD if crown_h > 2 * BOARD else 0), front_z], assembly="carcass",
                           material_from=crown["id"] if crown else None))
    base = (kinds.get("base_panel") or [None])[0]
    base_y = plinth_h
    parts.append(_part(base, base["id"] if base else "base_panel", (base or {}).get("name") or "Base panel", "base_panel",
                       [-W / 2 + BOARD, base_y, -D / 2], [W / 2 - BOARD, base_y + BOARD, inner_front],
                       estimated=base is None or base.get("visibility") == "inferred", assembly="carcass"))
    if plinth_h:
        parts.append(_part(plinth, plinth["id"], plinth.get("name") or "Plinth", "plinth", [-W / 2 + BOARD, 0, front_z - BOARD], [W / 2 - BOARD, base_y, front_z], assembly="carcass"))
    back = (kinds.get("back_panel") or [None])[0]
    parts.append(_part(back, back["id"] if back else "back_panel", (back or {}).get("name") or "Back panel", "back_panel",
                       [-W / 2 + BOARD, base_y + BOARD, -D / 2], [W / 2 - BOARD, H - BOARD, -D / 2 + BACK],
                       estimated=back is None or back.get("visibility") == "inferred", assembly="carcass"))

    # ---- the opening the fronts fill ----
    ox0, ox1 = -W / 2 + BOARD, W / 2 - BOARD
    oy0, oy1 = base_y + BOARD if not plinth_h else plinth_h, H - crown_h
    fronts: list[tuple[dict[str, Any], list[float]]] = []
    for c in comps:  # a front is a physical front part, or a logical door assembly (its frame and pane are placed in it)
        is_front = c.get("kind") in FRONT_KINDS and (c["id"] in logical or c in physical)
        if is_front and rect(c):
            fronts.append((c, rect(c)))
    sliding = [c for c, _ in fronts if c.get("kind") == "sliding_door" or (c.get("motion") or {}).get("type") == "slide"]
    xs_targets = [ox0, ox1]
    placed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if len(sliding) >= 2 and len(sliding) == len([c for c, _ in fronts if c.get("kind") in ("door", "sliding_door")]):
        # Sliding doors: the same width, overlapping, on alternating tracks; the photo's order left to right.
        order = sorted(sliding, key=lambda c: rect(c)[0])
        n = len(order)
        width = (ox1 - ox0 + (n - 1) * OVERLAP) / n
        visible = [rect(c)[2] - rect(c)[0] for c in order]
        front_first = int(np.argmax(visible[:2])) if n == 2 else 0  # the wider-looking one is on the front track
        for i, c in enumerate(order):
            x0 = ox0 + i * (width - OVERLAP)
            on_front = (i % 2) == (front_first % 2)
            z1 = front_z - (0 if on_front else SLIDING + TRACK_GAP)
            placed[c["id"]] = (np.array([x0, oy0, z1 - SLIDING]), np.array([x0 + width, oy1, z1]))
        notes.append(f"{n} sliding doors: equal widths ({width:.2f} m), overlapping {OVERLAP * 100:.0f} cm on two tracks.")
    seams: dict[tuple[str, str], float] = {}
    for c, r in fronts:  # neighbours side by side meet at the middle of the seam between them
        for o, q in fronts:
            if o is c or abs(q[0] - r[2]) > SNAP or min(r[3], q[3]) - max(r[1], q[1]) < 0.3 * (r[3] - r[1]):
                continue
            seams[(c["id"], "right")] = seams[(o["id"], "left")] = x_of((r[2] + q[0]) / 2)
    for c, r in fronts:
        if c["id"] in placed:
            continue
        x0 = seams.get((c["id"], "left"), _snap(x_of(r[0]), xs_targets, SNAP * W))
        x1 = seams.get((c["id"], "right"), _snap(x_of(r[2]), xs_targets, SNAP * W))
        y1 = _snap(y_of(r[1]), [oy1], SNAP * H)
        y0 = _snap(y_of(r[3]), [oy0], SNAP * H)
        x0, x1 = max(min(x0, x1), ox0), min(max(x0, x1), ox1)
        y0, y1 = max(min(y0, y1), oy0), min(max(y0, y1), oy1)
        placed[c["id"]] = (np.array([x0 + 0.001, y0 + 0.001, front_z - DOOR]), np.array([x1 - 0.001, y1 - 0.001, front_z]))
    for c, r in fronts:
        lo, hi = placed[c["id"]]
        kind = c.get("kind") or "door"
        children = [k for k in physical if k.get("assembly") == c["id"]]
        if c["id"] in logical and children:
            frame = next((k for k in children if k.get("kind") == "door_frame" or k.get("geometry") == "frame"), None)
            panes = [k for k in children if k.get("kind") in ("mirror", "glass") or k.get("geometry") == "pane"]
            opening = None
            if panes and rect(panes[0]):
                pr, fr = rect(panes[0]), r
                fx = lambda u: lo[0] + (u - fr[0]) / max(fr[2] - fr[0], 1e-6) * (hi[0] - lo[0])  # noqa: E731
                fy = lambda v: hi[1] - (v - fr[1]) / max(fr[3] - fr[1], 1e-6) * (hi[1] - lo[1])  # noqa: E731
                border = 0.02
                opening = [max(fx(pr[0]), lo[0] + border), max(fy(pr[3]), lo[1] + border), min(fx(pr[2]), hi[0] - border), min(fy(pr[1]), hi[1] - border)]
            frame_spec = frame or c
            parts.append(_part(frame_spec, frame["id"] if frame else f"{c['id']}_frame", (frame or {}).get("name") or f"{c.get('name') or c['id']} frame",
                               "door_frame", lo, hi, geometry="frame" if opening else "panel", opening=opening, assembly=c["id"],
                               mouldings=(frame or {}).get("mouldings") or [], mover_of=c["id"]))
            for pane in panes:
                if not opening:
                    break
                ox_0, oy_0, ox_1, oy_1 = opening
                mid = (lo[2] + hi[2]) / 2
                parts.append(_part(pane, pane["id"], pane.get("name") or "Mirror", pane.get("kind") or "mirror",
                                   [ox_0, oy_0, mid - PANE / 2], [ox_1, oy_1, mid + PANE / 2], geometry="pane", assembly=c["id"]))
                break  # one pane fills one opening; a second listed pane is the same glass seen twice
        else:
            parts.append(_part(c, c["id"], c.get("name") or c["id"], kind, lo, hi, mouldings=c.get("mouldings") or [], mover_of=c["id"],
                               assembly=c.get("assembly")))

    # ---- hardware and inside parts ----
    hosts = {p["id"]: p for p in parts}
    for c in kinds.get("handle", []) + kinds.get("knob", []):
        r = rect(c)
        if not r:
            continue
        cx, cy = x_of((r[0] + r[2]) / 2), y_of((r[1] + r[3]) / 2)
        on = lambda p: p["lo"][0] <= cx <= p["hi"][0] and p["lo"][1] <= cy <= p["hi"][1]  # noqa: E731
        # The reading's relation only when the handle really lies on that part (a handle by the line between two doors
        # can be tied to the wrong one); else the front it lies on.
        said = next((hosts.get(t) for t in [x.get("target") for x in c.get("relations") or []] + [c.get("assembly")] if t and t in hosts), None)
        host = said if said is not None and on(said) else next((p for p in parts if p.get("mover_of") and on(p)), None)
        z0 = host["hi"][2] if host else front_z
        parts.append(_part(c, c["id"], c.get("name") or "Handle", c.get("kind"), [x_of(r[0]), y_of(r[3]), z0], [x_of(r[2]), y_of(r[1]), z0 + HANDLE_PROUD],
                           host=host["id"] if host else None, assembly=(host or {}).get("assembly")))
    for c in [*kinds.get("shelf", []), *kinds.get("divider", [])]:
        r = rect(c)
        if not r:
            continue
        x0, x1 = max(x_of(r[0]), ox0), min(x_of(r[2]), ox1)
        if c.get("kind") == "shelf":
            y = y_of((r[1] + r[3]) / 2)
            lo, hi = [x0, y - BOARD / 2, -D / 2 + BACK], [x1, y + BOARD / 2, inner_front - 0.005]
        else:
            x = (x0 + x1) / 2
            lo, hi = [x - BOARD / 2, oy0, -D / 2 + BACK], [x + BOARD / 2, oy1, inner_front - 0.005]
        parts.append(_part(c, c["id"], c.get("name") or c["id"], c.get("kind"), lo, hi, assembly="carcass"))
    # anything else listed and physical (a mirror on its own, a fixed panel…) that has not been placed
    done = {p["id"] for p in parts}
    for c in physical:
        if c["id"] in done or c.get("kind") in CARCASS_KINDS | HARDWARE | INSIDE_KINDS or (c.get("assembly") in logical and c.get("kind") in ("door_frame", "mirror", "glass")):
            continue
        r = rect(c)
        if r and c.get("kind") not in FRONT_KINDS:
            notes.append(f"{c.get('name')}: not placed (its role in the structure is unclear).")
    return {"parts": parts, "assemblies": [{"id": "carcass", "name": "Carcass", "kind": "carcass"}, *assemblies], "notes": notes, "frontPlane": quad}


# Depth by role for pieces that are not a carcass (tables, beds, chairs…).
_DEPTH = {"tabletop": "full", "seat": "full", "headboard": "back", "footboard": "front", "backrest": "back", "leg": "post", "rail": "front",
          "armrest": "full", "shelf": "full", "door": "front", "drawer_front": "front", "handle": "proud", "knob": "proud"}


def _loose(physical: list[dict[str, Any]], rect, W: float, H: float, D: float) -> list[dict[str, Any]]:
    """Parts of a non-carcass piece: each at its front rectangle, with its role's depth and the board thickness for
    flat parts (a headboard is a board at the back, not as deep as its photographed rectangle)."""
    out = []
    for c in physical:
        r = rect(c)
        if not r:
            continue
        x0, x1 = -W / 2 + r[0] * W, -W / 2 + r[2] * W
        y0, y1 = H - r[3] * H, H - r[1] * H
        role = _DEPTH.get(c.get("kind") or "", "full")
        if role == "back":
            z0, z1 = -D / 2, -D / 2 + BOARD * 2
        elif role == "front":
            z0, z1 = D / 2 - BOARD, D / 2
        elif role == "post":
            side = max(min(x1 - x0, D / 4), BOARD * 2)
            z0, z1 = D / 2 - side, D / 2
        elif role == "proud":
            z0, z1 = D / 2, D / 2 + HANDLE_PROUD
        else:
            z0, z1 = -D / 2, D / 2
            if (c.get("kind") in ("tabletop", "shelf")) and y1 - y0 > 0.06:
                y0 = y1 - max(BOARD * 2, 0.025)  # a top is a board, however tall its photographed edge looks
        out.append(_part(c, c["id"], c.get("name") or c["id"], c.get("kind") or "other", [x0, y0, z0], [x1, y1, z1], mouldings=c.get("mouldings") or [],
                         mover_of=c["id"] if (c.get("motion") or {}).get("type") not in (None, "none") else None))
    # A full-depth part never runs through a board on its front or back (a bed frame and its footboard, a pedestal and
    # its drawer front): it stops at that board.
    for a in out:
        if _DEPTH.get(a["kind"], "full") != "full":
            continue
        for b in out:
            role = _DEPTH.get(b["kind"], "full")
            if b is a or role not in ("front", "back"):
                continue
            ox = min(a["hi"][0], b["hi"][0]) - max(a["lo"][0], b["lo"][0])
            oy = min(a["hi"][1], b["hi"][1]) - max(a["lo"][1], b["lo"][1])
            if ox > 0.3 * (b["hi"][0] - b["lo"][0]) and oy > 0.3 * (b["hi"][1] - b["lo"][1]):
                if role == "front":
                    a["hi"][2] = min(a["hi"][2], b["lo"][2])
                else:
                    a["lo"][2] = max(a["lo"][2], b["hi"][2])
    # legs seen at the front stand at the back too (a table's four legs): mirrored where the reading lists none there
    legs = [p for p in out if p["kind"] == "leg"]
    if legs and not any(p["lo"][2] < 0 for p in legs):
        for p in list(legs):
            back = dict(p)
            back.update(id=f"{p['id']}_back", name=f"{p['name']} (back)", estimated=True,
                        lo=np.array([p["lo"][0], p["lo"][1], -p["hi"][2]]), hi=np.array([p["hi"][0], p["hi"][1], -p["lo"][2]]))
            out.append(back)
    return out


def stable_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
