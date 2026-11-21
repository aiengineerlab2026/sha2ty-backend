"""The structural model, built from the photo's spec alone (no AI call, no visual mesh).

The spec (services/furniture_spec.py) gives each component's box in the photo, as fractions of the object's own
front box, and the whole object's estimated size. Each structural component becomes a 3D box in metres (x width,
y height, z depth, front +z, standing on y = 0, centred on x and z); `carpenter.reconstruct` makes the closed
solids, adds the hidden structure a type needs (marked inferred) and writes the GLB. Materials come from the
shared material spec (services/material_spec.py).

Depth is placed by role: fronts (doors, drawers, glass, mirrors) at the front, a back panel and a headboard at the
back, handles proud of the front, panels and shelves through the depth. The photo is one view: every size is an
estimate, and too little structure to rebuild from is reported instead of guessed.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.services import carpenter, material_spec

logger = logging.getLogger(__name__)

BOARD = carpenter.BOARD_CM / 100
HANDLE_PROUD = 0.03  # m a handle stands out from its front
MIN_STRUCTURAL = 2  # fewer located structural parts than this: not enough to build from
MIN_CONFIDENCE = 0.25  # a part read below this is not built (listed as uncertain)

_FRONT = ("door", "drawer", "glass", "mirror", "front", "footboard")
_BACK = ("back panel", "headboard", "back")
_PROUD = ("handle", "knob", "pull")


def _words(component: dict[str, Any]) -> str:
    return f"{component.get('id') or ''} {component.get('name') or ''}".lower().replace("_", " ")


def _has(blob: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}s?\b", blob) for w in words)


def size(details: dict[str, Any] | None) -> tuple[tuple[float, float, float] | None, bool, list[str]]:
    """(width, height, depth) in metres, whether the scale is known (written on the image), and what is uncertain."""
    d = details or {}
    w, h, dep = d.get("width_m"), d.get("height_m"), d.get("depth_m")
    notes: list[str] = []
    known = d.get("dimension_source") == "annotated"
    prop = d.get("proportions")  # width : depth : height
    if not all((w, h, dep)) and prop and len(prop) == 3 and all(p > 0 for p in prop):
        pw, pd, ph = prop
        ref = next(((v, p) for v, p in ((w, pw), (h, ph), (dep, pd)) if v), None)
        if ref:
            unit = ref[0] / ref[1]
            w, h, dep = w or pw * unit, h or ph * unit, dep or pd * unit
            notes.append("Some overall sizes were derived from the photo's proportions.")
        else:
            w, h, dep = pw, ph, pd
            notes.append("No size could be estimated: built at the photo's proportions with the largest side 1 m. Scale unknown.")
    if not all((w, h, dep)):
        return None, False, ["The overall size and proportions could not be read from the photo."]
    if not known:
        notes.append("Overall size estimated from the photo, not measured.")
    return (float(w), float(h), float(dep)), known, notes


def _box3(c: dict[str, Any], W: float, H: float, D: float) -> tuple[list[float], list[float]] | None:
    box = c.get("box")
    if not box or len(box) != 4:
        return None
    x0, y0, x1, y1 = box
    lo_x, hi_x = (x0 - 0.5) * W, (x1 - 0.5) * W
    lo_y, hi_y = (1 - y1) * H, (1 - y0) * H  # the photo's y runs top → down
    dims = c.get("dimensions") or {}
    blob = _words(c)
    depth_share = dims.get("depth")
    if _has(blob, _PROUD):
        lo_z, hi_z = D / 2, D / 2 + HANDLE_PROUD
    elif _has(blob, _FRONT):
        lo_z, hi_z = D / 2 - BOARD, D / 2
    elif _has(blob, _BACK) and not _has(blob, ("backrest",)):
        lo_z, hi_z = -D / 2, -D / 2 + BOARD
    elif _has(blob, ("leg",)):
        side = max(min(hi_x - lo_x, D / 4), BOARD)
        lo_z, hi_z = D / 2 - side, D / 2  # a front leg (the photo shows the front)
    else:
        share = min(max(float(depth_share), 0.02), 1.0) if depth_share else 1.0
        lo_z, hi_z = -D * share / 2, D * share / 2
        if _has(blob, ("backrest",)):
            lo_z, hi_z = -D / 2, -D / 2 + D * (share if depth_share else 0.25)
    return [lo_x, lo_y, lo_z], [hi_x, hi_y, hi_z]


def build(details: dict[str, Any] | None, analysis: dict[str, Any] | None, category: str | None = None, kind: str | None = None) -> tuple[bytes | None, dict[str, Any]]:
    """(the structural GLB or None, its report). The report always says what was built from what, and why not."""
    status: dict[str, Any] = {"source": "vision_spec", "built": False, "uncertain": [], "excludedLoose": []}
    comps = (analysis or {}).get("components") or []
    if not comps:
        status["uncertain"].append("The photo reading has no components: no structural model was built.")
        return None, status
    dims, known, notes = size(details)
    status["uncertain"] += notes
    if dims is None:
        return None, status
    W, H, D = dims
    materials = material_spec.for_analysis(analysis)["parts"]
    parts: list[dict[str, Any]] = []
    for c in comps:
        cid = str(c.get("id") or "")
        if c.get("role") == "decorative":
            status["excludedLoose"].append(str(c.get("name") or cid))
            continue
        if float(c.get("confidence") if c.get("confidence") is not None else 0.5) < MIN_CONFIDENCE:
            status["uncertain"].append(f"{c.get('name')}: read with low confidence, not built.")
            continue
        box = _box3(c, W, H, D)
        if box is None:
            if c.get("visibility") != "inferred":
                status["uncertain"].append(f"{c.get('name')}: no position in the photo, not built.")
            continue
        x0, y0, x1, y1 = c["box"]
        whole = (x1 - x0) >= 0.8 and (y1 - y0) >= 0.8
        carcass = whole and (cid == (analysis or {}).get("bodyId") or _has(_words(c), ("body", "carcass", "frame", "case", "cabinet")))
        parts.append({"name": str(c.get("name") or cid), "type": "", "box": box, "specId": cid, "carcass": carcass,
                      "inferred": c.get("visibility") == "inferred", "material": materials.get(cid)})
    if len(parts) < MIN_STRUCTURAL:
        status["uncertain"].append(f"Only {len(parts)} located structural part(s): too little structure to rebuild without guessing.")
        return None, status
    # Every part already in metres: no rescaling, and the scale's certainty goes with it.
    built = carpenter.reconstruct({"components": parts}, {"scale_known": known, "category": category}, kind, category)
    if not built:
        status["uncertain"].append("No closed solid could be made from the parts.")
        return None, status
    glb, report = built
    status.update(built=True, sizeM={"width": round(W, 3), "height": round(H, 3), "depth": round(D, 3)}, scaleKnown=known)
    return glb, {**report, **status, "materials": materials}
