"""Movable parts: the animation configuration of a generated model, and its estimated interior.

  component document (meshes matched to the vision analysis, mesh_analysis.match)
    └─ configure() ─▶ doc["animation"] = {animations: [...], flagged: [...], interior: {...}}
                    └ estimated shelves / rails / dividers / drawer boxes appended to the GLB as their own parts

Every animation is a rigid move of ONE separate mesh (and the estimated parts that travel with it, `follow`), in the
model's own coordinates (glTF scene, y up, +z front as Meshy writes it):
  rotation:    {"axis": "x"|"y"|"z", "pivot": [x, y, z], "from": 0, "to": ±degrees}
  translation: {"axis": "x"|"y"|"z", "distance": ±metres (model units)}
Geometry is never changed; a movement that cannot be done that way (a fused mesh, the part is the whole model, the
part shares its mesh with others, the photo does not show how it moves) is listed in `flagged` and left disabled
for the vendor to correct, never forced. The Three.js editor plays and edits these (frontend lib/vendor/animation.ts).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.services.glb_parts import append_boxes

logger = logging.getLogger(__name__)

# Below: detected but disabled, flagged for the vendor. Low on purpose: the vision model gives a door it cannot see
# hinges on ~0.4, yet it is still a door; the viewer settles hinges and handles from the real meshes (frontend refine).
MIN_CONFIDENCE = 0.3
INTERIOR_CONFIDENCE = 0.6  # estimated interior parts are only added when the analysis is sure
DEFAULT_LIMIT = {"rotate": 95.0, "fold": 90.0, "lift": 80.0}
DEFAULT_TRAVEL = 0.5
DURATION = {"rotation": 0.8, "translation": 0.6}

# hinge edge → rotation axis, the pivot on the part's box (0 = min, 1 = max, 0.5 = centre per axis), and the angle's
# sign that swings the part towards the front (+z) / up, away from the body.
HINGES = {
    "left": ("y", (0, 0.5, 1), -1),
    "right": ("y", (1, 0.5, 1), 1),
    "bottom": ("x", (0.5, 0, 1), 1),
    "top": ("x", (0.5, 1, 1), -1),
    "back": ("x", (0.5, 1, 0), -1),
    "front": ("x", (0.5, 1, 1), 1),
}
DIRECTIONS = {"front": ("z", 1), "back": ("z", -1), "right": ("x", 1), "left": ("x", -1), "up": ("y", 1), "down": ("y", -1)}
AXIS = {"x": 0, "y": 1, "z": 2}
ACCESSORY = {"handle", "handles", "knob", "pull", "hinge", "lock", "key", "keyhole", "ring", "blade", "tooth", "backing", "surround",
             "trim", "strip", "moulding", "molding", "decor", "decoration", "ornament", "escutcheon", "plate", "interior", "cavity", "box"}
INTERIOR_AR = {"shelf": "رف", "rail": "قضيب تعليق", "divider": "فاصل", "drawer_box": "صندوق درج"}
INTERIOR_EN = {"shelf": "Shelf", "rail": "Hanging rail", "divider": "Divider", "drawer_box": "Drawer box"}


def _whole(components: list[dict[str, Any]]) -> tuple[list[float], list[float]] | None:
    boxes = [c["box"] for c in components if c.get("box")]
    if not boxes:
        return None
    return [min(b[0][i] for b in boxes) for i in range(3)], [max(b[1][i] for b in boxes) for i in range(3)]


def _guess(c: dict[str, Any]) -> dict[str, Any] | None:
    """Without a vision analysis (the built wardrobe, GPT models): doors and drawers by their ids and names. A door
    whose side is not in its name is a guess (low confidence: disabled until the vendor checks it)."""
    words = set(re.split(r"[^a-z]+", f"{c.get('id', '')} {c.get('name', '')}".lower()))
    if words & ACCESSORY:  # on a door, not a door: "door lock key ring", "mirror door backing", "drawer box"
        return None
    if "drawer" in words or "drawers" in words:
        return {"type": "slide", "direction": "front", "travel": DEFAULT_TRAVEL, "confidence": 0.8, "reason": "named a drawer", "revealsInterior": True}
    if "door" in words or "doors" in words or "leaf" in words:
        side = "left" if "left" in words else "right" if "right" in words else None
        # No side in the name: the viewer takes it from the geometry (the handle, the other door of a pair).
        return {"type": "rotate", "hinge": side or "left", "limit": DEFAULT_LIMIT["rotate"], "confidence": 0.8 if side else 0.7,
                "reason": "named a door" + ("" if side else "; hinge side from the geometry"), "revealsInterior": True}
    return None


def _entry(c: dict[str, Any], motion: dict[str, Any], whole: tuple[list[float], list[float]], label: str, label_ar: str | None, source: str) -> dict[str, Any]:
    (lo, hi), issues = c["box"], []
    kind = motion.get("type")
    confidence = float(motion.get("confidence") or 0)
    entry: dict[str, Any] = {"id": f"anim_{c['id']}", "component": c["id"], "follow": [], "label": label, "labelAr": label_ar, "motion": kind, "source": source,
                             "confidence": round(confidence, 2), "reason": motion.get("reason") or "", "revealsInterior": bool(motion.get("revealsInterior"))}
    if kind in ("rotate", "fold", "lift"):
        hinge = motion.get("hinge")
        if hinge not in HINGES:
            hinge = {"fold": "bottom", "lift": "back"}.get(kind, "left")
            issues.append(f"The hinge side was not detected; {hinge} is assumed.")
        axis, at, sign = HINGES[hinge]
        limit = float(motion.get("limit") or DEFAULT_LIMIT[kind])
        if not 5 <= limit <= 180:
            issues.append(f"An opening of {limit:g}° is not realistic; limited to 5–180°.")
            limit = min(max(limit, 5.0), 180.0)
        pivot = [round(lo[i] + (hi[i] - lo[i]) * at[i], 4) for i in range(3)]
        entry.update(type="rotation", hinge=hinge, axis=axis, pivot=pivot, **{"from": 0.0, "to": sign * limit}, unit="degrees", duration=DURATION["rotation"])
    else:
        direction = motion.get("direction")
        if direction not in DIRECTIONS:
            direction = "back" if kind == "retract" else "front"
            issues.append(f"The direction was not detected; {direction} is assumed.")
        axis, sign = DIRECTIONS[direction]
        travel = float(motion.get("travel") or DEFAULT_TRAVEL)
        if not 0.02 <= travel <= 1:
            issues.append("The travel was outside the object's size; limited to it.")
            travel = min(max(travel, 0.02), 1.0)
        span = whole[1][AXIS[axis]] - whole[0][AXIS[axis]]
        entry.update(type="translation", direction=direction, axis=axis, distance=round(sign * travel * span, 4), unit="meters", duration=DURATION["translation"])
    if confidence < MIN_CONFIDENCE:
        issues.append(f"Uncertain ({confidence:.0%}): the photo does not clearly show how it moves.")
    entry["issues"] = issues
    entry["enabled"] = confidence >= MIN_CONFIDENCE
    return entry


def _blocked(c: dict[str, Any] | None, components: list[dict[str, Any]], whole, separated: bool) -> str | None:
    """Why this part cannot move on its own (None: it can)."""
    if c is None:
        return "It is not a separate mesh in the generated model."
    if not separated or len(components) < 2:
        return "The model is one fused mesh: moving it would move the whole object."
    if not c.get("box"):
        return "Its size could not be read from the file."
    (lo, hi), (wlo, whi) = c["box"], whole
    if all((hi[i] - lo[i]) >= 0.9 * (whi[i] - wlo[i]) for i in range(3)):
        return "This mesh is (almost) the whole model: moving it would move everything."
    shared = [o["name"] for o in components if o is not c and o.get("mesh") == c.get("mesh") and o.get("node") == c.get("node")]
    if shared:
        return f"It shares one mesh with {', '.join(shared[:3])}: they would move with it."
    return None


def _interior_boxes(interior: dict[str, Any], entries: list[dict[str, Any]], components: list[dict[str, Any]], whole, color: str | None) -> list[dict[str, Any]]:
    """Simple estimated parts inside the body: evenly spaced shelves, a rail high up, dividers, and a box behind
    each sliding drawer front (it travels with it). Sizes are fractions of the body: no detailed layout."""
    body = max((c for c in components if c.get("box")), key=lambda c: (c["box"][1][0] - c["box"][0][0]) * (c["box"][1][1] - c["box"][0][1]), default=None)
    (blo, bhi) = body["box"] if body else whole
    w, h, d = (bhi[i] - blo[i] for i in range(3))
    wall = 0.03 * min(w, h)
    by_analysis = {c.get("analysisId"): c for c in components if c.get("analysisId")}
    out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    def add(kind: str, lo: list[float], hi: list[float], follow: str | None = None) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        n = counts[kind]
        out.append({"kind": kind, "name": f"{INTERIOR_EN[kind]} {n} (estimated)", "nameAr": f"{INTERIOR_AR[kind]} {n} (تقديري)",
                    "min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi], "color": color, "follow": follow})

    for item in interior.get("items") or []:
        kind, count = item["kind"], min(int(item.get("count") or 1), 8)
        behind = [by_analysis[i]["box"] for i in item.get("behind") or [] if i in by_analysis and by_analysis[i].get("box")]
        x0 = min(b[0][0] for b in behind) if behind else blo[0] + wall
        x1 = max(b[1][0] for b in behind) if behind else bhi[0] - wall
        x0, x1 = max(x0, blo[0] + wall), min(x1, bhi[0] - wall)
        z0, z1 = blo[2] + wall, bhi[2] - 0.08 * d  # behind the doors, in front of the back
        if x1 - x0 <= wall or z1 - z0 <= wall:
            continue
        if kind == "shelf":
            for k in range(count):
                y = blo[1] + h * (0.15 + 0.7 * (k + 1) / (count + 1))
                add(kind, [x0, y, z0], [x1, y + 0.02 * h, z1])
        elif kind == "rail":
            for k in range(count):
                y, z = bhi[1] - 0.12 * h, (z0 + z1) / 2
                add(kind, [x0, y - 0.008 * h, z - 0.008 * h], [x1, y + 0.008 * h, z + 0.008 * h])
        elif kind == "divider":
            for k in range(count):
                x = x0 + (x1 - x0) * (k + 1) / (count + 1)
                add(kind, [x - 0.01 * w, blo[1] + wall, z0], [x + 0.01 * w, bhi[1] - wall, z1])
        elif kind == "drawer_box":
            for e in entries:
                c = next((o for o in components if o["id"] == e["component"]), None)
                if e["type"] != "translation" or e.get("direction") != "front" or not c or not c.get("box"):
                    continue
                (dlo, dhi) = c["box"]
                dw, dh = dhi[0] - dlo[0], dhi[1] - dlo[1]
                add(kind, [dlo[0] + 0.05 * dw, dlo[1] + 0.08 * dh, z0], [dhi[0] - 0.05 * dw, dhi[1] - 0.25 * dh, dlo[2]], follow=e["id"])
    return out


def configure(glb: bytes, doc: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """The animation configuration of the model (doc["animation"]), and the GLB with its estimated interior parts
    (only for a storage piece the analysis is sure about, with a part that opens to show it)."""
    components = list(doc.get("components") or [])
    whole = _whole(components)
    analysis = doc.get("analysis")
    entries: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    if whole is None:
        return glb, {**doc, "animation": {"version": 1, "movable": False, "animations": [], "flagged": [], "source": "none"}}
    separated = bool(doc.get("separated"))
    movers: list[tuple[dict[str, Any] | None, dict[str, Any], str, str | None, str]] = []
    if analysis:
        by_analysis = {c.get("analysisId"): c for c in components if c.get("analysisId")}
        for a in analysis.get("components") or []:
            motion = a.get("motion") or {}
            if motion.get("type") not in (None, "none"):
                movers.append((by_analysis.get(a["id"]), motion, a.get("name") or a["id"], a.get("nameAr") or None, "vision"))
    else:
        for c in components:
            if guess := _guess(c):
                movers.append((c, guess, c.get("name") or c["id"], c.get("nameAr"), "names"))
    for c, motion, label, label_ar, source in movers:
        why = _blocked(c, components, whole, separated)
        if why:
            flagged.append({"label": label, "labelAr": label_ar, "motion": motion.get("type"), "reason": why})
            continue
        entry = _entry(c, motion, whole, label, label_ar, source)
        entries.append(entry)
        if not entry["enabled"] or entry["issues"]:
            flagged.append({"label": label, "labelAr": label_ar, "motion": motion.get("type"), "component": c["id"], "reason": " ".join(entry["issues"]) or "Check it."})

    config: dict[str, Any] = {"version": 1, "movable": bool(entries), "animations": entries, "flagged": flagged, "source": "vision" if analysis else "names"}
    interior = (analysis or {}).get("interior") or {}
    opens = any(e["enabled"] and e["revealsInterior"] for e in entries)
    if interior.get("hasInterior") and float(interior.get("confidence") or 0) >= INTERIOR_CONFIDENCE and opens:
        body = next((a for a in analysis.get("components") or [] if a.get("color")), {})
        boxes = _interior_boxes(interior, entries, components, whole, body.get("color"))
        if boxes:
            try:
                glb, added = append_boxes(glb, boxes)
            except (ValueError, KeyError) as exc:
                logger.warning("Estimated interior not added: %s", exc)
                added = []
            for box, idx in zip(boxes, added):
                part_id = re.sub(r"[^a-z0-9]+", "_", f"estimated {box['name']}".lower().replace("(estimated)", "")).strip("_")
                components.append({"id": part_id, "name": box["name"], "nameAr": box["nameAr"], "estimated": True, "naming": "estimated", "editable": True,
                                   "box": [box["min"], box["max"]], **idx})
                if box["follow"]:
                    next(e for e in entries if e["id"] == box["follow"])["follow"].append(part_id)
            config["interior"] = {"added": [c["id"] for c in components if c.get("estimated")],
                                  "note": "Estimated from the object's type, not seen in the photo: edit or remove these parts. If the generated body is solid behind its doors, they stay hidden."}
    logger.info("Animation (%s): %s; flagged %s", config["source"], [(e["component"], e["type"], e["enabled"]) for e in entries], [f["label"] for f in flagged])
    return glb, {**doc, "components": components, "animation": config}
