"""Descriptive names for a GLB's meshes ("Model part0" → "Seat Cushion"), from the furniture type the vendor chose.

1. A mesh that already has a real name in the file keeps it.
2. Geometry: where the part sits in the whole model's box (legs on the floor, a table's top, small handles…).
3. AI: what is still generic goes to the vision model as numbers (box, size, colour), with the names allowed for
   this furniture type; only answers at confidence ≥ MIN_CONFIDENCE are used.
Anything not identified keeps a neutral "Part N". Component ids never change; only display names do.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.7

VOCABULARY = {
    "sofa": ["Frame", "Seat Cushion", "Back Cushion", "Armrest", "Pillow", "Leg"],
    "bed": ["Frame", "Mattress", "Headboard", "Pillow", "Leg"],
    "table": ["Tabletop", "Leg", "Frame"],
    "wardrobe": ["Body", "Door", "Drawer", "Handle"],
    "chair": ["Backrest", "Seat Cushion", "Frame", "Front Leg", "Rear Leg", "Armrest"],
    "door": ["Leaf", "Frame", "Handle"],
}

_GENERIC = re.compile(r"^(model|mesh|object|node|geometry|group|default|part|whole model)?[\s_\-]*(part|mesh|object|geometry)?[\s_\-]*\d*(\s\d+)?$", re.I)


def is_generic(name: str) -> bool:
    return bool(_GENERIC.match(name.strip()))


def _relative(components: list[dict[str, Any]]) -> dict[str, dict[str, float]] | None:
    """Each part's box as fractions of the whole model's box (y up), plus `f0`..`f1`: how far towards the front
    (0 = back, 1 = front). The back is the side the tall, shallow parts (backrest, headboard, back cushions) lean
    to; with none, glTF's +Z front, which Meshy uses."""
    boxes = [c["box"] for c in components if c.get("box")]
    if len(boxes) != len(components):
        return None
    lo = [min(b[0][i] for b in boxes) for i in range(3)]
    hi = [max(b[1][i] for b in boxes) for i in range(3)]
    span = [max(hi[i] - lo[i], 1e-6) for i in range(3)]
    out = {}
    for c in components:
        (x0, y0, z0), (x1, y1, z1) = c["box"]
        out[c["id"]] = {
            "x0": (x0 - lo[0]) / span[0], "x1": (x1 - lo[0]) / span[0],
            "y0": (y0 - lo[1]) / span[1], "y1": (y1 - lo[1]) / span[1],
            "z0": (z0 - lo[2]) / span[2], "z1": (z1 - lo[2]) / span[2],
        }
    tall = [(r["z0"] + r["z1"]) / 2 for r in out.values() if r["y1"] > 0.85 and r["y1"] - r["y0"] > 0.3 and r["z1"] - r["z0"] < 0.6]
    front_high = not tall or sum(tall) / len(tall) < 0.5
    for r in out.values():
        r["f0"], r["f1"] = (r["z0"], r["z1"]) if front_high else (1 - r["z1"], 1 - r["z0"])
    return out


def _geometry_name(r: dict[str, float], family: str) -> str | None:
    """Rules that hold for any model of the family; anything less certain is left to the AI step."""
    dx, dy, dz = r["x1"] - r["x0"], r["y1"] - r["y0"], r["z1"] - r["z0"]
    xc, fc = (r["x0"] + r["x1"]) / 2, (r["f0"] + r["f1"]) / 2
    on_floor = r["y0"] < 0.03
    side = xc < 0.25 or xc > 0.75
    if family == "table" and r["y0"] > 0.6 and dx > 0.7 and dz > 0.7 and dy < 0.25:
        return "Tabletop"
    if family == "chair" and on_floor and dx < 0.3 and dz < 0.3:
        return "Front Leg" if fc > 0.5 else "Rear Leg"  # rear legs may run up into the backrest
    if "Leg" in VOCABULARY.get(family, []) and on_floor and dx < 0.2 and dz < 0.2 and (family == "table" or r["y1"] < 0.35):
        return "Leg"
    if "Handle" in VOCABULARY.get(family, []) and max(dx, dy, dz) < 0.15 and not on_floor:
        return "Handle"
    if family in ("sofa", "bed", "chair") and dx > 0.9 and dz > 0.9 and dy > 0.6:
        return "Frame"
    if family == "wardrobe" and dx > 0.9 and dz > 0.8 and dy > 0.85:
        return "Body"
    if family in ("chair", "sofa") and side and dx < 0.25 and dz > 0.35 and r["y0"] > 0.15 and r["y1"] > 0.4:
        return "Armrest"
    if family == "chair":
        if r["y0"] > 0.3 and r["f1"] < 0.55 and dz < 0.45 and dx > 0.4:
            return "Backrest"
        if 0.15 < r["y0"] < 0.65 and dy < 0.3 and dx > 0.4 and dz > 0.35:
            return "Seat Cushion"
    if family == "sofa" and not on_floor:
        if 0.12 < r["y0"] < 0.6 and dy < 0.35 and dz > 0.3 and fc > 0.4:
            return "Seat Cushion"
        if r["y0"] > 0.3 and fc < 0.5 and dz < 0.5 and dy > 0.2:
            return "Back Cushion"
        if max(dx, dz) < 0.35 and r["y0"] > 0.3:
            return "Pillow"
    if family == "bed":
        if r["f1"] < 0.3 and dy > 0.45 and dx > 0.6:
            return "Headboard"
        if dx > 0.7 and dz > 0.6 and r["y0"] > 0.08 and dy < 0.45 and r["y1"] < 0.85:
            return "Mattress"
        if max(dx, dz) < 0.5 and r["y0"] > 0.3 and fc < 0.45 and dy < 0.3:
            return "Pillow"
    return None


Ask = Callable[[str, dict], Awaitable[str | None]]

_SCHEMA = {
    "type": "object",
    "properties": {
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "name": {"type": ["string", "null"]}, "confidence": {"type": "number"}},
                "required": ["id", "name", "confidence"],
            },
        }
    },
    "required": ["parts"],
}


async def _ai_names(family: str, rel: dict[str, dict[str, float]], todo: list[dict[str, Any]], known: dict[str, str], ask: Ask) -> dict[str, str]:
    allowed = VOCABULARY[family]
    rows = [{"id": c["id"], **{k: round(v, 2) for k, v in rel[c["id"]].items()}, **({"name": known[c["id"]]} if c["id"] in known else {})} for c in todo]
    prompt = (
        f"A 3D model of a {family} is split into the meshes below. Each box is a fraction of the whole model's box: "
        "x0..x1 width, y0..y1 height (0 = floor, 1 = top), z0..z1 depth (0 = back, 1 = front, may be reversed). "
        f"Name each mesh with exactly one of {allowed}, or null when the box alone does not show what it is. "
        "Give confidence 0..1. Do not guess: a wrong name is worse than null.\n"
        + json.dumps(rows)
    )
    text = await ask(prompt, _SCHEMA)
    if not text:
        return {}
    try:
        answers = json.loads(text).get("parts") or []
    except (ValueError, AttributeError):
        logger.warning("Part naming returned invalid JSON")
        return {}
    out = {}
    for a in answers:
        if isinstance(a, dict) and a.get("name") in allowed and float(a.get("confidence") or 0) >= MIN_CONFIDENCE:
            out[str(a.get("id"))] = a["name"]
    return out


async def name_components(doc: dict[str, Any], family: str, ask: Ask | None = None) -> dict[str, Any]:
    """The component document with descriptive names where they were identified (`naming` says how)."""
    components = doc.get("components") or []
    if len(components) < 2 or family not in VOCABULARY:
        return doc
    rel = _relative(components)
    found: dict[str, tuple[str, str]] = {}
    for c in components:
        if not is_generic(c["name"]):
            c.setdefault("naming", "file")
            continue
        if rel and (name := _geometry_name(rel[c["id"]], family)):
            found[c["id"]] = (name, "geometry")
    todo = [c for c in components if is_generic(c["name"]) and c["id"] not in found]
    if todo and rel and ask is not None:
        try:
            ai = await _ai_names(family, rel, [c for c in components if is_generic(c["name"])], {k: v[0] for k, v in found.items()}, ask)
        except Exception as exc:  # the AI step is optional: geometry names and neutral names still apply
            logger.warning("Part naming AI step failed: %s", str(exc)[:200])
            ai = {}
        for c in todo:
            if c["id"] in ai:
                found[c["id"]] = (ai[c["id"]], "ai")
    counts: dict[str, int] = {}
    for name, _ in found.values():
        counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    unknown = 0
    for c in components:
        if not is_generic(c["name"]):
            continue
        if c["id"] in found:
            name, how = found[c["id"]]
            seen[name] = seen.get(name, 0) + 1
            c["name"] = f"{name} {seen[name]}" if counts[name] > 1 else name
            c["naming"] = how
        else:
            unknown += 1
            c["name"] = f"Part {unknown}"
            c["naming"] = "unidentified"
    logger.info("Part names (%s): %s", family, [(c["name"], c.get("naming")) for c in components])
    return doc
