"""Wardrobes from a photo as real, rigid, separately editable parts.

Image-to-3D (Meshy, Astra) returns one free-form surface: a wardrobe comes back with bent panels and doors fused to
the body, and no geometric split can recover the doors, drawers and handles from it. A wardrobe is a box of flat
panels, so instead:

1. The vision model reads the photo into a layout (`WardrobeSpec`): size, columns left → right, and each column's
   sections bottom → top (doors, drawers, open shelves), the handle style and the colours.
2. `build` turns the layout into one mesh per real component with a semantic name (wardrobe_body, back_panel,
   door_left, drawer_1, handle_1, shelf_1, plinth / leg_1) and its own material, so each one can be selected,
   recoloured, resized and given a material on its own.
3. `services/model_check.py` validates the result before it is saved.

Metres, y up, the front is +z, centred on x and z, standing on y = 0.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

PANEL = 0.018  # carcass and front thickness
BACK = 0.006
GAP = 0.003  # between a front and its opening
DRAWER_SIDE = 0.012

Hex = Field(default="#e8e4dc", pattern=r"^#[0-9a-fA-F]{6}$")


class Section(BaseModel):
    kind: Literal["door", "drawer", "open"] = Field(description="door: hinged door(s) cover it; drawer: one drawer; open: no front, shelves visible")
    height: float = Field(gt=0, description="Share of the column's height, bottom to top; a column's shares are scaled to add up to 1")
    leaves: Literal[1, 2] = Field(default=1, description="door only: 1 single door or 2 doors side by side")
    shelves: int = Field(default=0, ge=0, le=8, description="Inner shelves in this section (open: those visible; door: those likely behind it, 0 for hanging space)")


class Column(BaseModel):
    width: float = Field(gt=0, description="Share of the wardrobe's width; the columns' shares are scaled to add up to 1")
    sections: list[Section] = Field(min_length=1, max_length=8, description="Bottom to top")


class WardrobeSpec(BaseModel):
    width_m: float = Field(description="Outside width in metres")
    height_m: float = Field(description="Outside height in metres, including any base")
    depth_m: float = Field(description="Outside depth in metres")
    base: Literal["plinth", "legs", "none"] = "plinth"
    base_height_m: float = 0.06
    handle: Literal["knob", "bar", "none"] = "knob"
    body_color: str = Hex
    front_color: str = Hex
    handle_color: str = Field(default="#c9c3b8", pattern=r"^#[0-9a-fA-F]{6}$")
    columns: list[Column] = Field(min_length=1, max_length=6, description="Left to right, as seen from the front")


PROMPT = """You read a product photo of {what} for a furniture store that rebuilds it in 3D from flat panels.
Describe its layout exactly as the photo shows it. Columns are the vertical bays left to right as seen from the
front (a column ends where a full-height divider or a change of fronts is). Each column's sections go bottom to top:
"door" (a hinged door covers it; leaves 2 when two doors meet in the middle), "drawer" (one drawer each), "open"
(no front: shelves are visible). Count the drawers, doors and visible shelves exactly. Heights and widths are shares
of the whole. Colours are the photo's own, as #rrggbb: body_color for the carcass and sides, front_color for the
doors and drawer fronts, handle_color for the handles. handle: "knob" for round knobs, "bar" for bar pulls, "none"
for push-to-open or cut-outs. Size in metres{size}."""


def _child(kind: str | None) -> bool:
    return bool(re.search(r"child|kid|baby|أطفال|اطفال|طفل", kind or "", re.I))


def normalize(spec: WardrobeSpec, kind: str | None = None) -> WardrobeSpec:
    """Real-world bounds (a child's wardrobe is lower), shares scaled to 1, a base height that fits."""
    child = _child(kind)
    s = spec.model_copy(deep=True)
    s.width_m = min(3.0, max(0.4, s.width_m or 1.0))
    s.height_m = min(2.0 if child else 2.7, max(0.8, s.height_m or (1.6 if child else 2.0)))
    s.depth_m = min(0.7, max(0.3, s.depth_m or (0.45 if child else 0.58)))
    s.base_height_m = 0.0 if s.base == "none" else min(0.15, max(0.03, s.base_height_m or 0.06))
    total = sum(c.width for c in s.columns)
    for c in s.columns:
        c.width /= total
        h = sum(x.height for x in c.sections)
        for x in c.sections:
            x.height /= h
    return s


def spec_from_json(text: str | None, kind: str | None = None) -> WardrobeSpec | None:
    if not text:
        return None
    try:
        return normalize(WardrobeSpec.model_validate(json.loads(text)), kind)
    except (ValueError, ValidationError) as exc:
        logger.warning("Wardrobe layout unreadable: %s", str(exc)[:300])
        return None


async def read_spec(ask, what: str, kind: str | None, details=None) -> WardrobeSpec | None:
    """`ask(prompt, schema)` sends the prompt with the photo to the vision model and returns its JSON text."""
    dims = [getattr(details, k, None) for k in ("width_m", "height_m", "depth_m")] if details else []
    size = f" (an earlier reading estimated {dims[0]:.2f} wide × {dims[1]:.2f} tall × {dims[2]:.2f} deep)" if dims and all(dims) else ""
    if _child(kind):
        size += "; it is a child's wardrobe, usually 1.0–1.8 m tall"
    return spec_from_json(await ask(PROMPT.format(what=what, size=size), WardrobeSpec.model_json_schema()), kind)


# ---- building ----

# Display names: whole ids first, then numbered ones by their base ("drawer_2" → "درج 2").
AR = {"wardrobe_body": "هيكل الدولاب", "back_panel": "اللوح الخلفي", "plinth": "القاعدة", "door": "الباب", "door_left": "الباب الأيسر", "door_right": "الباب الأيمن"}
EN = {"wardrobe_body": "Wardrobe body", "back_panel": "Back panel", "plinth": "Plinth", "door": "Door", "door_left": "Left door", "door_right": "Right door"}
AR_N = {"door": "باب", "drawer": "درج", "handle": "مقبض", "shelf": "رف", "leg": "رجل"}
EN_N = {"door": "Door", "drawer": "Drawer", "handle": "Handle", "shelf": "Shelf", "leg": "Leg"}


def part_name(part_id: str, lang: str = "ar") -> str:
    words, numbered = (AR, AR_N) if lang == "ar" else (EN, EN_N)
    if part_id in words:
        return words[part_id]
    base, _, n = part_id.rpartition("_")
    return f"{numbered.get(base, base)} {n}" if n.isdigit() else part_id


def _rgba(hex_color: str, alpha: float = 1.0) -> list[float]:
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha]


def build(spec: WardrobeSpec, lang: str = "ar") -> tuple[bytes, list[dict]]:
    """The GLB (one node, mesh and material per part) and its parts as [{id, name}] in file order."""
    import numpy as np
    import trimesh
    from trimesh.visual.material import PBRMaterial

    W, H, D, b, t = spec.width_m, spec.height_m, spec.depth_m, spec.base_height_m, PANEL
    parts: dict[str, list] = {}
    colors: dict[str, tuple[str, float, float]] = {}  # part → colour, roughness, metalness

    def box(name: str, w: float, h: float, d: float, x: float, y: float, z: float, color: str, rough=0.6, metal=0.0):
        m = trimesh.creation.box(extents=(max(w, 0.002), max(h, 0.002), max(d, 0.002)))
        m.apply_translation((x, y, z))
        parts.setdefault(name, []).append(m)
        colors.setdefault(name, (color, rough, metal))

    body, front, handle = spec.body_color, spec.front_color, spec.handle_color
    z_front = D / 2 - t / 2  # fronts sit flush inside the carcass
    inner_d = D - t - BACK  # inside depth, behind the fronts

    # Carcass: sides, top and bottom; full-height dividers between the columns.
    side0 = 0.0 if spec.base == "plinth" else b  # sides run down to the floor around a plinth; legs carry them
    box("wardrobe_body", t, H - side0, D, -W / 2 + t / 2, (side0 + H) / 2, 0, body)
    box("wardrobe_body", t, H - side0, D, W / 2 - t / 2, (side0 + H) / 2, 0, body)
    box("wardrobe_body", W - 2 * t, t, D - BACK, 0, H - t / 2, BACK / 2, body)
    box("wardrobe_body", W - 2 * t, t, D - BACK, 0, b + t / 2, BACK / 2, body)
    box("back_panel", W - 2 * t, H - b - 2 * t, BACK, 0, b + t + (H - b - 2 * t) / 2, -D / 2 + BACK / 2, body)
    if spec.base == "plinth":
        box("plinth", W - 2 * t, b, t, 0, b / 2, D / 2 - t * 1.5, body)
    elif spec.base == "legs":
        for i, (sx, sz) in enumerate(((-1, 1), (1, 1), (-1, -1), (1, -1)), 1):
            leg = trimesh.creation.cylinder(radius=0.02, height=b, sections=24)
            leg.apply_translation((sx * (W / 2 - 0.05), b / 2, sz * (D / 2 - 0.05)))
            parts[f"leg_{i}"] = [leg]
            colors[f"leg_{i}"] = (handle, 0.4, 0.3)

    y0, y1 = b + t, H - t  # inside, floor to ceiling
    usable = W - 2 * t - t * (len(spec.columns) - 1)
    x = -W / 2 + t
    fronts: list[tuple[str, float, float, float, float, str]] = []  # kind, x0, x1, y0, y1, handle side
    shelves: list[tuple[float, float, float]] = []  # x0, x1, y
    for ci, col in enumerate(spec.columns):
        cw = usable * col.width
        x0, x1 = x, x + cw
        if ci:
            box("wardrobe_body", t, y1 - y0, D - BACK, x0 - t / 2, (y0 + y1) / 2, BACK / 2, body)
        y = y0
        n = len(col.sections)
        span = y1 - y0 - t * (n - 1)
        for si, sec in enumerate(col.sections):
            sh = span * sec.height
            s0, s1 = y, y + sh
            if si < n - 1:  # a fixed panel between sections
                box("wardrobe_body", cw, t, inner_d, (x0 + x1) / 2, s1 + t / 2, (D - inner_d) / 2 - t, body)
            if sec.kind == "door":
                if sec.leaves == 2:
                    mid = (x0 + x1) / 2
                    fronts += [("door", x0, mid, s0, s1, "right"), ("door", mid, x1, s0, s1, "left")]
                else:
                    fronts.append(("door", x0, x1, s0, s1, "left" if ci == len(spec.columns) - 1 and len(spec.columns) > 1 else "right"))
            elif sec.kind == "drawer":
                fronts.append(("drawer", x0, x1, s0, s1, "center"))
            for k in range(sec.shelves if sec.kind != "drawer" else 0):
                shelves.append((x0, x1, s0 + sh * (k + 1) / (sec.shelves + 1)))
            y = s1 + t
        x = x1 + t

    doors = [f for f in fronts if f[0] == "door"]
    door_names = ["door_left", "door_right"] if len(doors) == 2 else ["door"] if len(doors) == 1 else [f"door_{i}" for i in range(1, len(doors) + 1)]
    door_names = iter(door_names)
    # Drawers numbered top to bottom, then left to right, as a shop lists them.
    drawer_order = sorted((f for f in fronts if f[0] == "drawer"), key=lambda f: (-f[3], f[1]))
    drawer_names = {id(f): f"drawer_{i}" for i, f in enumerate(drawer_order, 1)}
    hn = 0
    for f in fronts:
        kind, fx0, fx1, fy0, fy1, side = f
        fw, fh = fx1 - fx0 - 2 * GAP, fy1 - fy0 - 2 * GAP
        cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        name = next(door_names) if kind == "door" else drawer_names[id(f)]
        box(name, fw, fh, t, cx, cy, z_front, front)
        if kind == "drawer":  # the drawer box behind its front: sides, back and bottom
            dd, dh, dw = inner_d - 0.03, fh * 0.7, fw - 0.03
            bz = z_front - t / 2 - dd / 2
            box(name, DRAWER_SIDE, dh, dd, cx - dw / 2 + DRAWER_SIDE / 2, fy0 + GAP + dh / 2, bz, body)
            box(name, DRAWER_SIDE, dh, dd, cx + dw / 2 - DRAWER_SIDE / 2, fy0 + GAP + dh / 2, bz, body)
            box(name, dw - 2 * DRAWER_SIDE, dh, DRAWER_SIDE, cx, fy0 + GAP + dh / 2, bz - dd / 2 + DRAWER_SIDE / 2, body)
            box(name, dw - 2 * DRAWER_SIDE, DRAWER_SIDE, dd, cx, fy0 + GAP + DRAWER_SIDE / 2, bz, body)
        if spec.handle == "none":
            continue
        hn += 1
        hx = cx if side == "center" else (fx1 - GAP - 0.05 if side == "right" else fx0 + GAP + 0.05)
        hy = cy if kind == "drawer" else min(fy0 + 1.0, cy)  # doors: at hand height, never above the middle
        hz = D / 2 + 0.0125
        if spec.handle == "knob":
            knob = trimesh.creation.cylinder(radius=0.014, height=0.025, sections=24)
            knob.apply_translation((hx, hy, hz))
            parts[f"handle_{hn}"] = [knob]
            colors[f"handle_{hn}"] = (handle, 0.35, 0.2)
        else:
            vertical = kind == "door"
            box(f"handle_{hn}", 0.012 if vertical else 0.12, 0.12 if vertical else 0.012, 0.025, hx, hy, hz, handle, 0.3, 0.7)
    for i, (sx0, sx1, sy) in enumerate(sorted(shelves, key=lambda s: (s[0], s[2])), 1):
        box(f"shelf_{i}", sx1 - sx0 - 0.002, t, inner_d - 0.01, (sx0 + sx1) / 2, sy, (D - inner_d) / 2 - t, body)

    scene = trimesh.Scene()
    out = []
    groups = ("wardrobe_body", "back_panel", "plinth", "leg", "door", "drawer", "shelf", "handle")

    def order(name: str) -> tuple[int, int]:
        base, _, n = name.rpartition("_")
        group = next(i for i, g in enumerate(groups) if name.startswith(g))
        return group, int(n) if n.isdigit() else {"left": 0, "right": 1}.get(n, 0)

    for name in sorted(parts, key=order):
        mesh = trimesh.util.concatenate(parts[name])
        mesh.merge_vertices()
        color, rough, metal = colors[name]
        # Its own material per part: recolouring one part never reaches another.
        mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(name=name, baseColorFactor=np.round(np.array(_rgba(color)) * 255).astype(np.uint8), roughnessFactor=rough, metallicFactor=metal))
        scene.add_geometry(mesh, geom_name=name, node_name=name)
        out.append({"id": name, "name": part_name(name, lang)})
    return scene.export(file_type="glb"), out


def components_doc(glb: bytes, names: list[dict]) -> dict:
    """The component document the editor reads: inspect_glb's list (boxes, mesh indices) with the display names."""
    from app.services.glb_parts import inspect_glb

    doc = inspect_glb(glb, "wardrobe")
    by_id = {n["id"]: n["name"] for n in names}
    for c in doc["components"]:
        c["name"] = by_id.get(c["id"], c["name"])
        c["naming"] = "file"
    return {**doc, "method": "procedural", "limits": None}
