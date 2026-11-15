"""Vision AI mesh analysis: the photo read as physical components before Meshy runs, and the generated GLB checked
against them afterwards.

  photo ─ vision model ─▶ MeshAnalysis (validated JSON) ─▶ the Meshy prompt
                                                         ├─ Image to 3D `texture_prompt` (colours, materials; ≤ 800)
                                                         └─ Auto Split `prompt` (the parts to cut, 1–10 names; ≤ 600)
  Meshy GLB ─ match() ─▶ each mesh named after the component it is (geometry untouched), modular / partial / none

Meshy's Image to 3D has no prompt that shapes geometry (docs.meshy.ai/en/api/image-to-3d: `texture_prompt` only,
which guides the texture). So the full prompt is not sent as one text: each part of it goes to the parameter that
acts on it, and `conditioning()` records where each went and what had nowhere to go. Nothing is dropped silently.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

Ask = Callable[[str, dict], Awaitable[str | None]]

TEXTURE_PROMPT_MAX = 800  # Meshy Image to 3D texture_prompt
SPLIT_PROMPT_MAX = 600  # Meshy Auto Split prompt
SPLIT_MAX_PARTS = 10  # Auto Split reads 1–10 part names; more is a 400
MAX_COMPONENTS = 30
MIN_IOU = 0.25  # a mesh is that component only when their front boxes overlap at least this much


class Relation(BaseModel):
    type: Literal["attached_to", "inside", "above", "below", "left_of", "right_of", "in_front_of", "behind", "on"]
    target: str


class Dimensions(BaseModel):
    """Fractions of the whole object's width / height / depth (0..1]; None where the photo does not show it."""

    width: float | None = Field(default=None, gt=0, le=1)
    height: float | None = Field(default=None, gt=0, le=1)
    depth: float | None = Field(default=None, gt=0, le=1)


class Motion(BaseModel):
    """How a part really moves (services/animation.py turns it into an animation). `type` "none" for fixed parts.
    hinge: the edge of the part's own box it turns about (rotate / fold / lift). direction: where it slides
    (slide / extend / retract). limit: degrees for turning parts. travel: how far it slides, as a fraction of the
    whole object's size in that direction."""

    type: Literal["none", "rotate", "slide", "fold", "lift", "extend", "retract"] = "none"
    hinge: Literal["left", "right", "top", "bottom", "back", "front"] | None = None
    direction: Literal["front", "back", "left", "right", "up", "down"] | None = None
    limit: float | None = None
    travel: float | None = None
    revealsInterior: bool = False
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: str = ""

    @field_validator("hinge", "direction", mode="before")
    @classmethod
    def _none(cls, v: Any) -> Any:
        return None if v in ("none", "", None) else v

    @field_validator("confidence", mode="before")
    @classmethod
    def _unit(cls, v: Any) -> float:
        try:
            return min(max(float(v), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0


class InteriorItem(BaseModel):
    kind: Literal["shelf", "rail", "divider", "drawer_box"]
    count: int = Field(default=1, ge=1, le=8)
    behind: list[str] = []  # the components it sits behind (door ids); empty: the whole interior


class Interior(BaseModel):
    """What is probably inside (the photo shows it closed): simple estimated parts, only when it is a storage piece."""

    hasInterior: bool = False
    confidence: float = Field(default=0.0, ge=0, le=1)
    items: list[InteriorItem] = []


class Component(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=60)
    nameAr: str = Field(default="", max_length=60)
    shape: str = ""
    position: str = ""
    box: list[float] | None = None  # x0, y0, x1, y1 in the object's own box in the photo, 0..1, origin top left
    dimensions: Dimensions | None = None
    color: str | None = None  # #RRGGBB
    material: str = ""
    texture: str = ""
    role: Literal["structural", "functional", "decorative"] = "structural"
    visibility: Literal["visible", "inferred"] = "visible"
    separateMesh: bool = True
    relations: list[Relation] = []
    motion: Motion = Motion()

    @field_validator("motion", mode="before")
    @classmethod
    def _motion(cls, v: Any) -> Any:
        return v if isinstance(v, dict) else {}

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(v or "").lower()).strip("_")[:40] or "part"

    @field_validator("color", mode="before")
    @classmethod
    def _color(cls, v: Any) -> str | None:
        m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(v or "").strip())
        return f"#{m.group(1).upper()}" if m else None

    @field_validator("box", mode="before")
    @classmethod
    def _box(cls, v: Any) -> list[float] | None:
        try:
            x0, y0, x1, y1 = (min(max(float(n), 0.0), 1.0) for n in v)
        except (TypeError, ValueError):
            return None
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return [x0, y0, x1, y1] if x1 - x0 > 0.005 and y1 - y0 > 0.005 else None


class MeshAnalysis(BaseModel):
    objectType: str = Field(min_length=1, max_length=80)
    summary: str = ""
    components: list[Component] = Field(min_length=1)
    movable: bool = False  # the object has parts a person opens, slides or folds
    bodyId: str = ""  # the main structural body (the stationary part everything is attached to)
    interior: Interior = Interior()

    @field_validator("interior", mode="before")
    @classmethod
    def _interior(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return {}
        v = {**v, "items": [i for i in (v.get("items") or []) if isinstance(i, dict) and i.get("kind") in ("shelf", "rail", "divider", "drawer_box")]}
        return v


_COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "snake_case, unique: body, door_left, drawer_1, leg_front_left"},
        "name": {"type": "string", "description": "English name, e.g. 'Left door'"},
        "nameAr": {"type": "string", "description": "The same name in Arabic, e.g. 'الباب الأيسر'"},
        "shape": {"type": "string"},
        "position": {"type": "string", "description": "Where it sits on the object: 'front left', 'main structure'"},
        "box": {"type": ["array", "null"], "items": {"type": "number"}, "description": "x0, y0, x1, y1 as fractions of the WHOLE OBJECT's box in the photo (0..1, origin top left); null if not visible"},
        "dimensions": {
            "type": ["object", "null"],
            "properties": {"width": {"type": ["number", "null"]}, "height": {"type": ["number", "null"]}, "depth": {"type": ["number", "null"]}},
            "description": "Approximate size as fractions (0..1] of the whole object's width, height, depth",
        },
        "color": {"type": ["string", "null"], "description": "#RRGGBB as seen in the photo"},
        "material": {"type": "string"},
        "texture": {"type": "string", "description": "Surface finish or pattern: matte, wood grain, woven fabric…"},
        "role": {"type": "string", "enum": ["structural", "functional", "decorative"]},
        "visibility": {"type": "string", "enum": ["visible", "inferred"]},
        "separateMesh": {"type": "boolean", "description": "true for a physical part a person could recolour or move on its own"},
        "motion": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["none", "rotate", "slide", "fold", "lift", "extend", "retract"]},
                "hinge": {"type": "string", "enum": ["none", "left", "right", "top", "bottom", "back", "front"], "description": "The edge of this part it turns about"},
                "direction": {"type": "string", "enum": ["none", "front", "back", "left", "right", "up", "down"], "description": "Where it slides"},
                "limit": {"type": ["number", "null"], "description": "Turning parts: the opening angle in degrees (a door ~100, a fold-down flap ~90)"},
                "travel": {"type": ["number", "null"], "description": "Sliding parts: how far, as a fraction of the whole object's size in that direction (a drawer ~0.6)"},
                "revealsInterior": {"type": "boolean"},
                "confidence": {"type": "number", "description": "0..1: how sure you are it moves this way"},
                "reason": {"type": "string", "description": "What in the photo shows it (hinges, a handle, a gap…)"},
            },
            "required": ["type", "confidence"],
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(Relation.model_fields["type"].annotation.__args__)},
                    "target": {"type": "string", "description": "another component's id"},
                },
                "required": ["type", "target"],
            },
        },
    },
    "required": ["id", "name", "nameAr", "shape", "position", "box", "color", "material", "texture", "role", "visibility", "separateMesh", "relations", "motion"],
}

SCHEMA = {
    "type": "object",
    "properties": {
        "objectType": {"type": "string"},
        "summary": {"type": "string", "description": "One sentence: the overall structure"},
        "components": {"type": "array", "items": _COMPONENT_SCHEMA},
        "movable": {"type": "boolean", "description": "Does this object have parts a person opens, slides, folds or lifts?"},
        "bodyId": {"type": "string", "description": "The id of the main structural body (stationary; the others attach to it)"},
        "interior": {
            "type": "object",
            "properties": {
                "hasInterior": {"type": "boolean", "description": "A storage piece with a hidden inside (wardrobe, cabinet, chest…)"},
                "confidence": {"type": "number"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["shelf", "rail", "divider", "drawer_box"]},
                            "count": {"type": "integer"},
                            "behind": {"type": "array", "items": {"type": "string"}, "description": "ids of the doors it sits behind; empty for the whole inside"},
                        },
                        "required": ["kind", "count", "behind"],
                    },
                },
            },
            "required": ["hasInterior", "confidence", "items"],
        },
    },
    "required": ["objectType", "summary", "components", "movable", "bodyId", "interior"],
}

PROMPT = """You are preparing a photo of a {what} for 3D reconstruction. List the meaningful PHYSICAL components the 3D
model should have as separate meshes (doors, drawers, shelves, legs, handles, cushions, frame, top, back… whatever
this object really has), plus the main body.

Rules:
- Describe only what this photo shows. A part you cannot see but that must exist (a hidden back panel, a fourth
  leg) is "inferred". Never add a component that is neither visible nor structurally certain.
- One entry per physical part: two doors are door_left and door_right, not "doors". Do not list surface details
  (grain, stitching, shadows) as components; mark ornaments as role "decorative".
- box: the part's rectangle in the photo as fractions of the whole object's own box (not of the image).
- color: the part's real colour in the photo as #RRGGBB (ignore highlights and shadows).
- relations: how parts connect (door_left attached_to body, shelf_1 inside body, drawer_1 below shelf_2…).
- motion: only for parts that really move in use (hinged doors, drawers, sliding doors, fold-down flaps, lids,
  recliner backs…). Everything else, and every part of an object that stays static (a table, a shelf, a framed
  picture), is type "none". Never invent a movable part; give a low confidence when the photo does not show how
  it moves (no hinges or handle visible). Doors: hinge on the side away from the handle; two doors meeting in the
  middle hinge on their outer sides. Do not give every door the same hinge unless the photo shows it.
- revealsInterior: true only when opening it shows a hollow inside (a cupboard, a drawer), not a decorative panel.
- interior: only for storage pieces whose inside is hidden. List simple, typical items (shelves, a hanging rail,
  dividers, drawer boxes) with a count, and only what such a piece usually has; hasInterior false otherwise.
- At most {limit} components; the most important first (the main body first).
{hint}"""


async def analyze(ask: Ask, what: str, hint: str = "") -> MeshAnalysis | None:
    """The photo's components (validated; ids unique, relations only between listed parts). None when the vision
    model gave nothing usable: generation then runs as before, and the model's notes say it was not analysed."""
    prompt = PROMPT.format(what=what, limit=MAX_COMPONENTS, hint=f"\nThe vendor says: {hint}" if hint else "")
    try:
        text = await ask(prompt, SCHEMA)
    except Exception as exc:  # noqa: BLE001 - the analysis guides generation; it never blocks it
        logger.warning("Mesh analysis failed: %s", str(exc)[:200])
        return None
    return parse(text)


def parse(text: str | None) -> MeshAnalysis | None:
    if not text:
        return None
    try:
        raw = json.loads(text)
        raw["components"] = [c for c in (raw.get("components") or []) if isinstance(c, dict) and str(c.get("name") or "").strip()][:MAX_COMPONENTS]
        analysis = MeshAnalysis.model_validate(raw)
    except (ValueError, TypeError, AttributeError, ValidationError) as exc:
        logger.warning("Mesh analysis returned invalid JSON: %s", str(exc)[:300])
        return None
    seen: dict[str, int] = {}
    for c in analysis.components:
        seen[c.id] = seen.get(c.id, 0) + 1
        if seen[c.id] > 1:
            c.id = f"{c.id}_{seen[c.id]}"
    ids = {c.id for c in analysis.components}
    for c in analysis.components:
        c.relations = [r for r in c.relations if r.target in ids and r.target != c.id]
    return analysis


# ---- what Meshy gets ----


def _part_text(c: Component) -> str:
    look = " ".join(x for x in (c.color or "", c.material, c.texture) if x).strip()
    return f"{c.name.lower()} ({c.shape or 'part'}, {c.position or 'on the object'}{f', {look}' if look else ''})"


def meshy_prompt(a: MeshAnalysis) -> str:
    """The whole generation prompt, written from the analysis (stored with the model; split by `conditioning`)."""
    separate = [c for c in a.components if c.separateMesh]
    relations = [f"{c.name.lower()} {r.type.replace('_', ' ')} {next(x.name.lower() for x in a.components if x.id == r.target)}" for c in a.components for r in c.relations]
    return " ".join(
        x
        for x in (
            f"Generate a realistic modular 3D {a.objectType}.",
            a.summary,
            f"Components: {'; '.join(_part_text(c) for c in a.components)}." if a.components else "",
            f"Model these as independent meshes: {', '.join(c.name.lower() for c in separate)}." if separate else "",
            f"Layout: {'; '.join(relations)}." if relations else "",
            "Preserve the reference image's proportions, component layout, colors, and materials. "
            "Keep flat panels flat and edges straight. Avoid warped geometry, invented parts, and arbitrary mesh fragmentation.",
        )
        if x
    )


def _fit(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: cut.rfind(".") + 1] or cut[: cut.rfind(" ")] or cut  # whole sentences (else words) only


def texture_prompt(a: MeshAnalysis, base: str) -> str:
    """Image to 3D `texture_prompt`: each part's colour and material first (what the photo-wide brief cannot say),
    then the base brief, cut to Meshy's 800 characters."""
    looks = [f"{c.name.lower()}: {' '.join(x for x in (c.color or '', c.material, c.texture) if x)}" for c in a.components if c.color or c.material or c.texture]
    parts = f"Photorealistic {a.objectType}. Component colors and materials: {'; '.join(looks)}." if looks else ""
    return _fit(" ".join(x for x in (parts, base) if x), TEXTURE_PROMPT_MAX)


def split_prompt(a: MeshAnalysis) -> str | None:
    """Auto Split `prompt`: the separate parts to cut, by name (Meshy reads 1–10 names). Same-kind parts beyond ten
    share a name ("doors"); None when there are fewer than two separate parts."""
    names = [c.name.lower() for c in a.components if c.separateMesh]
    if len(names) < 2:
        return None
    if len(names) > SPLIT_MAX_PARTS:
        groups: dict[str, int] = {}
        for n in names:
            key = re.sub(r"\b(left|right|top|bottom|upper|lower|front|back|rear|middle|centre|center|inner|outer|\d+)\b", "", n).strip() or n
            groups[key] = groups.get(key, 0) + 1
        names = [k if v == 1 else f"{k}s" for k, v in groups.items()][:SPLIT_MAX_PARTS]
    text = ", ".join(names)
    return _fit(text, SPLIT_PROMPT_MAX).rstrip(", ")


def conditioning(prompt: str, texture: str, split: str | None, split_on: bool) -> dict[str, Any]:
    """Where the prompt went: sent to the parameters that act on it, the rest listed as not sendable (and why)."""
    return {
        "prompt": prompt,
        "sent": {"image_to_3d.texture_prompt": texture, **({"auto_split.prompt": split} if split and split_on else {})},
        "notSent": [
            "Geometry, proportions and layout: Meshy Image to 3D has no prompt for geometry (only texture_prompt); "
            "the photo alone shapes the mesh. They are used to check and name the result instead.",
            *([] if split_on or not split else ["The part list: Auto Split is turned off (MESHY_AUTO_SPLIT)."]),
        ],
    }


# ---- the GLB against the analysis ----


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _front_boxes(components: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Each mesh's box seen from the front (glTF +Z, Meshy's front) in the whole model's box: x left→right, y top→down."""
    boxes = {c["id"]: c["box"] for c in components if c.get("box")}
    if not boxes:
        return {}
    lo = [min(b[0][i] for b in boxes.values()) for i in range(2)]
    hi = [max(b[1][i] for b in boxes.values()) for i in range(2)]
    w, h = max(hi[0] - lo[0], 1e-6), max(hi[1] - lo[1], 1e-6)
    return {k: [(b[0][0] - lo[0]) / w, (hi[1] - b[1][1]) / h, (b[1][0] - lo[0]) / w, (hi[1] - b[0][1]) / h] for k, b in boxes.items()}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def match(doc: dict[str, Any], analysis: MeshAnalysis | None) -> dict[str, Any]:
    """The component document with each mesh matched to the analysis component it is: by name when the file's
    mesh carries one (Auto Split names its parts after the prompt), else by where it sits (front boxes, one to one,
    best overlap first, at least MIN_IOU). Matched meshes take the component's English and Arabic names; nothing
    else changes (ids, geometry, UVs and materials stay as generated). `modularity` says how well it went."""
    if analysis is None:
        return doc
    meshes = doc.get("components") or []
    expected = [c for c in analysis.components if c.separateMesh]
    out = {**doc, "analysis": analysis.model_dump(mode="json")}
    if not doc.get("separated") or len(meshes) < 2:
        out["modularity"] = {"level": "none", "matched": [], "missing": [c.id for c in expected], "unmatchedMeshes": [m["id"] for m in meshes],
                             "note": "One fused mesh: the analysed components could not be found as separate parts."}
        return out

    pairs: dict[str, Component] = {}
    by_name = {_norm(c.name): c for c in analysis.components}
    for m in meshes:
        hit = by_name.get(_norm(re.sub(r"\s+\d+$", "", m.get("name") or "")))
        if hit and hit.id not in {p.id for p in pairs.values()}:
            pairs[m["id"]] = hit
    fronts = _front_boxes(meshes)
    scored = sorted(
        ((_iou(fronts[m["id"]], c.box), m["id"], c) for m in meshes if m["id"] in fronts and m["id"] not in pairs for c in analysis.components if c.box),
        key=lambda s: -s[0],
    )
    taken = {c.id for c in pairs.values()}
    for score, mesh_id, c in scored:
        if score < MIN_IOU:
            break
        if mesh_id in pairs or c.id in taken:
            continue
        pairs[mesh_id] = c
        taken.add(c.id)

    components = []
    for m in meshes:
        c = pairs.get(m["id"])
        components.append({**m, "name": c.name, "nameAr": c.nameAr or None, "naming": "vision", "analysisId": c.id} if c else m)
    missing = [c.id for c in expected if c.id not in taken]
    unmatched = [m["id"] for m in meshes if m["id"] not in pairs]
    level = "full" if not missing and not unmatched else "partial"
    out["components"] = components
    out["modularity"] = {
        "level": level, "matched": sorted(pairs[k].id for k in pairs), "missing": missing, "unmatchedMeshes": unmatched,
        "note": None if level == "full" else "Partially modular: some analysed components were not found as separate meshes, "
        "or some meshes match no component. The geometry was left as generated.",
    }
    logger.info("Mesh analysis match: %s matched=%d missing=%s unmatched=%s", level, len(pairs), missing, unmatched)
    return out
