"""Carpenter-friendly components for a generated model: its meshes grouped into the furniture's real parts.

Runs after Meshy (and its Auto Split, or `glb_parts.segment_glb`) on the generated GLB, for any furniture type:

1. `parts`: every primitive of every mesh instance (named or not), with its world geometry, material and colour.
2. Fragments: a speck (largest side under FRAGMENT_SIZE of the model's diagonal) that touches a larger part with the same material is folded
   into it (Meshy splits leave specks; they are not components). Nothing is folded across materials or gaps.
3. The configured vision model (vision.current_model) sees the product photo, the model's look and a numbered map
   of the parts, and answers which parts form which component, its carpenter name and type, with a confidence.
   `validate` checks each answer against the geometry: the parts exist, belong to one component only, are one
   physical piece (touching) or a set of identical repeats (four legs, two doors), and the name is a real English
   word list, never "Mesh 001". Rejected answers leave those parts as they were, with a warning.
4. `rebuild`: each component's parts become one mesh (one primitive per material) under one node carrying the
   component id (`extras.partId`, what the editor selects by), with world transforms baked in. Vertices, UVs,
   normals, materials and textures are copied as they are; `check` proves the triangle and vertex counts and the
   bounds are unchanged, else the input is kept.

Nothing here invents construction details: sizes come from the geometry (metres only when the model's real size is
known), materials from the file. Unsure components are flagged, not presented as identified.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from app.services.glb_size import _views, read_glb, write_glb
from app.services.part_names import is_generic
from app.services.realistic_materials import Region, Unsupported, _accessor, _instances, _linear_to_srgb, _texture, render

logger = logging.getLogger(__name__)

# A fragment is a speck: its largest side under this share of the model's diagonal. Measured by size, not surface:
# legs, knobs and handles are small in surface but real components, and are never folded away.
FRAGMENT_SIZE = 0.015
TOUCH = 0.02  # parts closer than this share of the model's diagonal touch
REPEAT_TOLERANCE = 0.25  # identical repeats: sizes within 25% (a carved front leg and back leg), same material
MIN_CONFIDENCE = 0.5  # below: the answer is not used
SURE = 0.75  # below: used, but flagged as uncertain
MAX_PARTS_FOR_AI = 40  # more parts than this: the map is unreadable, grouping is skipped with a warning
POINTS = 220_000  # surface samples for the views, shared by all parts
MAX_COMPONENTS = 40  # more components than this for one piece of furniture is flagged as fragmented
ATTRIBUTES = ("POSITION", "NORMAL", "TEXCOORD_0", "TEXCOORD_1", "TANGENT", "COLOR_0")

ComponentType = Literal[
    "panel", "top", "base", "frame", "leg", "door", "drawer", "shelf", "back", "side", "headboard", "footboard",
    "mattress", "cushion", "backrest", "armrest", "seat", "handle", "rail", "plinth", "mirror", "glass", "decor", "other",
]


class ComponentAnswer(BaseModel):
    name: str = Field(description="What a carpenter calls it, in English (Headboard, Left Side Panel, Legs, Doors…). Plural for a set.")
    type: ComponentType
    parts: list[str] = Field(description="The part ids (P1, P2, …) that make this component, as labelled on the map.")
    confidence: float = Field(ge=0, le=1)


class GroupingAnswer(BaseModel):
    furniture_type: str = Field(description="What the furniture is, e.g. double bed, 3-door wardrobe, dining chair.")
    components: list[ComponentAnswer] = Field(default_factory=list)


@dataclass
class Part:
    id: str  # P1, P2 … by surface share
    material: int | None
    name: str  # the mesh / node name from the file (often generic)
    mesh: int
    primitive: int
    world: np.ndarray  # 4×4 node world matrix
    lo: np.ndarray
    hi: np.ndarray
    area: float
    triangles: int
    vertices: int
    color: tuple[float, float, float]
    region: Region
    members: list[tuple[int, int, int]] = field(default_factory=list)  # (instance, mesh, primitive) folded into it


@dataclass
class Component:
    id: str
    name: str
    type: str
    parts: list[Part]
    confidence: float | None
    naming: str  # ai / file / unidentified
    uncertain: bool = False


Ask = Callable[[list[dict[str, Any]], dict[str, Any]], Awaitable[str | None]]


# ── 1–2. Parts and fragments ─────────────────────────────────────────────────


def parts(data: bytes) -> tuple[dict[str, Any], list[bytes], list[Part]]:
    """Every drawn primitive as a part, fragments folded in. Raises Unsupported for files this cannot rebuild safely."""
    try:
        doc, binary = read_glb(data)
    except (ValueError, json.JSONDecodeError) as exc:
        raise Unsupported("not a GLB file") from exc
    used = set(doc.get("extensionsUsed") or [])
    if used & {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_mesh_quantization"}:
        raise Unsupported("compressed geometry")
    if doc.get("skins") or doc.get("animations") or any(p.get("targets") for m in doc.get("meshes") or [] for p in m.get("primitives") or []):
        raise Unsupported("skinned, animated or morphing geometry")
    views = _views(doc, binary)
    materials = doc.get("materials") or []
    found: list[Part] = []
    textures: dict[int, np.ndarray | None] = {}
    rng = np.random.default_rng(11)
    pending: list[tuple] = []
    instances = _instances(doc)
    _instances_world = [world for _, world, _ in instances]
    for instance, (mesh_index, world, node_name) in enumerate(instances):
        mesh = doc["meshes"][mesh_index]
        for prim_index, prim in enumerate(mesh.get("primitives") or []):
            attrs = prim.get("attributes") or {}
            if prim.get("mode", 4) != 4 or "POSITION" not in attrs:
                raise Unsupported("non-triangle primitives")
            try:
                pos = _accessor(doc, views, attrs["POSITION"])[:, :3]
                idx = _accessor(doc, views, prim["indices"])[:, 0].astype(np.int64) if "indices" in prim else np.arange(len(pos))
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise Unsupported("unreadable geometry") from exc
            idx = idx[: len(idx) // 3 * 3]
            if not len(idx):
                continue
            w = pos @ world[:3, :3].T + world[:3, 3]
            tris = w[idx].reshape(-1, 3, 3)
            a = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
            mat = prim.get("material")
            pending.append((instance, mesh_index, prim_index, prim, mesh, node_name, w, idx, tris, a, mat))
    total_area = sum(float(item[9].sum()) for item in pending) or 1.0
    for instance, mesh_index, prim_index, prim, mesh, node_name, w, idx, tris, a, mat in pending:
        attrs = prim.get("attributes") or {}
        pbr = (materials[mat] if isinstance(mat, int) and mat < len(materials) else {}).get("pbrMetallicRoughness", {})
        factor = np.array((pbr.get("baseColorFactor") or [1, 1, 1, 1])[:3], dtype=float)
        if isinstance(mat, int) and mat not in textures:
            textures[mat] = _texture(doc, views, pbr.get("baseColorTexture"))
        texture = textures.get(mat) if isinstance(mat, int) else None
        # Points for the views: the same sampling and colouring as realistic materials (texture through the UVs),
        # in proportion to the surface so large panels read as solid.
        n = max(60, int(POINTS * float(a.sum()) / total_area))
        pick = rng.choice(len(tris), size=n, p=a / a.sum()) if a.sum() > 0 else rng.integers(0, len(tris), size=n)
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        bary = np.stack([1 - u - v, u, v], axis=1)[:, :, None]
        chosen = tris[pick]
        points = (chosen * bary).sum(axis=1)
        normals = np.cross(chosen[:, 1] - chosen[:, 0], chosen[:, 2] - chosen[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        tint = _linear_to_srgb(factor)
        colors = np.tile(tint * 255, (n, 1))
        if texture is not None and "TEXCOORD_0" in attrs:
            try:
                uv = _accessor(doc, views, attrs["TEXCOORD_0"])[:, :2][idx].reshape(-1, 3, 2)
                at = (uv[pick] * bary).sum(axis=1) % 1.0
                h, wd = texture.shape[:2]
                colors = texture[(at[:, 1] * (h - 1)).astype(int), (at[:, 0] * (wd - 1)).astype(int)] * tint
            except (KeyError, IndexError, ValueError, TypeError):
                pass
        region = Region(id="", material=mat if isinstance(mat, int) else -1, name="", meshes=[], color=(0, 0, 0), textured=False,
                        pbr_maps=False, unlit=False, points=points, normals=normals, colors=np.clip(colors, 0, 255))
        found.append(Part(
            id="", material=mat if isinstance(mat, int) else None, name=str(node_name or mesh.get("name") or ""),
            mesh=mesh_index, primitive=prim_index, world=_instances_world[instance], lo=w[idx].min(axis=0), hi=w[idx].max(axis=0),
            area=float(a.sum()), triangles=len(tris), vertices=len(w), color=tuple(colors.mean(axis=0) / 255),  # type: ignore[arg-type]
            region=region, members=[(instance, mesh_index, prim_index)],
        ))
    if not found:
        raise Unsupported("no geometry")
    found = _fold_fragments(found)
    found.sort(key=lambda p: -p.area)
    for i, p in enumerate(found, 1):
        p.id = p.region.id = f"P{i}"
    return doc, views, found


def _gap(a: Part, b: Part) -> float:
    """The distance between two parts' boxes (0 when they overlap)."""
    return float(np.linalg.norm(np.maximum(0, np.maximum(a.lo - b.hi, b.lo - a.hi))))


def _diagonal(found: list[Part]) -> float:
    lo = np.min([p.lo for p in found], axis=0)
    hi = np.max([p.hi for p in found], axis=0)
    return float(np.linalg.norm(hi - lo)) or 1.0


def _merge_into(big: Part, small: Part) -> None:
    big.lo, big.hi = np.minimum(big.lo, small.lo), np.maximum(big.hi, small.hi)
    big.area += small.area
    big.triangles += small.triangles
    big.vertices += small.vertices
    big.members += small.members
    big.region.points = np.concatenate([big.region.points, small.region.points])
    big.region.normals = np.concatenate([big.region.normals, small.region.normals])
    big.region.colors = np.concatenate([big.region.colors, small.region.colors])


def _fold_fragments(found: list[Part]) -> list[Part]:
    diagonal = _diagonal(found)
    touch = TOUCH * diagonal
    found.sort(key=lambda p: -p.area)
    kept: list[Part] = []
    for p in found:
        if float(np.max(p.hi - p.lo)) < FRAGMENT_SIZE * diagonal:
            host = min((k for k in kept if k.material == p.material and _gap(k, p) <= touch), key=lambda k: _gap(k, p), default=None)
            if host is not None:
                _merge_into(host, p)
                continue
        kept.append(p)
    return kept


# ── 3. What the parts are ────────────────────────────────────────────────────


def prompt(found: list[Part], hints: list[str]) -> str:
    lo = np.min([p.lo for p in found], axis=0)
    span = np.maximum(np.max([p.hi for p in found], axis=0) - lo, 1e-9)
    total = sum(p.area for p in found) or 1.0
    rows = []
    for p in found:
        a, b = (p.lo - lo) / span, (p.hi - lo) / span
        rows.append({
            "id": p.id, "x": [round(a[0], 2), round(b[0], 2)], "y_height": [round(a[1], 2), round(b[1], 2)], "z_depth": [round(a[2], 2), round(b[2], 2)],
            "surface_share": round(p.area / total, 3), "colour": "#%02x%02x%02x" % tuple(int(c * 255) for c in p.color),
            **({"file_name": p.name} if p.name and not is_generic(p.name) else {}),
        })
    return (
        "Picture 1 is the product photo. Picture 2 is a 3D model generated from it (front and back three-quarter views); "
        "picture 3 shows the same model with each mesh part in a flat false colour, labelled with its id. The table gives "
        "each part's box as fractions of the whole model (y_height 0 = floor). Group the parts into the furniture's real "
        "physical components, the way a carpenter would list what to build: e.g. Headboard, Left Side Panel, Tabletop, "
        "Legs, Doors, Drawers, Shelves, Mattress. A component may be one part or several parts of the same piece; a set "
        "of identical repeated parts (four legs, two doors) may be one component with a plural name. Never group parts "
        "only because they touch or share a colour, and never use generic names (Part, Mesh, Object). Name every part "
        "you can see clearly, small ones too (handles, rails, shelves); things that are not part of the furniture but "
        "came with the photo (clothes, books, bedding, vases) get type decor and a plain name such as Hanging Clothes. "
        "List each part in at most one component; leave out only parts you truly cannot identify, and give an honest "
        "confidence 0..1 for each component."
        + (f"\nThe photo was analysed before generation as having: {', '.join(hints)}." if hints else "")
        + "\nParts:\n" + json.dumps(rows)
    )


NAME = re.compile(r"^[A-Za-z][A-Za-z0-9 ()/&'\-]{1,39}$")


def clean_name(name: str) -> str | None:
    """An understandable English component name (title case), or None for a generic or malformed one."""
    text = re.sub(r"[_\s]+", " ", name).strip()
    if not NAME.match(text) or is_generic(text) or re.search(r"\b(mesh|object|node|geometry|part)\s*\d", text, re.I):
        return None
    return " ".join(w if w.isupper() and len(w) > 1 else w[:1].upper() + w[1:] for w in text.split(" "))


MOVABLE = {"door", "drawer"}  # each opens on its own: a set of them stays one mesh per element (animation, overlap check)
_NOT_PLURAL = {"clothes", "glass", "chassis", "series", "dress", "press", "brass", "canvas"}


def singular(name: str) -> str:
    """"Legs" → "Leg", "Shelves" → "Shelf", "Hanging Clothes" stays: one element of a named set."""
    words = name.split(" ")
    last = words[-1]
    low = last.lower()
    if low in _NOT_PLURAL or not low.endswith("s") or low.endswith("ss"):
        return name
    if low.endswith("ves"):
        last = last[:-3] + "f"
    elif low.endswith("ies"):
        last = last[:-3] + "y"
    elif low.endswith(("ches", "shes", "xes")):
        last = last[:-2]
    else:
        last = last[:-1]
    return " ".join(words[:-1] + [last])


def _order(parts: list[Part], kind: str) -> list[Part]:
    """Elements of a set in reading order: drawers top to bottom then left to right, others left to right."""
    centre = lambda p: (p.lo + p.hi) / 2  # noqa: E731
    return sorted(parts, key=(lambda p: (-round(float(centre(p)[1]), 2), float(centre(p)[0]))) if kind == "drawer" else (lambda p: float(centre(p)[0])))


def _repeats(members: list[Part]) -> bool:
    """Identical repeated elements: same material, and every size within REPEAT_TOLERANCE of the first's."""
    first = members[0]
    size0 = np.sort(first.hi - first.lo)
    for p in members[1:]:
        size = np.sort(p.hi - p.lo)
        if p.material != first.material or np.any(np.abs(size - size0) > REPEAT_TOLERANCE * np.maximum(size0, 1e-6)):
            return False
    return True


def _clusters(members: list[Part], touch: float) -> list[list[Part]]:
    """The members split into groups that touch (directly or through each other)."""
    left = set(range(len(members)))
    out = []
    while left:
        start = min(left)
        seen, todo = {start}, [start]
        while todo:
            i = todo.pop()
            for j in list(left - seen):
                if _gap(members[i], members[j]) <= touch:
                    seen.add(j)
                    todo.append(j)
        left -= seen
        out.append([members[i] for i in sorted(seen)])
    return out


def _connected(members: list[Part], touch: float) -> bool:
    return len(_clusters(members, touch)) == 1


def validate(answer: GroupingAnswer, found: list[Part]) -> tuple[list[Component], list[str]]:
    """The AI's grouping checked against the geometry; parts it left out (or that failed) stay single components."""
    by_id = {p.id: p for p in found}
    touch = TOUCH * _diagonal(found)
    taken: set[str] = set()
    out: list[Component] = []
    warnings: list[str] = []
    for c in answer.components:
        ids = [i for i in dict.fromkeys(c.parts) if i in by_id and i not in taken]
        name = clean_name(c.name)
        if not ids or c.confidence < MIN_CONFIDENCE:
            continue
        if name is None:
            warnings.append(f"“{c.name}” is not a usable component name; its parts were left as they are.")
            continue
        members = [by_id[i] for i in ids]
        taken.update(ids)
        if len(members) > 1 and not (_connected(members, touch) or _repeats(members)):
            # The name is right but the parts are apart (clothes on two rails): one component per touching cluster,
            # each with the name, rather than one component spanning unrelated places.
            groups = _clusters(members, touch)
            warnings.append(f"“{name}” covered parts in {len(groups)} separate places: kept as {len(groups)} components.")
            label = singular(name)
        else:
            groups, label = [members], name
        for g in groups:
            out.append(Component(id="", name=label, type=c.type, parts=g, confidence=round(c.confidence, 2), naming="ai", uncertain=c.confidence < SURE))
    for p in found:
        if p.id not in taken:
            own = clean_name(p.name) if p.name else None
            out.append(Component(id="", name=own or "", type="other", parts=[p], confidence=None, naming="file" if own else "unidentified"))
    unknown = 0
    for c in out:
        if not c.name:
            unknown += 1
            c.name = f"Unidentified Part {unknown}"
    # Unique names: a name given more than once is numbered on every one (Leg 1 … Leg 4), stable ids from them.
    total: dict[str, int] = {}
    for c in out:
        total[c.name] = total.get(c.name, 0) + 1
    seen: dict[str, int] = {}
    for c in out:
        if total[c.name] > 1:
            seen[c.name] = seen.get(c.name, 0) + 1
            c.name = f"{c.name} {seen[c.name]}"
        c.id = re.sub(r"[^a-z0-9]+", "_", c.name.lower()).strip("_")[:40] or "part"
    if unknown:
        warnings.append(f"{unknown} part(s) could not be identified and keep a neutral name.")
    return out, warnings


# ── 4. One mesh per component ────────────────────────────────────────────────


def _read(doc: dict[str, Any], views: list[bytes], prim: dict[str, Any], name: str) -> np.ndarray | None:
    index = (prim.get("attributes") or {}).get(name)
    return _accessor(doc, views, index) if isinstance(index, int) else None


def rebuild(doc: dict[str, Any], views: list[bytes], components: list[Component]) -> tuple[bytes, list[dict[str, Any]]]:
    """The GLB with one node + mesh per component (a primitive per material), world transforms baked in, the images
    and materials unchanged. Returns the file and, per component, its node / mesh indices."""
    image_views = sorted({img["bufferView"] for img in doc.get("images") or [] if isinstance(img.get("bufferView"), int)})
    new_views = [views[i] for i in image_views]
    remap = {old: new for new, old in enumerate(image_views)}
    out_views: list[dict[str, Any]] = [{"buffer": 0, "byteOffset": 0, "byteLength": len(views[i])} for i in image_views]
    accessors: list[dict[str, Any]] = []

    def add(array: np.ndarray, kind: str, target: int | None) -> int:
        data = np.ascontiguousarray(array, dtype=np.uint32 if kind == "SCALAR" else np.float32)
        out_views.append({"buffer": 0, "byteOffset": 0, "byteLength": data.nbytes, **({"target": target} if target else {})})
        new_views.append(data.tobytes())
        acc: dict[str, Any] = {"bufferView": len(out_views) - 1, "componentType": 5125 if kind == "SCALAR" else 5126, "count": len(data), "type": kind}
        if kind == "VEC3":
            acc["min"], acc["max"] = data.min(axis=0).tolist(), data.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    instances = _instances(doc)
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []  # index 0 is the root, added below
    placed: list[dict[str, Any]] = []
    top: list[int] = []
    def emit(ci: int, comp: Component, element: list[Part], base_id: str, base_name: str) -> list[int]:
        """One mesh node per material for these parts (each tagged with its own part id: the editor selects meshes
        by `extras.partId`, and a multi-primitive mesh would reach it as untagged children)."""
        by_material: dict[Any, list[tuple[int, int, int]]] = {}
        for p in element:
            for member in p.members:
                by_material.setdefault(doc["meshes"][member[1]]["primitives"][member[2]].get("material"), []).append(member)
        out_nodes: list[int] = []
        for k, (material, members) in enumerate(by_material.items()):
            prims = [doc["meshes"][m]["primitives"][q] for _, m, q in members]
            # Every vertex attribute is kept: parts with different attribute sets (or ones not handled here) are not
            # merged at all, rather than merged without some of their UVs or colours.
            sets = {frozenset((q.get("attributes") or {}).keys()) for q in prims}
            if len(sets) != 1 or not next(iter(sets)) <= set(ATTRIBUTES):
                raise Unsupported("parts with different vertex attributes")
            names = [n for n in ATTRIBUTES if n in next(iter(sets))]
            columns: dict[str, list[np.ndarray]] = {n: [] for n in names}
            indices: list[np.ndarray] = []
            offset = 0
            for (instance, _, _), prim in zip(members, prims):
                world = instances[instance][1]
                linear = world[:3, :3]
                mirrored = np.linalg.det(linear) < 0
                normal_m = np.linalg.inv(linear).T
                count = 0
                for n in names:
                    values = _read(doc, views, prim, n)
                    assert values is not None
                    if n == "POSITION":
                        values = values[:, :3] @ linear.T + world[:3, 3]
                        count = len(values)
                    elif n == "NORMAL":
                        values = values[:, :3] @ normal_m.T
                        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
                    elif n == "TANGENT":
                        xyz = values[:, :3] @ linear.T
                        xyz /= np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12)
                        values = np.hstack([xyz, values[:, 3:4] * (-1 if mirrored else 1)])
                    columns[n].append(values)
                idx = _accessor(doc, views, prim["indices"]) if "indices" in prim else None
                tri = (idx[:, 0].astype(np.int64) if idx is not None else np.arange(count))
                tri = tri[: len(tri) // 3 * 3].reshape(-1, 3)
                if mirrored:  # a mirroring transform flips the winding: flipped back so the faces still face out
                    tri = tri[:, ::-1]
                indices.append(tri.reshape(-1) + offset)
                offset += count
            attributes = {}
            for n in names:
                stacked = np.concatenate(columns[n])
                kind = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[stacked.shape[1]]
                attributes[n] = add(stacked, kind, 34962)
            prim_out: dict[str, Any] = {"attributes": attributes, "indices": add(np.concatenate(indices), "SCALAR", 34963), "mode": 4}
            if material is not None:
                prim_out["material"] = material
            part_id = base_id if k == 0 else f"{base_id}_{k + 1}"
            name = base_name if k == 0 else f"{base_name} ({k + 1})"
            meshes.append({"name": name, "primitives": [prim_out]})
            nodes.append({"name": name, "mesh": len(meshes) - 1, "extras": {"partId": part_id, "component": comp.id, "componentType": comp.type}})
            out_nodes.append(len(nodes))  # its index once the root is put first
            placed.append({"component": ci, "id": part_id, "name": name, "node": len(nodes), "mesh": len(meshes) - 1, "material": material, "parts": element})
        return out_nodes

    for ci, comp in enumerate(components):
        if comp.type in MOVABLE and len(comp.parts) > 1 and _repeats(comp.parts):
            # A set of doors or drawers: one component, each element its own mesh (each opens on its own hinge or
            # runner), named and ordered as a carpenter would count them.
            one = singular(comp.name)
            children = [i for n, element in enumerate(_order(comp.parts, comp.type), 1) for i in emit(ci, comp, [element], f"{comp.id}_{n}", f"{one} {n}")]
        else:
            children = emit(ci, comp, comp.parts, comp.id, comp.name)
        if len(children) == 1:
            top.append(children[0])
        else:
            nodes.append({"name": comp.name, "children": children, "extras": {"component": comp.id, "componentType": comp.type}})
            top.append(len(nodes))
    for img in doc.get("images") or []:
        if isinstance(img.get("bufferView"), int):
            img["bufferView"] = remap[img["bufferView"]]
    doc["nodes"] = [{"name": "Furniture", "children": top}] + nodes
    doc["meshes"], doc["accessors"], doc["bufferViews"] = meshes, accessors, out_views
    doc["scenes"], doc["scene"] = [{"nodes": [0]}], 0
    return write_glb(doc, new_views), placed


def _totals(data: bytes) -> tuple[int, int, np.ndarray, np.ndarray]:
    doc, binary = read_glb(data)
    views = _views(doc, binary)
    tris = verts = 0
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for mesh_index, world, _ in _instances(doc):
        for prim in doc["meshes"][mesh_index].get("primitives") or []:
            pos = _accessor(doc, views, prim["attributes"]["POSITION"])[:, :3]
            n = doc["accessors"][prim["indices"]]["count"] if "indices" in prim else len(pos)
            tris += n // 3
            verts += len(pos)
            w = pos @ world[:3, :3].T + world[:3, 3]
            lo, hi = np.minimum(lo, w.min(axis=0)), np.maximum(hi, w.max(axis=0))
    return tris, verts, lo, hi


def check(before: bytes, after: bytes) -> list[str]:
    """What the rebuild changed that it must not: triangles, vertices, bounds, materials, images."""
    problems = []
    t0, v0, lo0, hi0 = _totals(before)
    t1, v1, lo1, hi1 = _totals(after)
    if t0 != t1:
        problems.append(f"triangles {t0} → {t1}")
    if v0 != v1:
        problems.append(f"vertices {v0} → {v1}")
    tolerance = 1e-4 * float(np.linalg.norm(hi0 - lo0) or 1)
    if np.any(np.abs(lo0 - lo1) > tolerance) or np.any(np.abs(hi0 - hi1) > tolerance):
        problems.append("bounds changed")
    d0, _ = read_glb(before)
    d1, _ = read_glb(after)
    if d0.get("materials") != d1.get("materials") or len(d0.get("images") or []) != len(d1.get("images") or []):
        problems.append("materials or images changed")
    return problems


# ── The whole stage ──────────────────────────────────────────────────────────


def _document(components: list[Component], placed: list[dict[str, Any]], doc: dict[str, Any], real_height_m: float | None) -> list[dict[str, Any]]:
    every = [p for c in components for p in c.parts]
    lo = np.min([p.lo for p in every], axis=0)
    hi = np.max([p.hi for p in every], axis=0)
    span = np.maximum(hi - lo, 1e-9)
    to_m = real_height_m / span[1] if real_height_m else None  # model units → metres, when the real size is known
    materials = doc.get("materials") or []
    out = []
    for where in placed:
        comp = components[where["component"]]
        c_lo = np.min([p.lo for p in where["parts"]], axis=0)
        c_hi = np.max([p.hi for p in where["parts"]], axis=0)
        size = c_hi - c_lo
        m = where["material"]
        out.append({
            "id": where["id"], "name": where["name"], "component": comp.id, "type": comp.type, "mesh": where["mesh"], "primitive": 0,
            "node": where["node"], "material": m, "editable": True, "naming": comp.naming,
            "confidence": comp.confidence, **({"uncertain": True} if comp.uncertain else {}),
            "count": len(where["parts"]),
            "box": [[round(float(v), 4) for v in c_lo], [round(float(v), 4) for v in c_hi]],
            # Where it sits in the piece (0..1 of the whole: x left→right, y floor→top, z back→front).
            "position": [round(float(v), 3) for v in ((c_lo + c_hi) / 2 - lo) / span],
            "size": {"relative": [round(float(v), 3) for v in size / span], **({"m": [round(float(v * to_m), 3) for v in size]} if to_m else {})},
            **({"materialInfo": {"name": materials[m].get("name") or f"Material {m + 1}", "textured": "baseColorTexture" in (materials[m].get("pbrMetallicRoughness") or {}),
                                 "color": "#%02x%02x%02x" % tuple(int(c * 255) for c in next(p.color for p in where["parts"] if p.material == m))}}
               if isinstance(m, int) and m < len(materials) else {}),
        })
    return out


async def group(data: bytes, photo: bytes | None, ask: Ask | None, hints: list[str] | None = None, real_height_m: float | None = None) -> tuple[bytes, dict[str, Any]] | None:
    """The grouped GLB and its component document ({components, grouping}), or None to keep the input as it is (with
    the reason logged): unsupported files, no usable answer, or a rebuild that would change the geometry."""
    try:
        doc, views, found = parts(data)
    except Unsupported as exc:
        logger.info("Component grouping skipped: %s", exc)
        return None
    if len(found) < 2:
        return None  # one piece: nothing to group (the split step said why)
    if len(found) > MAX_PARTS_FOR_AI or ask is None:
        logger.info("Component grouping skipped: %d parts, vision %s", len(found), "configured" if ask else "not configured")
        return None
    look, regionmap = render([p.region for p in found])
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt(found, hints or [])}]
    if photo:
        mime = "image/jpeg" if photo[:2] == b"\xff\xd8" else "image/png"
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(photo).decode()}"})
    for png in (look, regionmap):
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{base64.b64encode(png).decode()}"})
    try:
        answer = GroupingAnswer.model_validate_json((await ask(content, GroupingAnswer.model_json_schema()) or "").strip())
    except ValidationError as exc:
        logger.warning("Component grouping: no usable answer (%s)", str(exc)[:200])
        return None
    components, warnings = validate(answer, found)
    if not any(c.naming == "ai" for c in components):
        logger.info("Component grouping: no component identified with enough confidence")
        return None
    try:
        glb, placed = rebuild(json.loads(json.dumps(doc)), views, components)
    except (Unsupported, ValueError, KeyError, IndexError) as exc:
        logger.warning("Component grouping not applied: %s", exc)
        return None
    problems = check(data, glb)
    if problems:
        logger.warning("Component grouping discarded: the rebuild changed the model (%s)", "; ".join(problems))
        return None
    if len(components) > MAX_COMPONENTS:
        warnings.append(f"{len(components)} components is a lot for one piece of furniture: some may still be fragments.")
    if any(c.uncertain for c in components):
        warnings.append("Some components were identified with low confidence: check their names.")
    before = sum(len(p.members) for p in found)
    return glb, {
        "components": _document(components, placed, doc, real_height_m),
        "grouping": {"method": "vision+geometry", "furnitureType": answer.furniture_type.strip()[:80], "partsBefore": before,
                     "components": len(components), "warnings": warnings},
    }
