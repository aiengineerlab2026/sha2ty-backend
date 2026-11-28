"""A carpenter model beside the visual one: closed wooden solids, not a renamed copy of the Meshy mesh.

The visual GLB is never read or written here. Each component is a box taken from the part's measured bounds
(the same axes as the model: x width, y height, z depth). Bedding and loose decoration are left out. A part
that is only upholstery becomes an inferred board on its structural face. Hidden structure a type usually
needs is proposed only when that part is missing, and is marked inferred. Nothing here is a cutting list.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from app.services import material_spec
from app.services.generation_prompt import furniture_family
from app.services.glb_size import write_glb

BOARD_CM = 1.8
PANE_CM = 0.6  # glass or mirror
NOTE = "Not fabrication-ready until dimensions, materials, structural design, and assembly are verified."

# Longer words first so "backrest" is not read as a wardrobe back.
_ROLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("headboard", ("headboard",)),
    ("footboard", ("footboard", "foot board")),
    ("backrest", ("backrest", "back rest")),
    ("armrest", ("armrest", "arm rest", "arm")),
    ("seat", ("seat",)),
    ("mattress", ("mattress",)),
    ("cushion", ("cushion", "pillow", "blanket", "duvet", "bedding", "quilt")),
    ("decor", ("decor", "ornament", "vase", "plant", "flower", "book", "clothes", "clothing", "rug", "carpet", "lamp", "sculpture")),
    ("drawer", ("drawer",)),
    ("door", ("door",)),
    ("shelf", ("shelf",)),
    ("leg", ("leg",)),
    ("rail", ("rail", "apron", "slat")),
    ("side", ("side",)),
    ("top", ("top", "tabletop")),
    ("base", ("base", "plinth")),
    ("back", ("back",)),
    ("frame", ("frame",)),
    ("glass", ("glass", "mirror")),
    ("handle", ("handle", "knob", "pull")),
)
_EXCLUDE = {"mattress", "cushion", "decor"}
_UPHOLSTERY = {"seat", "backrest", "armrest"}
_WOOD = {"headboard", "footboard", "side", "leg", "rail", "shelf", "door", "drawer", "top", "base", "back", "frame", "panel"}
# Roles a type is incomplete without. Added only when absent, inside the measured box, and marked inferred.
_MISSING: dict[str, tuple[str, ...]] = {
    "bed": ("headboard", "footboard", "side", "base"),
    "wardrobe": ("side", "top", "base", "back"),
    "table": ("top", "leg"),
    "desk": ("top", "leg"),
    "sofa": ("base", "backrest", "armrest"),
    "chair": ("seat", "leg", "backrest"),
    "cabinet": ("side", "top", "base", "back"),
}


def reconstruct(components: dict[str, Any], details: dict[str, Any] | None = None, kind: str | None = None, category: str | None = None) -> tuple[bytes, dict[str, Any]] | None:
    """The carpenter GLB and its report. None when the parts have no measured boxes to rebuild from."""
    parts = [p for p in components.get("components") or [] if _box(p) is not None]
    if not parts:
        return None
    # The envelope: handles stand proud of the fronts and do not widen the carcass.
    body = [p for p in parts if _role(p) != "handle"] or parts
    lo = np.min([_box(p)[0] for p in body], axis=0)
    hi = np.max([_box(p)[1] for p in body], axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scales, estimated = _scales(span, details)
    lo, hi = lo * scales, hi * scales
    board = np.full(3, BOARD_CM / 100.0)
    family = _family(components, details, kind, category)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    roles: set[str] = set()
    carcass: dict[str, Any] = {}  # the whole body's spec entry and id: its inferred panels are made of it
    for part in parts:
        role = _role(part)
        box = (_box(part)[0] * scales, _box(part)[1] * scales)
        spec = part.get("material")  # the shared material spec entry (services/material_spec.py), when built from it
        if part.get("carcass") and family in _MISSING:  # the whole body: built as its panels below, not one block
            excluded.append({"name": str(part.get("name") or role), "reason": "The whole body: built as its separate panels."})
            carcass = {"spec": part.get("material"), "spec_id": part.get("specId")}
            continue
        if role in _EXCLUDE or (role in _UPHOLSTERY and "frame" not in str(part.get("name") or "").lower()):
            excluded.append({"name": str(part.get("name") or role), "reason": "Not part of the wooden structure."})
            if role in _UPHOLSTERY:
                kept.append(_solid(_frame_box(box, role, board), _frame_name(role), role, "inferred", "wood", lo, estimated, spec_id=part.get("specId")))
                roles.add(role)
            continue
        material = _category(spec) if spec else "glass" if role == "glass" or _pane(part) else "metal" if role == "handle" else "wood"
        source = "inferred" if part.get("inferred") else "observed"
        # A glass or mirror pane keeps its own thickness (at least PANE_CM): padded to a board it would pass through
        # the wood behind it and show on the inside face.
        kept.append(_solid(_thicken(box, np.full(3, PANE_CM / 100.0) if material == "glass" else board), str(part.get("name") or role), role, source, material, lo, estimated, spec, part.get("specId")))
        roles.add(role)
    for role in _MISSING.get(family, ()):
        if role in roles:
            continue
        for name, box in _missing(role, lo, hi, board, family):
            spec = carcass.get("spec")
            kept.append(_solid(box, name, role, "inferred", _category(spec) if spec else "wood", lo, estimated, spec, carcass.get("spec_id")))
    if not kept:
        return None
    for index, part in enumerate(kept, 1):
        part["id"] = f"{_slug(part['name'])}_{index}"
    checks = _checks(kept, lo, hi)
    report = {
        "version": 1,
        "fabricationReady": False,
        "estimated": estimated,
        "furnitureType": family,
        "boardThicknessCm": BOARD_CM,
        "note": NOTE,
        "hierarchy": _hierarchy(kept),
        "components": [{k: v for k, v in part.items() if k not in ("box", "materialSpec")} for part in kept],
        "excluded": excluded,
        "checks": checks,
    }
    return _glb(kept), report


def _family(components: dict[str, Any], details: dict[str, Any] | None, kind: str | None, category: str | None) -> str:
    grouping = components.get("grouping") or {}
    hint = " ".join(str(x) for x in (kind, (details or {}).get("kind"), grouping.get("furnitureType")) if x)
    family = furniture_family(category or (details or {}).get("category"), hint)
    if family in {"furniture", "decor", ""} and re.search(r"\b(cabinet|cupboard|dresser|sideboard)\b", hint.lower()):
        return "cabinet"
    return family


def _scales(span: np.ndarray, details: dict[str, Any] | None) -> tuple[np.ndarray, bool]:
    """Model units to metres. A given overall size is used per axis; without one, the file is treated as metres and flagged."""
    given = [(details or {}).get(key) for key in ("width_m", "height_m", "depth_m")]
    scales = np.array([float(value) / float(span[i]) if value else 1.0 for i, value in enumerate(given)])
    return scales, not bool((details or {}).get("scale_known"))


_HANDLE = re.compile(r"\b(?:handles?|knobs?|grips?)\b|\bpulls?\b(?![- ]out)")


def _role(part: dict[str, Any]) -> str:
    blob = f"{part.get('type') or ''} {part.get('name') or ''}".lower()
    # A handle is named after what it is on ("Door 1 handle", "drawer pull"): it is the handle, not the door. Read as a
    # door it counted in the carcass envelope and pushed the inferred side panels out to the handles' front face.
    if _HANDLE.search(blob):
        return "handle"
    for role, words in _ROLES:
        if any(re.search(rf"\b{re.escape(word)}\b", blob) for word in words):
            return role
    return str(part.get("type") or "other")


def _pane(part: dict[str, Any]) -> bool:
    """Named as a door's glass or mirror ("Door mirror"), not its frame or the wood behind it ("mirror backing")."""
    name = str(part.get("name") or "").lower()
    return bool(re.search(r"\b(mirror|glass)\b", name)) and not re.search(r"\b(backing|frame|back)\b", name)


def _box(part: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    raw = part.get("box")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    lo, hi = np.array(raw[0], dtype=float), np.array(raw[1], dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(hi < lo):
        return None
    return lo, hi


def _thicken(box: tuple[np.ndarray, np.ndarray], board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = box
    size = hi - lo
    thin = int(np.argmin(size))
    if size[thin] >= board[thin]:
        return lo, hi
    mid = (lo[thin] + hi[thin]) / 2
    lo, hi = lo.copy(), hi.copy()
    if thin == 2:  # a front-facing panel keeps its seen face; it thickens backwards, never into what is in front of it
        lo[2] = hi[2] - board[2]
    else:
        lo[thin], hi[thin] = mid - board[thin] / 2, mid + board[thin] / 2
    return lo, hi


def _frame_box(box: tuple[np.ndarray, np.ndarray], role: str, board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A board on the structural face of an upholstered volume: under a seat, behind a back, on the outer side of an arm."""
    lo, hi = (v.copy() for v in box)
    if role == "backrest":
        hi[2] = lo[2] + board[2]
    elif role == "armrest":
        if lo[0] + hi[0] >= 0:
            lo[0] = hi[0] - board[0]
        else:
            hi[0] = lo[0] + board[0]
    else:
        hi[1] = lo[1] + board[1]
    return lo, hi


def _frame_name(role: str) -> str:
    return {"seat": "Seat frame", "backrest": "Backrest frame", "armrest": "Armrest frame"}.get(role, "Frame")


def _missing(role: str, lo: np.ndarray, hi: np.ndarray, board: np.ndarray, family: str = "") -> list[tuple[str, tuple[np.ndarray, np.ndarray]]]:
    span = hi - lo
    # A storage piece's sides run its full height; a bed's side rails about a third of it.
    height = hi[1] if family in ("wardrobe", "cabinet") else lo[1] + max(span[1] * 0.3, board[1])
    if role == "side":
        return [
            ("Left Side Panel", (lo.copy(), np.array([lo[0] + board[0], height, hi[2]]))),
            ("Right Side Panel", (np.array([hi[0] - board[0], lo[1], lo[2]]), np.array([hi[0], height, hi[2]]))),
        ]
    if role == "armrest":
        top = lo[1] + span[1] * 0.8
        bottom = lo[1] + span[1] * 0.4
        return [
            ("Left Armrest Frame", (np.array([lo[0], bottom, lo[2]]), np.array([lo[0] + board[0], top, hi[2]]))),
            ("Right Armrest Frame", (np.array([hi[0] - board[0], bottom, lo[2]]), np.array([hi[0], top, hi[2]]))),
        ]
    if role == "leg":
        reach = hi[1] - board[1]
        if reach <= lo[1]:
            reach = lo[1] + board[1]
        spots = ((lo[0], lo[2]), (hi[0] - board[0], lo[2]), (lo[0], hi[2] - board[2]), (hi[0] - board[0], hi[2] - board[2]))
        return [(f"Leg {i}", (np.array([x, lo[1], z]), np.array([x + board[0], reach, z + board[2]]))) for i, (x, z) in enumerate(spots, 1)]
    one = {
        "headboard": ("Headboard Frame", (np.array([lo[0], lo[1] + span[1] * 0.45, lo[2]]), np.array([hi[0], hi[1], lo[2] + board[2]]))),
        "footboard": ("Footboard", (np.array([lo[0], lo[1], hi[2] - board[2]]), np.array([hi[0], lo[1] + span[1] * 0.35, hi[2]]))),
        "base": ("Base Supports", (lo.copy(), np.array([hi[0], lo[1] + board[1], hi[2]]))),
        "top": ("Top Panel", (np.array([lo[0], hi[1] - board[1], lo[2]]), hi.copy())),
        "back": ("Back Panel", (lo.copy(), np.array([hi[0], hi[1], lo[2] + board[2]]))),
        "seat": ("Seat frame", _frame_box((lo, hi), "seat", board)),
        "backrest": ("Backrest frame", _frame_box((lo, hi), "backrest", board)),
    }.get(role)
    return [one] if one else []


def _category(spec: dict[str, Any]) -> str:
    """The report's coarse material category (wood / glass / metal / upholstery) for a material spec entry."""
    cls = spec.get("class")
    return {"glass": "glass", "mirror": "glass", "metal": "metal", "fabric": "upholstery", "leather": "upholstery"}.get(cls, "wood")


def _solid(box: tuple[np.ndarray, np.ndarray], name: str, role: str, source: str, material: str, origin: np.ndarray, estimated: bool,
           spec: dict[str, Any] | None = None, spec_id: str | None = None) -> dict[str, Any]:
    lo, hi = box
    size = (hi - lo) * 100
    center = (lo + hi) / 2 - origin
    return {
        "name": name, "role": role, "source": source, "materialCategory": material,
        **({"specId": spec_id} if spec_id else {}),
        **({"materialClass": spec.get("class"), "materialSpec": spec} if spec else {}),
        "classification": "inferred" if source == "inferred" else "functional" if material != "wood" else "structural",
        "sizeCm": {"width": round(float(size[0]), 1), "height": round(float(size[1]), 1), "depth": round(float(size[2]), 1)},
        "positionCm": {"x": round(float(center[0] * 100), 1), "y": round(float(center[1] * 100), 1), "z": round(float(center[2] * 100), 1)},
        "orientation": [0, 0, 0], "estimated": estimated, "box": (lo, hi),
    }


def _checks(parts: list[dict[str, Any]], lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
    warnings: list[str] = []
    for part in parts:
        if not _watertight(_corners(*part["box"])):
            warnings.append(f"{part['name']} is not a closed solid.")
        if part["source"] == "inferred":
            warnings.append(f"{part['name']} is inferred, not seen in the model.")
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            share = _overlap_share(a["box"], b["box"])
            if share > 0.25:
                warnings.append(f"{a['name']} and {b['name']} occupy the same space ({share:.0%}).")
    bounds = np.array([_corners(*part["box"])[0].min(axis=0) for part in parts])
    highs = np.array([_corners(*part["box"])[0].max(axis=0) for part in parts])
    grown = np.maximum(lo - bounds.min(axis=0), highs.max(axis=0) - hi)
    if float(np.max(grown) * 100) > 5:
        warnings.append("The carpenter model extends more than 5 cm outside the visual model.")
    return {"closed": not any("not a closed solid" in w for w in warnings), "triangles": 12 * len(parts), "warnings": warnings}


def _hierarchy(parts: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {"Structural Components": [], "Functional parts": [], "Inferred structure": []}
    for part in parts:
        key = "Inferred structure" if part["source"] == "inferred" else "Functional parts" if part["materialCategory"] != "wood" else "Structural Components"
        groups[key].append(part["name"])
    return {"name": "Furniture Model", "children": [{"name": name, "components": items} for name, items in groups.items() if items]}


def _glb(parts: list[dict[str, Any]]) -> bytes:
    """One mesh per component. Each accessor reads only that part's 8 corners and 12 triangles."""
    positions, indices, nodes, meshes, accessors = [], [], [], [], []
    materials = [{"name": "Wood", "pbrMetallicRoughness": {"baseColorFactor": [0.72, 0.58, 0.4, 1], "roughnessFactor": 0.6}},
                 {"name": "Glass", "pbrMetallicRoughness": {"baseColorFactor": [0.75, 0.85, 0.9, 0.4], "roughnessFactor": 0.05}},
                 {"name": "Metal", "pbrMetallicRoughness": {"baseColorFactor": [0.6, 0.6, 0.62, 1], "metallicFactor": 1, "roughnessFactor": 0.4}}]
    which = {"wood": 0, "glass": 1, "metal": 2}
    which = {**which, "upholstery": 0}
    used: set[str] = set()
    for part in parts:
        corners, faces = _corners(*part["box"])
        positions.append(corners.astype(np.float32))
        indices.append(faces.astype(np.uint32))
        if part.get("materialSpec"):  # its own material from the shared spec: class, colour, finish
            mat, ext = material_spec.gltf_material(part["materialSpec"], part["name"])
            materials.append(mat)
            used |= ext
            index = len(materials) - 1
        else:
            index = which[part["materialCategory"]]
        meshes.append({"name": part["name"], "primitives": [{"attributes": {"POSITION": len(accessors)}, "indices": len(parts) + len(accessors), "mode": 4, "material": index}]})
        accessors.append({"bufferView": 0, "byteOffset": len(positions) * 8 * 12 - 96, "componentType": 5126, "count": 8, "type": "VEC3", "min": corners.min(0).tolist(), "max": corners.max(0).tolist()})
        nodes.append({"name": part["name"], "mesh": len(meshes) - 1, "extras": {"partId": part["id"], "source": part["source"], "role": part["role"], **({"specId": part["specId"]} if part.get("specId") else {})}})
    for index, faces in enumerate(indices):
        accessors.append({"bufferView": 1, "byteOffset": index * 36 * 4, "componentType": 5125, "count": 36, "type": "SCALAR"})  # 12 triangles
    doc: dict[str, Any] = {
        "asset": {"version": "2.0"}, "scene": 0,
        "scenes": [{"nodes": [len(nodes)]}],
        "nodes": nodes + [{"name": "Carpenter model", "children": list(range(len(nodes)))}],
        "meshes": meshes, "materials": materials, "accessors": accessors,
        "buffers": [{}], "bufferViews": [{}, {"target": 34963}],
        **({"extensionsUsed": sorted(used)} if used else {}),
    }
    return write_glb(doc, [np.concatenate(positions).tobytes(), np.concatenate(indices).tobytes()])


def _corners(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]], dtype=np.int64)
    return corners, faces


def _watertight(mesh: tuple[np.ndarray, np.ndarray]) -> bool:
    corners, faces = mesh
    edges: dict[tuple[int, int], int] = {}
    for tri in faces:
        a, b, c = corners[tri]
        if np.linalg.norm(np.cross(b - a, c - a)) < 1e-12:
            return False
        for u, v in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
            edges[(u, v)] = edges.get((u, v), 0) + 1
    paired = {tuple(sorted(edge)) for edge in edges}
    return len(faces) == 12 and len(edges) == len(paired) * 2 and all(sum(edges.get(order, 0) for order in (edge, edge[::-1])) == 2 for edge in paired)


def _overlap_share(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    if np.any(hi <= lo):
        return 0.0
    inter = float(np.prod(hi - lo))
    smaller = min(float(np.prod(a[1] - a[0])), float(np.prod(b[1] - b[0])))
    return inter / smaller if smaller else 0.0


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "part"
