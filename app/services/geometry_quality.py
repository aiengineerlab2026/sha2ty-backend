"""Geometry validation, safe repair and an AR-optimised copy for generated furniture models (any type).

Runs on the GLB after its components are grouped and named (component_grouping, part_names):

1. `inspect`: per part (each primitive), degenerate and duplicate triangles, invalid normals, open boundary edges,
   non-manifold edges and how many separate pieces it has, on vertices welded by position (UV seams are not holes).
2. `repair`: only what cannot change how the model looks: zero-area and duplicate triangles removed, invalid normals
   rebuilt from the faces around them, unused vertices dropped. Each repair is verified (bounds and visible area
   unchanged) and reverted when it is not. Holes are never closed (seats, shells and panels are open on purpose)
   and vertices are never welded (that would break UV seams).
3. Materials: a material used by several parts is given a copy per part (textures stay shared), so recolouring one
   part never changes another, in the editor, the planner, AR or an export.
4. `classify` and `dimensions`: structural / upholstery / decorative / other from each component's type and name,
   and its bounding box in centimetres when the model's real size is known (marked uncalibrated when that size is
   only the photo's estimate). Bounding boxes are not cutting sizes and are never presented as such.
5. `ar_copy`: each heavy part decimated (fast_simplification, quadric, open borders and UV seams kept), keeping only
   original vertices (so every UV and normal is the vertex's own), at the lightest level that passes its checks:
   same bounds, same surface area, and the same surface when both are voxelised. A part that fails keeps its
   original geometry. The full model stays the one edited; the copy is for mobile AR.

Nothing here claims a model is ready for manufacturing: the report says what was measured and what was not.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

from app.services.glb_size import _views, read_glb, write_glb
from app.services.realistic_materials import Unsupported, _accessor, _instances

logger = logging.getLogger(__name__)

VERSION = 1
ATTRIBUTES = ("POSITION", "NORMAL", "TEXCOORD_0", "TEXCOORD_1", "TANGENT", "COLOR_0")
WELD = 1e-6  # positions closer than this share of the part's diagonal are one point for topology
AREA_EPS = 1e-12  # a triangle under this share of the part's squared diagonal has no area
MIN_DECIMATE = 3000  # parts lighter than this are left alone in the AR copy
KEEP_LEVELS = (0.12, 0.25, 0.5)  # the AR copy tries the lightest first
BOUNDS_TOLERANCE = 0.005  # of the model's diagonal
# Surface area is a sanity check, not the quality measure: generated surfaces are finely bumpy and a simplified one
# spans the bumps, so its area moves a few percent while the shape (the voxel match below) stays the same.
AREA_TOLERANCE = 0.08
SURFACE_VOXEL = 0.004  # of the model's diagonal
SURFACE_MATCH = 0.97  # share of sampled surface found again (both ways), one voxel apart at most
AR_WORTH = 0.8  # an AR copy is made only when it has at most this share of the triangles
SAMPLES = 60_000


# ── Reading and writing primitives ───────────────────────────────────────────


def _prims(doc: dict[str, Any], views: list[bytes]) -> dict[tuple[int, int], dict[str, Any]]:
    """Every primitive's attributes and triangles as arrays (its own space). Unsupported for what cannot be rewritten."""
    used = set(doc.get("extensionsUsed") or [])
    if used & {"KHR_draco_mesh_compression", "EXT_meshopt_compression"}:
        raise Unsupported("compressed geometry")
    if doc.get("skins") or doc.get("animations"):
        raise Unsupported("skinned or animated geometry")
    out = {}
    for m, mesh in enumerate(doc.get("meshes") or []):
        for k, prim in enumerate(mesh.get("primitives") or []):
            attrs = prim.get("attributes") or {}
            if prim.get("targets") or prim.get("mode", 4) != 4 or not set(attrs) <= set(ATTRIBUTES) or "POSITION" not in attrs:
                raise Unsupported("primitives this stage cannot rewrite (morph targets, lines, points or other attributes)")
            arrays = {n: _accessor(doc, views, i) for n, i in attrs.items()}
            n = len(arrays["POSITION"])
            tri = _accessor(doc, views, prim["indices"])[:, 0].astype(np.int64) if "indices" in prim else np.arange(n)
            out[(m, k)] = {"attrs": arrays, "tri": tri[: len(tri) // 3 * 3].reshape(-1, 3)}
    return out


def _encode(doc: dict[str, Any], views: list[bytes], prims: dict[tuple[int, int], dict[str, Any]]) -> bytes:
    """The file with these primitives' geometry (float vertex data, uint32 indices); images and everything else as is."""
    doc = json.loads(json.dumps(doc))
    image_views = sorted({img["bufferView"] for img in doc.get("images") or [] if isinstance(img.get("bufferView"), int)})
    new_views = [views[i] for i in image_views]
    remap = {old: new for new, old in enumerate(image_views)}
    out_views: list[dict[str, Any]] = [{"buffer": 0, "byteOffset": 0, "byteLength": len(views[i])} for i in image_views]
    accessors: list[dict[str, Any]] = []

    def add(array: np.ndarray, kind: str, target: int, integer: bool = False) -> int:
        data = np.ascontiguousarray(array, dtype=np.uint32 if integer else np.float32)
        out_views.append({"buffer": 0, "byteOffset": 0, "byteLength": data.nbytes, "target": target})
        new_views.append(data.tobytes())
        acc: dict[str, Any] = {"bufferView": len(out_views) - 1, "componentType": 5125 if integer else 5126, "count": len(data), "type": kind}
        if kind == "VEC3" and not integer:
            acc["min"], acc["max"] = data.min(axis=0).tolist(), data.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    for (m, k), p in prims.items():
        prim = doc["meshes"][m]["primitives"][k]
        prim["attributes"] = {n: add(a, {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[a.shape[1]], 34962) for n, a in p["attrs"].items()}
        prim["indices"] = add(p["tri"].reshape(-1), "SCALAR", 34963, integer=True)
        prim["mode"] = 4
    for img in doc.get("images") or []:
        if isinstance(img.get("bufferView"), int):
            img["bufferView"] = remap[img["bufferView"]]
    doc["accessors"], doc["bufferViews"] = accessors, out_views
    for key in ("extensionsUsed", "extensionsRequired"):  # every vertex is float again: no quantization left
        kept = [e for e in doc.get(key) or [] if e != "KHR_mesh_quantization"]
        if kept:
            doc[key] = kept
        else:
            doc.pop(key, None)
    return write_glb(doc, new_views)


def _rows(a: np.ndarray, return_inverse: bool = False, return_counts: bool = False, return_index: bool = False):
    """np.unique(a, axis=0) for small non-negative integer rows, much faster: each row packed into one int64 key
    (a 1-D sort instead of numpy's row sort, which dominated on million-triangle models). Same results."""
    a = np.asarray(a, dtype=np.int64)
    width = a.shape[1] if a.ndim == 2 else 0
    bits = 63 // max(width, 1)
    if not len(a) or a.min() < 0 or int(a.max()) >= (1 << bits):
        return np.unique(a, axis=0, return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)
    key = np.zeros(len(a), dtype=np.int64)
    for i in range(width):
        key = (key << bits) | a[:, i]
    out = np.unique(key, return_index=True, return_inverse=return_inverse, return_counts=return_counts)
    first = out[1]
    result = [a[first]]
    if return_index:
        result.append(first)
    rest = list(out[2:])
    if return_inverse:
        result.append(rest.pop(0).reshape(-1))
    if return_counts:
        result.append(rest.pop(0))
    return tuple(result) if len(result) > 1 else result[0]


def _areas(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    if not len(tri):
        return np.zeros(0)
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _diag(pos: np.ndarray) -> float:
    return float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))) if len(pos) else 0.0


# ── 1. Inspection ────────────────────────────────────────────────────────────


def _pieces(n: int, edges: np.ndarray) -> int | None:
    """Connected pieces among the welded points that triangles use (label propagation with pointer jumping)."""
    if not len(edges):
        return 0
    label = np.arange(n)
    for _ in range(200):
        a, b = label[edges[:, 0]], label[edges[:, 1]]
        low = np.minimum(a, b)
        before = label.copy()
        np.minimum.at(label, a, low)
        np.minimum.at(label, b, low)
        for _ in range(64):
            jumped = label[label]
            if np.array_equal(jumped, label):
                break
            label = jumped
        if np.array_equal(before, label):
            used = np.unique(edges)
            return int(len(np.unique(label[used])))
    return None  # not settled: reported as not measured


def inspect_primitive(attrs: dict[str, np.ndarray], tri: np.ndarray) -> dict[str, Any]:
    pos = attrs["POSITION"][:, :3]
    diag = _diag(pos) or 1.0
    repeated = (tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])
    degenerate = repeated | (_areas(pos, tri) <= AREA_EPS * diag * diag)
    # The same triangle with the same winding, whichever corner it starts from.
    rolled = np.take_along_axis(tri, (np.arange(3)[None, :] + np.argmin(tri, axis=1)[:, None]) % 3, axis=1) if len(tri) else tri
    duplicate = len(tri) - len(_rows(rolled)) if len(tri) else 0
    normals = attrs.get("NORMAL")
    invalid_normals = int((~np.isfinite(normals).all(axis=1) | (np.linalg.norm(np.nan_to_num(normals[:, :3]), axis=1) < 0.5)).sum()) if normals is not None else 0
    # Topology on positions welded together: duplicated seam vertices are not open edges.
    grid = np.round((pos - pos.min(axis=0)) / (WELD * diag)).astype(np.int64)
    _, welded = _rows(grid, return_inverse=True)
    good = tri[~degenerate]
    wt = welded.reshape(-1)[good] if len(good) else good
    edges = np.sort(np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]], wt[:, [2, 0]]]), axis=1) if len(wt) else np.zeros((0, 2), np.int64)
    if len(edges):
        unique, counts = _rows(edges, return_counts=True)
    else:
        unique, counts = edges, np.zeros(0, np.int64)
    open_edges, non_manifold = int((counts == 1).sum()), int((counts > 2).sum())
    return {
        "triangles": int(len(tri)), "vertices": int(len(pos)), "degenerate": int(degenerate.sum()), "duplicate": int(duplicate),
        "invalidNormals": invalid_normals, "openEdges": open_edges, "nonManifoldEdges": non_manifold,
        "pieces": _pieces(int(welded.max()) + 1 if len(welded) else 0, unique), "watertight": bool(len(edges) and open_edges == 0 and non_manifold == 0),
    }


# ── 2. Safe repair ───────────────────────────────────────────────────────────


def _compact(attrs: dict[str, np.ndarray], tri: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    used, new = np.unique(tri, return_inverse=True)
    return {n: a[used] for n, a in attrs.items()}, new.reshape(-1, 3)


def _visible_area(pos: np.ndarray, tri: np.ndarray) -> float:
    """The area that is drawn: each distinct triangle once (a duplicate draws over the same place)."""
    return float(_areas(pos, _rows(np.sort(tri, axis=1))).sum()) if len(tri) else 0.0


def _bounds(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    pts = pos[np.unique(tri)] if len(tri) else pos
    return np.stack([pts.min(axis=0), pts.max(axis=0)])


def repair_primitive(attrs: dict[str, np.ndarray], tri: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """The primitive with only invisible defects fixed, and what was done. Anything that would change what is drawn
    (its bounds or its visible area) is reverted."""
    done: list[str] = []
    pos = attrs["POSITION"][:, :3]
    diag = _diag(pos) or 1.0
    bounds_before = _bounds(pos, tri)
    area_before = _visible_area(pos, tri)

    repeated = (tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])
    degenerate = repeated | (_areas(pos, tri) <= AREA_EPS * diag * diag)
    candidate = tri[~degenerate]
    rolled = np.take_along_axis(candidate, (np.arange(3)[None, :] + np.argmin(candidate, axis=1)[:, None]) % 3, axis=1) if len(candidate) else candidate
    if len(candidate):
        _, first = _rows(rolled, return_index=True)
        dedup = candidate[np.sort(first)]
    else:
        dedup = candidate
    removed_degenerate, removed_duplicate = int(degenerate.sum()), int(len(candidate) - len(dedup))
    if (removed_degenerate or removed_duplicate) and len(dedup):
        area_after = _visible_area(pos, dedup)
        same_bounds = np.allclose(_bounds(pos, dedup), bounds_before, atol=1e-6 * diag)
        same_area = abs(area_after - area_before) <= 1e-6 * max(area_before, 1e-12)
        if same_bounds and same_area:
            tri = dedup
            if removed_degenerate:
                done.append(f"removed {removed_degenerate} zero-area triangle(s)")
            if removed_duplicate:
                done.append(f"removed {removed_duplicate} duplicate triangle(s)")
        else:
            done.append("kept its zero-area / duplicate triangles: removing them would change the drawn shape")

    normals = attrs.get("NORMAL")
    if normals is not None:
        n3 = normals[:, :3]
        bad = ~np.isfinite(n3).all(axis=1) | (np.linalg.norm(np.nan_to_num(n3), axis=1) < 0.5)
        if bad.any():
            face = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])  # area-weighted
            acc = np.zeros_like(pos)
            for i in range(3):
                np.add.at(acc, tri[:, i], face)
            length = np.linalg.norm(acc, axis=1)
            fixable = bad & (length > 0)
            fixed = n3.copy()
            fixed[fixable] = acc[fixable] / length[fixable, None]
            attrs = {**attrs, "NORMAL": np.hstack([fixed, normals[:, 3:]]) if normals.shape[1] > 3 else fixed}
            done.append(f"rebuilt {int(fixable.sum())} invalid normal(s)")
    if done and any(d.startswith("removed") for d in done):
        attrs, tri = _compact(attrs, tri)
    return attrs, tri, done


# ── 5. AR copy ───────────────────────────────────────────────────────────────


def _surface(pos: np.ndarray, tri: np.ndarray, rng: np.random.Generator, n: int = SAMPLES) -> np.ndarray:
    a = _areas(pos, tri)
    if not len(tri) or a.sum() <= 0:
        return pos[:0]
    pick = rng.choice(len(tri), size=n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[pick]
    return pos[t[:, 0]] + u[:, None] * (pos[t[:, 1]] - pos[t[:, 0]]) + v[:, None] * (pos[t[:, 2]] - pos[t[:, 0]])


def surface_match(pos_a: np.ndarray, tri_a: np.ndarray, pos_b: np.ndarray, tri_b: np.ndarray, voxel: float) -> float:
    """How much of each surface lies on the other (worse direction), voxelised with one voxel of slack: 1 = same."""
    rng = np.random.default_rng(5)
    keys = []
    for pos, tri in ((pos_a, tri_a), (pos_b, tri_b)):
        cells = np.unique(np.floor(_surface(pos, tri, rng) / voxel).astype(np.int64), axis=0)  # small (≤ SAMPLES rows)
        keys.append(cells)
    offsets = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)])

    def code(c: np.ndarray) -> np.ndarray:
        c = c + (1 << 20)
        return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]

    def found(src: np.ndarray, dst: np.ndarray) -> float:
        if not len(src):
            return 1.0
        near = np.unique(code((dst[:, None, :] + offsets[None, :, :]).reshape(-1, 3)))
        return float(np.isin(code(src), near).mean())

    return min(found(keys[0], keys[1]), found(keys[1], keys[0]))


def decimate_primitive(attrs: dict[str, np.ndarray], tri: np.ndarray, model_diag: float) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]] | None:
    """The lightest decimation of this part that passes its checks, or None (keep the part as it is)."""
    import fast_simplification as fs

    pos = attrs["POSITION"][:, :3].astype(np.float64)
    area = float(_areas(pos, tri).sum())
    bounds = _bounds(pos, tri)
    for keep in KEEP_LEVELS:
        try:
            _, _, collapses = fs.simplify(pos, tri.astype(np.int32), target_reduction=1 - keep, preserve_border=True, return_collapses=True)
            new_pos, new_tri, mapping = fs.replay_simplification(pos.astype(np.float32), tri.astype(np.int32), collapses)
        except Exception as exc:  # noqa: BLE001 - the library's own errors: keep the part
            logger.info("Decimation failed at %.2f: %s", keep, exc)
            continue
        survivors = np.setdiff1d(np.arange(len(pos)), collapses[:, 1]) if len(collapses) else np.arange(len(pos))
        source = np.full(len(new_pos), -1)
        source[mapping[survivors]] = survivors
        if (source < 0).any() or not len(new_tri):
            continue
        # Each kept vertex is an original one: its own position, UV, normal and colour (no sliding texture).
        out_attrs = {n: a[source] for n, a in attrs.items()}
        out_tri = np.asarray(new_tri, dtype=np.int64).reshape(-1, 3)
        p = out_attrs["POSITION"][:, :3].astype(np.float64)
        live = _areas(p, out_tri) > AREA_EPS * model_diag * model_diag
        out_tri = out_tri[live]
        if len(out_tri) >= 0.9 * len(tri) or not len(out_tri):
            return None  # nothing worth saving at any level
        checks = {
            "bounds": float(np.abs(_bounds(p, out_tri) - bounds).max()) <= BOUNDS_TOLERANCE * model_diag,
            "area": abs(float(_areas(p, out_tri).sum()) / max(area, 1e-12) - 1) <= AREA_TOLERANCE,
        }
        match = surface_match(pos, tri, p, out_tri, SURFACE_VOXEL * model_diag) if all(checks.values()) else 0.0
        if all(checks.values()) and match >= SURFACE_MATCH:
            out_attrs, out_tri = _compact(out_attrs, out_tri)
            return out_attrs, out_tri, {"keep": keep, "triangles": int(len(out_tri)), "surfaceMatch": round(match, 3)}
    return None


# ── 4. Classes and sizes ─────────────────────────────────────────────────────

STRUCTURAL = {"panel", "top", "base", "frame", "leg", "door", "drawer", "shelf", "back", "side", "headboard", "footboard", "rail", "plinth", "handle", "seat", "backrest", "armrest"}
UPHOLSTERY = {"cushion"}
DECORATIVE = {"decor", "mirror", "glass"}
_SOFT = re.compile(r"cushion|upholster|padd|fabric|foam|tufted|quilted|velvet|linen|leather|وسادة|تنجيد|منجد|قماش|إسفنج|جلد", re.I)
_DECOR = re.compile(r"pillow|throw|blanket|duvet|quilt|bedding|sheet|mattress|cloth|book|vase|plant|lamp|rug|carpet|decor|accessor|"
                    r"مخدة|بطانية|لحاف|مفرش|مرتبة|ملابس|كتب|فازة|نبات|سجادة|إضاءة|ديكور", re.I)
_STRUCT = re.compile(r"frame|leg|panel|shelf|shelves|door|drawer|top|base|plinth|rail|support|carcass|body|side|back|headboard|footboard|"
                     r"apron|stretcher|post|slat|إطار|رجل|أرجل|لوح|رف|باب|درج|قاعدة|جانب|ظهر|هيكل|جسم", re.I)


def classify(component: dict[str, Any]) -> str:
    """structural / upholstery / decorative / other, from the component's type and names (English and Arabic)."""
    kind = str(component.get("type") or "")
    names = " ".join(str(component.get(k) or "") for k in ("name", "nameAr"))
    if kind in DECORATIVE or _DECOR.search(names):
        return "decorative"
    if kind in UPHOLSTERY or _SOFT.search(names):
        return "upholstery"
    if kind in STRUCTURAL or _STRUCT.search(names):
        return "structural"
    return "other"


# ── The whole stage ──────────────────────────────────────────────────────────


def _world_boxes(doc: dict[str, Any], prims: dict[tuple[int, int], dict[str, Any]]) -> dict[tuple[int, int], np.ndarray]:
    boxes: dict[tuple[int, int], np.ndarray] = {}
    for mesh, world, _ in _instances(doc):
        for (m, k), p in prims.items():
            if m != mesh or not len(p["tri"]):
                continue
            w = p["attrs"]["POSITION"][np.unique(p["tri"])][:, :3] @ world[:3, :3].T + world[:3, 3]
            box = np.stack([w.min(axis=0), w.max(axis=0)])
            boxes[(m, k)] = np.stack([np.minimum(boxes[(m, k)][0], box[0]), np.maximum(boxes[(m, k)][1], box[1])]) if (m, k) in boxes else box
    return boxes


def _isolate_materials(doc: dict[str, Any]) -> int:
    """A copy of each material for every mesh after the first that uses it (textures shared); returns copies made."""
    materials = doc.get("materials") or []
    seen: set[int] = set()
    copies = 0
    for mesh in doc.get("meshes") or []:
        local: dict[int, int] = {}
        for prim in mesh.get("primitives") or []:
            m = prim.get("material")
            if not isinstance(m, int):
                continue
            if m in local:
                prim["material"] = local[m]
                continue
            if m in seen:
                clone = json.loads(json.dumps(materials[m]))
                clone["name"] = f"{clone.get('name') or 'material'} ({mesh.get('name') or len(materials)})"[:80]
                materials.append(clone)
                local[m] = len(materials) - 1
                prim["material"] = local[m]
                copies += 1
            else:
                seen.add(m)
                local[m] = m
    return copies


def process(data: bytes, components: dict[str, Any], real_height_m: float | None = None, calibrated: bool = False) -> tuple[bytes, dict[str, Any], dict[str, Any] | None]:
    """(the checked and safely repaired GLB, its component document with classes, sizes and the quality report,
    the arrays needed for the AR copy or None when it is not worth making). The input is returned as it was when
    the file cannot be rewritten (its report says why)."""
    report: dict[str, Any] = {"version": VERSION, "repairs": [], "warnings": [], "unresolved": [], "parts": {},
                              "manufacturing": "Not assessed: the sizes are bounding boxes of a generated model, not cutting sizes; "
                                               "a carpenter confirms construction, joinery and materials."}
    doc_out = {**components, "components": [dict(c) for c in components.get("components") or []]}
    try:
        doc, binary = read_glb(data)
        views = _views(doc, binary)
        prims = _prims(doc, views)
    except (Unsupported, ValueError, KeyError, IndexError, TypeError) as exc:
        report["unresolved"].append(f"Geometry not checked: {exc}.")
        doc_out["quality"] = report
        return data, doc_out, None

    names = {(c.get("mesh"), c.get("primitive", 0)): c for c in doc_out["components"] if isinstance(c.get("mesh"), int)}
    label = lambda key: str((names.get(key) or {}).get("name") or f"mesh {key[0]}")  # noqa: E731
    before = sum(len(p["tri"]) for p in prims.values())
    for key, p in prims.items():
        stats = inspect_primitive(p["attrs"], p["tri"])
        attrs, tri, done = repair_primitive(p["attrs"], p["tri"])
        if done:
            report["repairs"] += [f"{label(key)}: {d}" for d in done]
        if any(d.startswith(("removed", "rebuilt")) for d in done):
            prims[key] = {"attrs": attrs, "tri": tri}
            stats["afterRepair"] = inspect_primitive(attrs, tri)
        part = names.get(key, {}).get("id") or f"{key[0]}_{key[1]}"
        report["parts"][part] = stats
        if stats["nonManifoldEdges"]:
            report["unresolved"].append(f"{label(key)}: {stats['nonManifoldEdges']} non-manifold edge(s), left as generated (fixing them would reshape the part).")
    open_parts = [label(k) for k, p in prims.items() if report["parts"].get(names.get(k, {}).get("id") or f"{k[0]}_{k[1]}", {}).get("openEdges")]
    if open_parts:
        report["warnings"].append(f"{len(open_parts)} part(s) have open edges (not closed solids): {', '.join(open_parts[:6])}. Not closed on purpose; "
                                  "the model looks right, but these surfaces are not solid volumes.")

    copies = _isolate_materials(doc)
    if copies:
        report["repairs"].append(f"gave {copies} part(s) their own material copy (textures shared), so recolouring one never changes another")
    repaired = _encode(doc, views, prims) if report["repairs"] else data
    after = sum(len(p["tri"]) for p in prims.values())

    # Classes, sizes and per-component triangle counts.
    boxes = _world_boxes(doc, prims)
    every = np.concatenate(list(boxes.values())) if boxes else np.zeros((2, 3))
    span = every.max(axis=0) - every.min(axis=0)
    to_cm = 100 * real_height_m / span[1] if real_height_m and span[1] > 0 else None
    mesh_of_material = {}
    for m, mesh in enumerate(doc.get("meshes") or []):
        for k, prim in enumerate(mesh.get("primitives") or []):
            mesh_of_material[(m, k)] = prim.get("material")
    for c in doc_out["components"]:
        c["class"] = classify(c)
        key = (c.get("mesh"), c.get("primitive", 0))
        if key in mesh_of_material:
            c["material"] = mesh_of_material[key]
        box = boxes.get(key)  # type: ignore[arg-type]
        if box is not None:
            size = box[1] - box[0]
            c["dimensions"] = {
                "basis": "bounding box",
                "relative": [round(float(v), 3) for v in size / np.maximum(span, 1e-9)],
                **({"cm": [round(float(v * to_cm), 1) for v in size], "calibrated": calibrated} if to_cm else {"cm": None, "calibrated": False}),
                "note": "The space it takes on the model, not a cutting size.",
            }
            c["triangles"] = int(len(prims[key]["tri"]))  # type: ignore[index]
    if not to_cm:
        report["warnings"].append("The model's real size is not known yet: component sizes are relative until the catalogue size is set.")
    elif not calibrated:
        report["warnings"].append("Component sizes in cm follow the photo's estimated height; they become exact once the catalogue size is set.")
    decorative = [c["name"] for c in doc_out["components"] if c["class"] == "decorative"]
    if decorative:
        report["warnings"].append(f"Decorative, not in the carpenter's parts list by default: {', '.join(decorative[:8])}.")

    report["triangles"] = {"original": int(before), "afterRepair": int(after), "ar": None}
    generic = [c["name"] for c in doc_out["components"] if re.match(r"^(mesh|object|node|part|model)[\s_\-]*\d*$", str(c.get("name")), re.I)]
    report["checks"] = [
        {"id": "names", "ok": not generic, "detail": f"Generic names: {', '.join(generic[:5])}" if generic else "Every component has a meaningful name."},
        {"id": "degenerate", "ok": all((s.get("afterRepair") or s)["degenerate"] == 0 for s in report["parts"].values()), "detail": "Zero-area triangles after repair."},
        {"id": "nonManifold", "ok": not any(s["nonManifoldEdges"] for s in report["parts"].values()), "detail": "Edges shared by more than two triangles."},
        {"id": "materials", "ok": True, "detail": "Each part has its own material; textures and UVs unchanged."},
        {"id": "selection", "ok": all(isinstance(c.get("mesh"), int) for c in doc_out["components"]), "detail": "Every component maps to its own mesh in the file."},
    ]
    doc_out["quality"] = report
    ar = {"doc": doc, "views": views, "prims": prims} if after > MIN_DECIMATE * 2 else None
    return repaired, doc_out, ar


def ar_copy(job: dict[str, Any]) -> tuple[bytes, dict[str, Any]] | None:
    """The AR copy and its report, or None when it is not lighter enough to be worth it (or a check failed)."""
    doc, views, prims = job["doc"], job["views"], job["prims"]
    boxes = _world_boxes(doc, prims)
    every = np.concatenate(list(boxes.values()))
    model_diag = float(np.linalg.norm(every.max(axis=0) - every.min(axis=0))) or 1.0
    out = dict(prims)
    levels: dict[str, Any] = {}
    for key, p in prims.items():
        if len(p["tri"]) < MIN_DECIMATE:
            continue
        result = decimate_primitive(p["attrs"], p["tri"], model_diag)
        if result:
            out[key] = {"attrs": result[0], "tri": result[1]}
            levels[f"{key[0]}_{key[1]}"] = result[2]
    before = sum(len(p["tri"]) for p in prims.values())
    after = sum(len(p["tri"]) for p in out.values())
    if after > AR_WORTH * before:
        return None
    glb = _encode(doc, views, out)
    # The whole copy against the full model: same bounds, same surface.
    full_pos = np.concatenate([b for b in boxes.values()])
    copy_boxes = _world_boxes(doc, out)
    copy_pos = np.concatenate(list(copy_boxes.values()))
    if np.abs(copy_pos.min(axis=0) - full_pos.min(axis=0)).max() > BOUNDS_TOLERANCE * model_diag or np.abs(copy_pos.max(axis=0) - full_pos.max(axis=0)).max() > BOUNDS_TOLERANCE * model_diag:
        logger.warning("AR copy discarded: its bounds differ from the model's")
        return None
    return glb, {"triangles": int(after), "original": int(before), "parts": levels}
