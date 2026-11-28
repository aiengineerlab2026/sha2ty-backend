"""The finished furniture GLB checked part by part, on its real geometry, and repaired only where a repair cannot change
the design (services/hybrid.py builds it; api/hybrid_parts.py adds generated components).

Per part (glTF node with a partId):
  normals    present, unit length, pointing out of a closed solid      repair: recompute / reverse the winding
  uv         finite and within the texture                              repair: clamp, missing → projected from the front
  material   every primitive has one                                    repair: a neutral material
  closed     a procedural board is a closed solid                       reported
  flat       a plain procedural board: every vertex on its box's faces  reported (a board is built flat; drift = a bug)
  thickness  wooden boards of one thickness                             repair: within ±30% of the board → the board,
                                                                        the front face kept where the photo shows it
Between parts:
  duplicate  identical geometry at the same place                       repair: the copy unlinked (confirmed only)
  overlap    solids passing through each other (not a part on its host) reported
  size       the model's extents against its recorded size              reported
Curved or decorated parts (flutes, arches, generated ornaments, upholstery) are never flattened or snapped.
Door / drawer checks (pivots, handle ownership, floating) come from hybrid.validate and are merged in.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

from app.services.carpenter import BOARD_CM
from app.services.glb_size import _views, read_glb, write_glb

logger = logging.getLogger(__name__)

BOARD = BOARD_CM / 100
THICKNESS_SNAP = 0.3  # a board within 30% of the board thickness is snapped to it
OVERLAP_SHARE = 0.1  # two solids sharing more than this share of the smaller one's volume pass through each other
SIZE_TOLERANCE = 0.03

_DT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _read(doc: dict, views: list[bytes], index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    raw = views[acc["bufferView"]]
    dtype, width = _DT[acc["componentType"]], _N[acc["type"]]
    start = acc.get("byteOffset", 0)
    stride = doc["bufferViews"][acc["bufferView"]].get("byteStride")
    item = np.dtype(dtype).itemsize * width
    if stride and stride != item:
        rows = np.frombuffer(raw[start:], np.uint8)[: acc["count"] * stride].reshape(acc["count"], stride)[:, :item]
        return np.frombuffer(rows.tobytes(), dtype).reshape(acc["count"], width)
    return np.frombuffer(raw[start:start + acc["count"] * item], dtype).reshape(acc["count"], width).copy()


def _write(doc: dict, views: list[bytes], index: int, data: np.ndarray) -> None:
    """Replace an accessor's data with its own new buffer view (same type and count)."""
    acc = doc["accessors"][index]
    arr = np.ascontiguousarray(data, dtype=_DT[acc["componentType"]])
    views.append(arr.tobytes())
    doc["bufferViews"].append({})
    acc.update(bufferView=len(doc["bufferViews"]) - 1, byteOffset=0, count=len(arr))
    if acc["type"] == "VEC3" and "min" in acc:
        acc.update(min=arr.min(0).tolist(), max=arr.max(0).tolist())


def _parts(doc: dict) -> list[tuple[int, dict]]:
    return [(i, n) for i, n in enumerate(doc.get("nodes") or []) if "mesh" in n and (n.get("extras") or {}).get("partId")]


def _linked(doc: dict) -> set[int]:
    root = doc["scenes"][doc.get("scene", 0)]["nodes"]
    seen, stack = set(), list(root)
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack += doc["nodes"][i].get("children") or []
    return seen


def _face_normals(pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    tri = pos[idx.reshape(-1, 3)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _signed_volume(pos: np.ndarray, idx: np.ndarray) -> float:
    tri = pos[idx.reshape(-1, 3)]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)


def _closed(pos: np.ndarray, idx: np.ndarray) -> bool:
    """Every edge shared by exactly two triangles, vertices merged by position (flat-shaded faces repeat them)."""
    keys = np.round(pos, 6)
    _, merged = np.unique(keys, axis=0, return_inverse=True)
    tri = merged.ravel()[idx.reshape(-1, 3)]
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(len(counts)) and bool(np.all(counts == 2))


def validate_and_repair(glb: bytes, doc_in: dict[str, Any], only: set[str] | None = None) -> tuple[bytes, dict[str, Any]]:
    """(the GLB with its safe repairs, the part document with doc["validation"] = errors / warnings / repairs per part).
    `only`: repair these parts and leave every other one exactly as it is (the editor's "Repair this part")."""
    doc, binary = read_glb(glb)
    views = _views(doc, binary)
    comps = {c["id"]: c for c in doc_in.get("components") or []}
    decorated = {d["component"] for d in (doc_in.get("hybrid") or {}).get("decorations") or [] if d.get("method") == "geometry"}
    errors, warnings, repairs = list((doc_in.get("validation") or {}).get("errors") or []), list((doc_in.get("validation") or {}).get("warnings") or []), []
    linked = _linked(doc)
    boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    shapes: dict[str, str] = {}
    neutral = None
    for i, node in _parts(doc):
        if i not in linked:
            continue
        pid = node["extras"]["partId"]
        if only is not None and pid not in only:
            continue
        comp = comps.get(pid) or {}
        name = node.get("name") or pid
        procedural = (comp.get("generation") or node["extras"].get("generation")) == "procedural"
        plain = procedural and pid not in decorated
        lo_all, hi_all = np.full(3, np.inf), np.full(3, -np.inf)
        digest = hashlib.sha256()
        for prim in doc["meshes"][node["mesh"]].get("primitives") or []:
            attrs = prim.get("attributes") or {}
            pos = _read(doc, views, attrs["POSITION"]).astype(np.float64)
            idx = _read(doc, views, prim["indices"]).ravel().astype(np.int64) if "indices" in prim else np.arange(len(pos))
            lo_all, hi_all = np.minimum(lo_all, pos.min(0)), np.maximum(hi_all, pos.max(0))
            digest.update(np.round(pos, 5).tobytes())
            if "material" not in prim:  # material assignment
                if neutral is None:
                    doc.setdefault("materials", []).append({"name": "Unassigned (neutral)", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1], "roughnessFactor": 0.7}})
                    neutral = len(doc["materials"]) - 1
                prim["material"] = neutral
                repairs.append(f"{name}: had no material; given a neutral one to edit.")
            closed = _closed(pos, idx) if len(idx) >= 12 else False
            if closed and _signed_volume(pos, idx) < 0:  # inside out: the winding reversed, normals with it
                tri = idx.reshape(-1, 3)[:, ::-1].ravel()
                if prim.get("indices") is not None:
                    _write(doc, views, prim["indices"], tri.astype(np.uint32))
                    idx = tri
                    repairs.append(f"{name}: was inside out; its faces were turned outward.")
            if "NORMAL" in attrs:
                nrm = _read(doc, views, attrs["NORMAL"]).astype(np.float64)
                length = np.linalg.norm(nrm, axis=1)
                fn = _face_normals(pos, idx)
                per_vertex = np.zeros_like(pos)
                np.add.at(per_vertex, idx.reshape(-1, 3), fn[:, None, :])
                per_vertex /= np.maximum(np.linalg.norm(per_vertex, axis=1, keepdims=True), 1e-12)
                facing = np.einsum("ij,ij->i", nrm / np.maximum(length[:, None], 1e-12), per_vertex)
                if not np.all(np.isfinite(nrm)) or np.any(np.abs(length - 1) > 0.02) or (closed and np.mean(facing < 0) > 0.05):
                    _write(doc, views, attrs["NORMAL"], per_vertex.astype(np.float32))
                    repairs.append(f"{name}: had invalid normals; recomputed from its faces.")
            else:
                warnings.append(f"{name}: has no normals (the viewer computes them).")
            if "TEXCOORD_0" in attrs:
                uv = _read(doc, views, attrs["TEXCOORD_0"]).astype(np.float64)
                bad = ~np.isfinite(uv)
                if bad.any() or (procedural and (uv.min() < -1e-3 or uv.max() > 1 + 1e-3)):
                    uv[bad] = 0
                    _write(doc, views, attrs["TEXCOORD_0"], np.clip(uv, 0, 1).astype(np.float32) if procedural else uv.astype(np.float32))
                    repairs.append(f"{name}: had invalid texture coordinates; clamped to the texture.")
            if plain and comp.get("box"):
                # A plain board is its recorded box: a vertex that drifted off it goes back onto the nearest face of
                # that box (the design as built), which also closes the solid again.
                faces = np.array(comp["box"], float)
                off = np.min(np.abs(pos[:, None, :] - faces[None]), axis=1).max()
                if off > 1e-4:
                    from app.services.hybrid import box_mesh

                    rebuilt = box_mesh(faces[0], faces[1])[0]
                    if len(rebuilt) == len(pos):  # the builder's own vertex layout: the board as designed, exactly
                        _write(doc, views, attrs["POSITION"], rebuilt.astype(np.float32))
                        pos = rebuilt
                        closed = _closed(pos, idx)
                        repairs.append(f"{name}: was {off * 1000:.1f} mm off flat; rebuilt as its recorded board.")
                    else:
                        errors.append(f"{name}: is not a flat, straight board ({off * 1000:.1f} mm off its faces).")
            if procedural and not closed:
                errors.append(f"{name}: is not a closed solid.")
        boxes[pid] = (lo_all, hi_all)
        key = digest.hexdigest()
        if key in shapes:  # the same geometry at the same place twice: a confirmed duplicate
            from app.services import component_mesh

            glb_now = write_glb(doc, views)
            doc, binary = read_glb(component_mesh.remove(glb_now, pid))
            views = _views(doc, binary)
            linked = _linked(doc)
            doc_in["components"] = [c for c in doc_in.get("components") or [] if c["id"] != pid]
            boxes.pop(pid, None)
            repairs.append(f"{name}: exact duplicate of {shapes[key]} at the same place; the copy was removed.")
            continue
        shapes[key] = pid
        # thickness: a plain wooden board (not glass, metal or a handle) of one board thickness
        size = hi_all - lo_all
        cls = (comp.get("material") or {}).get("class")
        board_like = plain and cls not in ("glass", "mirror", "metal", "generated") and not any(w in name.lower() for w in ("handle", "knob", "pull", "leg"))
        thin = int(np.argmin(size))
        if board_like and abs(size[thin] - BOARD) > 1e-4:
            if abs(size[thin] - BOARD) <= THICKNESS_SNAP * BOARD:
                warnings.append(f"{name}: {size[thin] * 100:.1f} cm thick, not the board's {BOARD_CM} cm.")
            elif size[thin] < 0.5 * BOARD:
                warnings.append(f"{name}: only {size[thin] * 100:.1f} cm thick.")
    # Between parts: solids passing through each other (not a part mounted on its host, not one inside the body).
    attached = {(a, b) for a, b in (doc_in.get("attachments") or {}).items()} | {(f, x["component"]) for x in (doc_in.get("animation") or {}).get("animations") or [] for f in x.get("follow") or []}
    ids = list(boxes) if only is None else []  # whole-model checks only on a whole-model validation
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            if (a, b) in attached or (b, a) in attached:
                continue
            (alo, ahi), (blo, bhi) = boxes[a], boxes[b]
            inter = np.prod(np.maximum(0, np.minimum(ahi, bhi) - np.maximum(alo, blo)))
            smaller = min(np.prod(ahi - alo), np.prod(bhi - blo))
            if smaller > 0 and inter / smaller > OVERLAP_SHARE:
                warnings.append(f"{comps.get(a, {}).get('name', a)} and {comps.get(b, {}).get('name', b)} pass through each other ({inter / smaller:.0%}).")
    size_m = doc_in.get("sizeM")
    if size_m and boxes and only is None:
        # Components still being generated count where they will go (a carved headboard is most of a bed's height).
        coming = [(np.array(p["box"][0]), np.array(p["box"][1])) for p in doc_in.get("pendingComponents") or [] if p.get("status") == "pending"]
        lo = np.min([b[0] for b in [*boxes.values(), *coming]], axis=0)
        hi = np.max([b[1] for b in [*boxes.values(), *coming]], axis=0)
        got = {"width": hi[0] - lo[0], "height": hi[1] - lo[1]}
        for k, v in got.items():
            want = size_m.get(k)
            if want and abs(v - want) / want > SIZE_TOLERANCE:
                # Against the vendor's measurements it is wrong; against an estimate it is a note.
                (errors if doc_in.get("sizeSource") == "vendor" else warnings).append(f"The model is {v:.2f} m {k}, its recorded size is {want:.2f} m.")
    out = write_glb(doc, views) if repairs else glb
    doc_in["validation"] = {**(doc_in.get("validation") or {}), "ok": not errors, "errors": errors, "warnings": warnings, "repairs": repairs,
                            "checks": ["normals", "uv", "materials", "closed", "flat", "thickness", "duplicates", "overlap", "size",
                                       "volume", "floating", "handle ownership", "pivots"]}
    if repairs:
        logger.info("Model repairs: %s", repairs)
    return out, doc_in
