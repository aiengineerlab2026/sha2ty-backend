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
    # Every pass is computed afresh from the geometry as it is now: an earlier pass's messages are never carried
    # over (re-validating would repeat them). What an earlier pass found and this one no longer does is "resolved".
    previous = (doc_in.get("validation") or {}).get("issues") or []
    errors, warnings, repairs = [], [], []
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
                if bad.any():
                    # Only an invalid number is replaced: real-size UVs tile past 0..1 on purpose (a texture repeats
                    # across a large panel). A collapsed or stretched mapping is reported by geometry_between, never
                    # "repaired" by clamping, which would destroy it.
                    rows = bad.any(axis=1)
                    if procedural and "NORMAL" in attrs:
                        from app.services import solids

                        uv[rows] = solids.face_uv(pos[rows], _read(doc, views, attrs["NORMAL"]).astype(np.float64)[rows])
                    else:
                        uv[bad] = 0
                    _write(doc, views, attrs["TEXCOORD_0"], uv.astype(np.float32))
                    repairs.append(f"{name}: had invalid texture coordinates; recomputed from its faces.")
            if plain and comp.get("box"):
                # A plain board is its recorded box: a vertex that drifted off it goes back onto the nearest face of
                # that box (the design as built), which also closes the solid again.
                faces = np.array(comp["box"], float)
                off = np.min(np.abs(pos[:, None, :] - faces[None]), axis=1).max()
                if off > 1e-4:
                    from app.services import solids

                    rebuilt = solids.mesh_arrays(solids.box(faces[0], faces[1]))[0]
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
        board_like = plain and cls not in ("glass", "mirror", "metal", "generated") and comp.get("kind") not in ("back_panel", "mirror", "glass") \
            and not any(w in name.lower() for w in ("handle", "knob", "pull", "leg"))
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
    if only is None:  # the whole model: real geometry between parts and the UVs, repaired where it is safe
        between = geometry_between(out, doc_in)
        for _ in range(REPAIR_PASSES):
            fixed, made = repair_boards(out, doc_in, between["metrics"])
            if not made:
                break
            out, repairs = fixed, repairs + made
            between = geometry_between(out, doc_in)
        errors += between["errors"]
        warnings += between["warnings"]
        doc_in.setdefault("hybrid", {})["geometryMetrics"] = between["metrics"]
    issues = _issues(errors, warnings, between["metrics"] if only is None else {}, "geometry")
    errors, warnings = list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))
    now = {i["id"] for i in issues}
    doc_in["validation"] = {**{k: v for k, v in (doc_in.get("validation") or {}).items() if k not in ("errors", "warnings", "issues", "resolved")},
                            "ok": not errors, "errors": errors, "warnings": warnings, "repairs": repairs, "issues": issues,
                            "resolved": [i for i in previous if i.get("id") not in now],
                            "checks": ["normals", "uv", "materials", "closed", "flat", "thickness", "duplicates", "overlap", "size",
                                       "volume", "floating", "handle ownership", "pivots"]}
    doc_in["validation"] = verdict(doc_in)
    if errors:
        logger.info("Model validation: %d unresolved issue(s) after repair: %s", len(errors), errors[:5])
    if repairs:
        logger.info("Model repairs: %s", repairs)
    return out, doc_in


# ---- geometry between parts, UVs, and the verdict ----

INTERSECTION_CM3 = 1.0  # two solids sharing more than this many cm³ pass through each other
COPLANAR_CM2 = 1.0  # two same-facing faces of different parts overlapping more than this in one plane: z-fighting
FRAME_FILL = 0.85  # a frame whose volume is this share of its box is a filled board, not a frame
UV_STRETCH = 4.0  # texel density varying more than this across one part: stretched
REPAIR_PASSES = 2  # bounded: each pass trims the boards found overlapping, then everything is checked again
# Which of two overlapping boards gives way: the one further down this list (an inside board yields to the carcass).
YIELD = ["side_panel", "top_panel", "base_panel", "plinth", "crown_trim", "back_panel", "divider", "shelf"]


def _issues(errors: list[str], warnings: list[str], metrics: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    """Each finding once, with a stable id (type, parts, stage): repeated validation finds the same ids, so nothing
    is listed twice and a repaired issue simply disappears."""
    out: dict[str, dict[str, Any]] = {}
    for kind, key in (("intersection", "intersections"), ("coplanar", "zFighting")):
        for m in metrics.get(key) or []:
            a, b = sorted((m["a"], m["b"]))
            iid = f"{kind}:{a}:{b}:{stage}"
            out[iid] = {"id": iid, "type": kind, "components": [a, b], "stage": stage, "measure": m.get("cm3", m.get("cm2"))}
    for severity, lines in (("error", errors), ("warning", warnings)):
        for line in lines:
            if " pass through each other" in line or " in the same place" in line:
                continue  # recorded above from the metrics, by part ids
            iid = f"{severity}:{hashlib.sha1(line.encode()).hexdigest()[:10]}:{stage}"
            out.setdefault(iid, {"id": iid, "type": severity, "stage": stage, "text": line})
    return list(out.values())


def repair_boards(glb: bytes, doc_in: dict[str, Any], metrics: dict[str, Any]) -> tuple[bytes, list[str]]:
    """Plain procedural boards found passing through each other, or with same-facing faces in one plane, set right:
    the board that yields (YIELD: a divider to the base, a shelf to a divider) is trimmed to stop at the other's
    face along the axis it overlaps least (a joint, faces touching); a coplanar same-facing face is moved 0.5 mm
    inward. Only boards still exactly their recorded box are touched, and only when the board keeps most of its
    size; the part's recorded box follows. (new GLB, what was done)."""
    from app.services import solids

    comps = {c["id"]: c for c in doc_in.get("components") or []}
    decorated = {d["component"] for d in (doc_in.get("hybrid") or {}).get("decorations") or [] if d.get("method") == "geometry"}

    def board(pid):
        c = comps.get(pid) or {}
        # carcass and inside boards only: a door or a drawer moves on its own pivot and is never trimmed
        return c if c.get("generation") == "procedural" and c.get("box") and pid not in decorated and c.get("kind") in YIELD else None

    def rank(c):
        return YIELD.index(c["kind"]) if c.get("kind") in YIELD else len(YIELD)

    boxes = {pid: [np.array(c["box"][0], float), np.array(c["box"][1], float)] for pid, c in comps.items() if board(pid)}
    changed: dict[str, list[str]] = {}
    crossing = {frozenset((m["a"], m["b"])) for m in metrics.get("intersections") or []}
    for kind, key in (("intersection", "intersections"), ("coplanar", "zFighting")):
        for m in metrics.get(key) or []:
            a, b = comps.get(m["a"]), comps.get(m["b"])
            if not a or not b or not board(a["id"]) or not board(b["id"]) or rank(a) == rank(b):
                continue
            if kind == "coplanar" and frozenset((m["a"], m["b"])) in crossing:
                continue  # trimming the overlap already parts their faces
            give, keep = (a, b) if rank(a) > rank(b) else (b, a)
            glo, ghi = boxes[give["id"]]
            klo, khi = boxes[keep["id"]]
            if kind == "intersection":
                depth = np.minimum(ghi, khi) - np.maximum(glo, klo)
                if np.any(depth <= 0):
                    continue
                # every way to stop it at the other's face (each axis, from below or above): the one that keeps the
                # most of the yielding board is the joint (a divider stands on the base, not beside it)
                options = []
                for ax in range(3):
                    lo_cut = (np.r_[glo[:ax], khi[ax], glo[ax + 1:]], ghi)  # it starts where the other ends
                    hi_cut = (glo, np.r_[ghi[:ax], klo[ax], ghi[ax + 1:]])  # it ends where the other starts
                    options += [o for o in (lo_cut, hi_cut) if np.all(o[1] - o[0] > 0)]
                if not options:
                    continue
                new = max(options, key=lambda o: float(np.prod(o[1] - o[0])))
                what = f"stops at {keep.get('name') or keep['id']} instead of passing through it"
            else:
                new = None
                for ax in range(3):
                    for side in (0, 1):
                        if abs(boxes[give["id"]][side][ax] - boxes[keep["id"]][side][ax]) < 1e-4:
                            lo, hi = glo.copy(), ghi.copy()
                            (lo if side == 0 else hi)[ax] += 0.0005 if side == 0 else -0.0005
                            new = (lo, hi)
                            break
                    if new:
                        break
                if new is None:
                    continue
                what = f"its face no longer lies in the same place as {keep.get('name') or keep['id']}'s"
            size_before, size_after = ghi - glo, new[1] - new[0]
            if np.any(size_after <= 0.5 * size_before):
                continue  # it would lose most of a dimension: not a safe repair
            boxes[give["id"]] = [new[0], new[1]]
            changed.setdefault(give["id"], []).append(f"{give.get('name') or give['id']}: {what}.")
    if not changed:
        return glb, []
    doc, binary = read_glb(glb)
    views = _views(doc, binary)
    done = []
    for i, node in _parts(doc):
        pid = (node.get("extras") or {}).get("partId")
        if pid not in changed:
            continue
        prims = doc["meshes"][node["mesh"]].get("primitives") or []
        if len(prims) != 1:
            continue
        attrs = prims[0]["attributes"]
        pos, nrm, uv, _ = solids.mesh_arrays(solids.box(*boxes[pid]))
        if len(pos) != len(_read(doc, views, attrs["POSITION"])):
            continue  # not the builder's own board layout: left as it is
        _write(doc, views, attrs["POSITION"], pos.astype(np.float32))
        if "NORMAL" in attrs:
            _write(doc, views, attrs["NORMAL"], nrm.astype(np.float32))
        if "TEXCOORD_0" in attrs:
            _write(doc, views, attrs["TEXCOORD_0"], uv.astype(np.float32))
        comps[pid]["box"] = [[round(float(v), 4) for v in boxes[pid][0]], [round(float(v), 4) for v in boxes[pid][1]]]
        done += changed[pid]
    return (write_glb(doc, views), done) if done else (glb, [])


def _triangles(doc: dict, views: list[bytes], node: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pos_all, idx_all, uv_all, off = [], [], [], 0
    for prim in doc["meshes"][node["mesh"]].get("primitives") or []:
        attrs = prim.get("attributes") or {}
        pos = _read(doc, views, attrs["POSITION"]).astype(np.float64)
        idx = _read(doc, views, prim["indices"]).ravel().astype(np.int64) if "indices" in prim else np.arange(len(pos))
        pos_all.append(pos)
        idx_all.append(idx + off)
        uv_all.append(_read(doc, views, attrs["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in attrs else None)
        off += len(pos)
    uv = np.concatenate(uv_all) if uv_all and all(u is not None for u in uv_all) else None
    return np.concatenate(pos_all), np.concatenate(idx_all), uv


def _manifold(pos: np.ndarray, idx: np.ndarray):
    import manifold3d as mf

    keys = np.round(pos, 6)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    tri = inv.ravel()[idx.reshape(-1, 3)]
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
    m = mf.Manifold(mf.Mesh(vert_properties=uniq.astype(np.float32), tri_verts=tri.astype(np.uint32)))
    return m if m.status() == mf.Error.NoError else None


def _planes(pos: np.ndarray, idx: np.ndarray) -> dict[tuple, list[np.ndarray]]:
    """Triangles grouped by the plane they lie in (axis-aligned planes only: boards, frames, panes)."""
    tri = pos[idx.reshape(-1, 3)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1)
    n = n / np.maximum(area[:, None], 1e-12)
    out: dict[tuple, list[np.ndarray]] = {}
    for t, nn, a in zip(tri, n, area):
        axis = int(np.argmax(np.abs(nn)))
        if a < 1e-10 or abs(nn[axis]) < 0.999:
            continue
        key = (axis, int(np.sign(nn[axis])), round(float(t[0][axis]), 4))
        out.setdefault(key, []).append(np.delete(t, axis, axis=1))
    return out


def _coplanar_overlap(a: dict, b: dict) -> float:
    """m² of same-facing faces of two parts lying in the same plane over each other (what z-fights)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    total = 0.0
    for key in set(a) & set(b):
        pa = unary_union([Polygon(t) for t in a[key] if Polygon(t).area > 0])
        pb = unary_union([Polygon(t) for t in b[key] if Polygon(t).area > 0])
        total += pa.intersection(pb).area
    return total


def geometry_between(glb: bytes, doc_in: dict[str, Any]) -> dict[str, Any]:
    """Real-geometry checks between parts and on each part's UVs: {errors, warnings, metrics}."""
    doc, binary = read_glb(glb)
    views = _views(doc, binary)
    linked = _linked(doc)
    comps = {c["id"]: c for c in doc_in.get("components") or []}
    parts = []
    for i, node in _parts(doc):
        if i not in linked:
            continue
        pid = node["extras"]["partId"]
        pos, idx, uv = _triangles(doc, views, node)
        parts.append({"id": pid, "name": node.get("name") or pid, "pos": pos, "idx": idx, "uv": uv, "lo": pos.min(0), "hi": pos.max(0),
                      "generated": (comps.get(pid) or {}).get("generation") == "meshy", "kind": (comps.get(pid) or {}).get("kind")})
    errors, warnings = [], []
    metrics: dict[str, Any] = {"intersections": [], "zFighting": [], "uv": {}}
    solids = {p["id"]: _manifold(p["pos"], p["idx"]) for p in parts}
    planes = {p["id"]: _planes(p["pos"], p["idx"]) for p in parts}
    for n, a in enumerate(parts):
        # a frame is a frame: its opening is really there
        # (a frameless mirror door's backing is a board on purpose: its mirror lies on its face, not in an opening)
        if a["kind"] == "door_frame" and solids[a["id"]] is not None and not comps.get(a["id"], {}).get("frameless"):
            fill = solids[a["id"]].volume() / max(float(np.prod(a["hi"] - a["lo"])), 1e-12)
            if fill > FRAME_FILL and any((c.get("assembly") == comps.get(a["id"], {}).get("assembly")) and c.get("kind") in ("mirror", "glass") for c in comps.values()):
                errors.append(f"{a['name']}: is a filled board, not a frame around its pane.")
        for b in parts[n + 1:]:
            if np.any(a["hi"] < b["lo"] - 1e-4) or np.any(b["hi"] < a["lo"] - 1e-4):
                continue
            sa, sb = solids[a["id"]], solids[b["id"]]
            if sa is not None and sb is not None:
                vol = (sa ^ sb).volume() * 1e6
                if vol > INTERSECTION_CM3:
                    metrics["intersections"].append({"a": a["id"], "b": b["id"], "cm3": round(vol, 1)})
                    (warnings if a["generated"] or b["generated"] else errors).append(f"{a['name']} and {b['name']} pass through each other ({vol:.0f} cm³).")
            area = _coplanar_overlap(planes[a["id"]], planes[b["id"]]) * 1e4
            if area > COPLANAR_CM2:
                metrics["zFighting"].append({"a": a["id"], "b": b["id"], "cm2": round(area, 1)})
                errors.append(f"{a['name']} and {b['name']} have {area:.0f} cm² of faces in the same place (they flicker).")
    for p in parts:  # UVs: present, finite, not degenerate, not stretched
        if p["uv"] is None or p["generated"]:
            continue
        tri = p["pos"][p["idx"].reshape(-1, 3)]
        uvt = p["uv"][p["idx"].reshape(-1, 3)]
        area3 = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        areauv = 0.5 * np.abs((uvt[:, 1, 0] - uvt[:, 0, 0]) * (uvt[:, 2, 1] - uvt[:, 0, 1]) - (uvt[:, 2, 0] - uvt[:, 0, 0]) * (uvt[:, 1, 1] - uvt[:, 0, 1]))
        real = area3 > 1e-8
        degenerate = int(np.sum(real & (areauv < 1e-12)))
        density = areauv[real] / area3[real]
        stretch = float(density.max() / max(density.min(), 1e-12)) if density.size and density.min() > 0 else float("inf") if density.size else 1.0
        metrics["uv"][p["id"]] = {"degenerate": degenerate, "stretch": round(min(stretch, 1e6), 2)}
        if not np.all(np.isfinite(p["uv"])):
            errors.append(f"{p['name']}: has invalid texture coordinates.")
        elif degenerate:
            errors.append(f"{p['name']}: {degenerate} faces have no texture area (the texture collapses to a line there).")
        elif stretch > UV_STRETCH:
            warnings.append(f"{p['name']}: its texture is stretched ({stretch:.1f}× between faces).")
    return {"errors": errors, "warnings": warnings, "metrics": metrics}


def verdict(doc_in: dict[str, Any]) -> dict[str, Any]:
    """The model's state, kept as separate properties (technical quality never decides publishing):
      technical    "passed" | "issues": what the geometry checks found after repair (errors stay recorded as found)
      status       the QUALITY: valid | valid_with_warnings (estimates, notes) | needs_repair (geometry issues left
                   after automatic repair) | failed (no geometry at all)
      repair       what was repaired automatically and how many issues are left
      usable       a model that loads and shows its parts: it has geometry
      publishable  = usable. Intersections, coplanar faces, UVs, colour or likeness differences, estimates and
                   uncertain movements never stop a usable model from being saved or published; only a missing or
                   broken model (no parts) does. The vendor's own listing choice is never changed here.
    `notes`: the estimates a person may want to adjust (never a technical message, never an obligation)."""
    v = doc_in.get("validation") or {}
    notes: list[dict[str, str]] = []
    for label in (doc_in.get("animation") or {}).get("uncertain") or []:
        notes.append({"kind": "movement", "text": f"How {label} opens was estimated from the photo."})
    for f in (doc_in.get("hybrid") or {}).get("materialFallbacks") or []:
        notes.append({"kind": "material", "text": f"{f.get('name') or f['component']}: material estimated ({f['reason']})."})
    for p in doc_in.get("componentFailures") or []:
        notes.append({"kind": "component", "text": f"{p['name']} could not be generated; it can be regenerated."})
    for w in v.get("warnings") or []:
        notes.append({"kind": "estimate" if w.startswith("Estimated, not seen") else "check", "text": w})
    if doc_in.get("sizeSource") not in (None, "vendor"):
        notes.append({"kind": "dimension", "text": "Overall sizes were estimated from the photo."})
    errors = v.get("errors") or []
    usable = bool(doc_in.get("components"))
    status = "failed" if not usable else "needs_repair" if errors else "valid_with_warnings" if notes else "valid"
    out = {**v, "status": status, "technical": "issues" if errors else "passed", "notes": notes, "usable": usable, "publishable": usable,
           "repair": {"repaired": len(v.get("repairs") or []), "unresolved": len(errors), "automatic": True}}
    out.pop("confirm", None)  # the old mandatory confirmations: gone
    return out
