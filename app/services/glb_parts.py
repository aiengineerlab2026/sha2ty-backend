"""What a generated GLB can actually be edited as.

Meshy is asked for separate parts, but image-to-3D often returns one mesh. This reads the glTF
scene (nodes, primitives, materials) and lists only the pieces that exist as their own mesh.
A single mesh stays one editable object — cushions and legs are not invented here.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def inspect_glb(data: bytes, model_type: str = "furniture") -> dict[str, Any]:
    try:
        doc = _gltf_json(data)
    except (ValueError, json.JSONDecodeError):
        return {"modelType": model_type, "separated": False, "components": [], "limits": "This file could not be read as a GLB."}
    nodes = doc.get("nodes") or []
    meshes = doc.get("meshes") or []
    materials = doc.get("materials") or []
    components: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    def add(mesh_index: int, name: str, node_index: int | None = None, matrix=None) -> None:
        mesh = meshes[mesh_index] if mesh_index < len(meshes) else {}
        prims = mesh.get("primitives") or [{}]
        label = (name or mesh.get("name") or "").strip()
        for index, prim in enumerate(prims):
            mat = prim.get("material")
            mat_name = ""
            if isinstance(mat, int) and mat < len(materials):
                mat_name = str(materials[mat].get("name") or "")
            title = label or mat_name or f"Part {len(components) + 1}"
            if len(prims) > 1:
                title = f"{title} {index + 1}"
            base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40] or "part"
            seen[base] = seen.get(base, 0) + 1
            part_id = base if seen[base] == 1 else f"{base}_{seen[base]}"
            components.append(
                {
                    "id": part_id,
                    "name": _pretty(title),
                    "mesh": mesh_index,
                    "primitive": index,
                    "material": mat if isinstance(mat, int) else None,
                    "node": node_index,
                    "box": _box(doc, prim, matrix),
                    "editable": True,
                }
            )

    def walk(index: int, parent=None) -> None:
        if not isinstance(index, int) or index < 0 or index >= len(nodes) or index in walking:
            return
        walking.add(index)
        node = nodes[index]
        matrix = _matrix(node) if parent is None else parent @ _matrix(node)
        if isinstance(node.get("mesh"), int):
            add(node["mesh"], str(node.get("name") or ""), index, matrix)
        for child in node.get("children") or []:
            walk(child, matrix)

    walking: set[int] = set()
    scene_index = doc.get("scene") or 0
    scenes = doc.get("scenes") or []
    roots = scenes[scene_index].get("nodes") if scene_index < len(scenes) else None
    if not roots:
        roots = [i for i, node in enumerate(nodes) if isinstance(node.get("mesh"), int)]
    for root in roots or []:
        walk(root)
    if len(components) == 1 and components[0]["name"].startswith("Part"):
        components[0]["name"] = "Whole model"
        components[0]["id"] = "whole"
    separated = len(components) > 1
    limits = None if separated else "One mesh. Color and texture change the whole model. Separate cushions, legs or doors were not in this file."
    return {"modelType": model_type, "separated": separated, "components": components, "limits": limits}


def _matrix(node: dict[str, Any]):
    """The node's local transform (glTF: a column-major matrix, or translation · rotation · scale)."""
    import numpy as np

    if node.get("matrix"):
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    x, y, z, w = node.get("rotation") or (0, 0, 0, 1)
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    out = np.eye(4)
    out[:3, :3] = rot * np.array(node.get("scale") or (1, 1, 1), dtype=float)
    out[:3, 3] = node.get("translation") or (0, 0, 0)
    return out


def _box(doc: dict[str, Any], prim: dict[str, Any], matrix) -> list[list[float]] | None:
    """World-space bounds of a primitive from its POSITION accessor's min/max (required by glTF)."""
    import numpy as np

    accessors = doc.get("accessors") or []
    index = (prim.get("attributes") or {}).get("POSITION")
    if not isinstance(index, int) or index >= len(accessors) or matrix is None:
        return None
    lo, hi = accessors[index].get("min"), accessors[index].get("max")
    if not lo or not hi:
        return None
    corners = np.array([[a, b, c, 1.0] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    world = (matrix @ corners.T).T[:, :3]
    return [[round(float(v), 4) for v in world.min(0)], [round(float(v), 4) for v in world.max(0)]]


def write_names(data: bytes, components: list[dict[str, Any]]) -> bytes:
    """The same GLB with each component's display name on its node (and on its mesh when it is one primitive)."""
    doc = _gltf_json(data)
    chunk_len = int.from_bytes(data[12:16], "little")
    nodes, meshes = doc.get("nodes") or [], doc.get("meshes") or []
    for c in components:
        if isinstance(c.get("node"), int) and c["node"] < len(nodes):
            nodes[c["node"]]["name"] = c["name"]
        mesh = meshes[c["mesh"]] if isinstance(c.get("mesh"), int) and c["mesh"] < len(meshes) else None
        if mesh is not None and len(mesh.get("primitives") or []) == 1:
            mesh["name"] = c["name"]
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)  # chunks are 4-byte aligned; JSON pads with spaces
    rest = data[20 + chunk_len :]
    total = 12 + 8 + len(text) + len(rest)
    return data[:8] + total.to_bytes(4, "little") + len(text).to_bytes(4, "little") + b"JSON" + text + rest


def _gltf_json(data: bytes) -> dict[str, Any]:
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("not a glb")
    chunk_len = int.from_bytes(data[12:16], "little")
    chunk_type = data[16:20]
    if chunk_type != b"JSON":
        raise ValueError("missing json chunk")
    return json.loads(data[20 : 20 + chunk_len])


def _pretty(name: str) -> str:
    text = re.sub(r"[_\-]+", " ", name).strip()
    return text[:1].upper() + text[1:] if text else "Part"


# ---- splitting a single generated mesh into parts ----
#
# Meshy's image-to-3D API has no part / segmentation option (its texture_prompt only guides texturing), so
# its GLB is usually one textured mesh. When it is, `segment_glb` splits what the geometry itself shows:
#   1. disconnected pieces (generated legs, cushions or handles are often separate islands), found on a copy
#      with vertices merged by position so UV seams do not cut the mesh into fake islands;
#   2. legs: the bottom band, only when it covers little of the footprint (thin supports, not a solid plinth);
#   3. a tabletop: the top band of a table, only when it covers most of the footprint (a real flat top);
#   4. a chair or sofa: backrest / back cushions, seat cushion(s) and front / rear legs, cut at the seat (`_seating`).
# Parts keep their faces' UVs and texture, so the model looks exactly as generated; names come from the
# furniture type the vendor selected. What geometry cannot tell apart (cushions fused to a frame, doors to a
# wardrobe body) stays one piece, and the returned document says so.

MIN_PART_FACES = 8  # a part needs at least a box's worth of triangles…
MIN_PART_SIZE = 0.04  # …and to be at least this share of the model's size (noise specks are not parts)
LEG_BAND = {"sofa": 0.1, "bed": 0.12, "chair": 0.12, "table": 0.0, "wardrobe": 0.08, "furniture": 0.1}
LEG_COVER = 0.35  # the bottom band covering less of the footprint than this = legs
TOP_BAND = 0.12
TOP_COVER = 0.5
GRID = 20


def segment_glb(data: bytes, family: str) -> tuple[bytes, dict[str, Any]] | None:
    """A modular GLB and its component document, or None when the file already has parts or nothing splits."""
    import io

    import numpy as np
    import trimesh

    try:
        scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene")
    except Exception:  # noqa: BLE001 - an unreadable file is simply not split
        return None
    meshes = [m for m in scene.dump() if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    if len(meshes) != 1:
        return None  # several meshes already (inspect_glb lists them) or nothing to split
    mesh = meshes[0]
    labels = _labels(mesh, family, np, trimesh)
    names = sorted(set(labels), key=_order)
    if len(names) < 2:
        return None
    out = trimesh.Scene()
    parts = []
    for name in names:
        faces = np.flatnonzero(np.array(labels) == name)
        part = mesh.submesh([faces], append=True)
        parts.append(part)
        out.add_geometry(part, geom_name=name, node_name=name)
    # Every original face in exactly one part, each with its own UVs: a cut that lost geometry or texture is not used.
    uv = getattr(mesh.visual, "uv", None)
    if sum(len(p.faces) for p in parts) != len(mesh.faces) or (uv is not None and any(getattr(p.visual, "uv", None) is None or len(p.visual.uv) != len(p.vertices) for p in parts)):
        logger.warning("Geometry split of a %s dropped faces or UVs; the model is kept whole", family)
        return None
    glb = out.export(file_type="glb")
    doc = inspect_glb(glb, family)
    if not doc.get("separated"):
        return None
    doc["segmented"] = True
    doc["method"] = "geometry"
    merged = {"sofa": "cushions and frame", "bed": "mattress, headboard and frame", "wardrobe": "doors and body", "table": "frame and legs"}.get(family)
    doc["limits"] = (
        "Split from the generated geometry (Meshy returns one mesh): " + ", ".join(_pretty(n) for n in names) + "."
        + (f" Parts fused in the generated surface ({merged}) can only be split where the geometry separates them." if merged else "")
    )
    return glb, doc


def _order(name: str) -> tuple[int, str]:
    first = ("frame", "body", "base", "leaf")
    return (0 if name.startswith(first) else 1, name)


def _labels(mesh, family: str, np, trimesh) -> list[str]:
    """One part name per face."""
    n = len(mesh.faces)
    centroids = mesh.triangles_center
    lo, hi = mesh.bounds
    height = max(hi[1] - lo[1], 1e-9)
    size = float(np.linalg.norm(hi - lo))
    base = {"sofa": "frame", "bed": "frame", "table": "frame", "chair": "frame", "wardrobe": "body", "door": "leaf"}.get(family, "body")
    labels = [base] * n

    # Connected islands on a position-merged copy (same face order), so UV seams do not split them.
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    island = _components(welded.face_adjacency, n, np)
    counts = np.bincount(island)
    main = int(np.argmax(counts))
    extra: dict[int, list[int]] = {}
    for k in range(len(counts)):
        if k == main or counts[k] < MIN_PART_FACES:
            continue
        faces = np.flatnonzero(island == k)
        ext = np.ptp(mesh.vertices[mesh.faces[faces].reshape(-1)], axis=0)
        if float(np.linalg.norm(ext)) >= MIN_PART_SIZE * size:
            extra[k] = faces.tolist()

    span = np.maximum(hi[[0, 2]] - lo[[0, 2]], 1e-9)
    rng = np.random.default_rng(0)

    def cover(faces) -> float:
        """Share of the model's footprint rectangle (a GRID × GRID raster) under these faces, seen from above."""
        tri = mesh.triangles[faces][:, :, [0, 2]]
        cells = (tri - lo[[0, 2]]) / span * GRID
        e1, e2 = cells[:, 1] - cells[:, 0], cells[:, 2] - cells[:, 0]
        area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2  # 2-D cross product
        hit = np.zeros((GRID, GRID), dtype=bool)
        for t, a in zip(cells, area):
            k = int(min(400, max(3, np.ceil(a * 3))))
            u, v = rng.random(k), rng.random(k)
            flip = u + v > 1
            u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
            pts = t[0] + np.outer(u, t[1] - t[0]) + np.outer(v, t[2] - t[0])
            ij = np.clip(np.floor(np.vstack([pts, t])).astype(int), 0, GRID - 1)
            hit[ij[:, 0], ij[:, 1]] = True
        return float(hit.mean())
    # Islands: named by where they sit (a table's top island, legs on the floor, cushions above the seat…).
    counter: dict[str, int] = {}
    for k, faces in extra.items():
        pts = centroids[faces]
        bottom, top = (pts[:, 1].min() - lo[1]) / height, (pts[:, 1].max() - lo[1]) / height
        small = float(np.linalg.norm(np.ptp(pts, axis=0))) < 0.15 * size
        if family == "table" and top > 1 - TOP_BAND and cover(faces) >= TOP_COVER:
            name = "tabletop"
        elif bottom < 0.05 and cover(faces) < 0.1:  # stands on the floor and is thin: a leg (any height)
            name = "leg"
        elif family in ("wardrobe", "door") and small:
            name = "handle"
        elif family in ("sofa", "chair"):
            continue  # cut by height below (_seating): an island above the seat is part of the back, etc.
        elif family == "bed":
            name = "pillow" if bottom > 0.35 else "mattress"
        else:
            name = "part"
        counter[name] = counter.get(name, 0) + 1
        part = name if name == "tabletop" else f"{name}_{counter[name]}"
        for f in faces:
            labels[f] = part

    rest = np.array([labels[f] == base for f in range(n)])
    # A table's top band, when it is a full flat top (most of the footprint).
    if family == "table" and not any(l == "tabletop" for l in labels):
        band = rest & (centroids[:, 1] > hi[1] - TOP_BAND * height)
        if band.sum() >= MIN_PART_FACES and cover(np.flatnonzero(band)) >= TOP_COVER:
            for f in np.flatnonzero(band):
                labels[f] = "tabletop"
            rest &= ~band
    # Legs: the bottom band when it covers little of the footprint (thin supports, not a solid base).
    leg_band = LEG_BAND.get(family, LEG_BAND["furniture"]) or (0.25 if family == "table" else 0.0)
    if leg_band and not any(l.startswith("leg") for l in labels):
        band = rest & (centroids[:, 1] < lo[1] + leg_band * height)
        if band.sum() >= MIN_PART_FACES and cover(np.flatnonzero(band)) < LEG_COVER:
            for f in np.flatnonzero(band):
                labels[f] = "legs"
    if family in ("chair", "sofa"):
        _seating(mesh, labels, base, family, cover, np)
    return labels


def _seating(mesh, labels: list[str], base: str, family: str, cover, np) -> None:
    """A chair or sofa from one fused surface: the seat is the height with the most upward-facing area (20–75% up),
    the back the side the surface above it leans to. Then: backrest / back cushions (above the seat, back part), seat
    cushion(s) (a band down from the seat top), front / rear legs (thin supports under the seat, by side). Without a
    clear seat or back it is left as it is (a stool, a bench)."""
    lo, hi = mesh.bounds
    height, depth = max(hi[1] - lo[1], 1e-9), max(hi[2] - lo[2], 1e-9)
    c = mesh.triangles_center
    y = (c[:, 1] - lo[1]) / height
    z = (c[:, 2] - lo[2]) / depth
    up = (mesh.face_normals[:, 1] > 0.7) & (y > 0.2) & (y < 0.75)
    if mesh.area_faces[up].sum() < 0.03 * mesh.area_faces.sum():
        return
    hist, edges = np.histogram(y[up], bins=22, range=(0.2, 0.75), weights=mesh.area_faces[up])
    seat = float(edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2)
    above = y > seat + 0.2
    if above.sum() < MIN_PART_FACES or abs(float(z[above].mean()) - 0.5) < 0.08:
        return  # nothing leans back above the seat: no backrest to cut
    back = z if z[above].mean() < 0.5 else 1 - z  # 0 at the back
    chair = family == "chair"
    # Legs: everything under the seat cushion, when it covers little of the footprint (not a solid base).
    under = np.array([l == base or l.startswith("leg") for l in labels]) & (y < seat - 0.15)
    legs = under.sum() >= MIN_PART_FACES and cover(np.flatnonzero(under)) < LEG_COVER
    for name, mask in (
        ("backrest" if chair else "back_cushions", (y > seat + 0.06) & (back < 0.4)),
        ("seat_cushion" if chair else "seat_cushions", (y > seat - 0.15) & (y <= seat + 0.06) & (back >= 0.15)),
    ):
        mask &= np.array([l == base for l in labels])
        if mask.sum() >= MIN_PART_FACES:
            for i in np.flatnonzero(mask):
                labels[i] = name
    for i, l in enumerate(labels):
        if l.startswith("leg") or (legs and under[i]):
            labels[i] = "front_legs" if back[i] >= 0.5 else "rear_legs"


def _components(pairs, n: int, np):
    """Connected component label per face from face-adjacency pairs (union-find; no SciPy in this deployment)."""
    parent = np.arange(n)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = root(int(a)), root(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels
