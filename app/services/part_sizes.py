"""Each part's size in a model's GLB, for the request report (the same rules as frontend lib/vendor/partSizes.ts):
width × height × depth of the part's box, one piece when a part is several alike pieces (four legs: one leg, ×4),
scaled to the requested size, and no size for soft furnishings (pillows, the mattress, bedding)."""

from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
import trimesh

SOFT = re.compile(r"pillow|mattress|duvet|blanket|throw|bedding|bed ?sheet|quilt|وساد|مخد|مرتب|فرش|لحاف|بطانية|ملاية|غطاء السرير", re.I)
MAX_FACES = 400_000  # ponytail: bigger parts are measured as one box


def is_soft(name: str) -> bool:
    return bool(SOFT.search(name or ""))


def _pieces(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """The separate pieces of one part, as (min, max) boxes: vertices at one place and of one triangle belong together."""
    if len(faces) == 0 or len(faces) > MAX_FACES:
        return [np.array([vertices.min(0), vertices.max(0)])]
    _, same = np.unique(np.round(vertices, 4), axis=0, return_inverse=True)
    parent = np.arange(same.max() + 1)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in same[faces]:
        ra, rb, rc = find(a), find(b), find(c)
        parent[ra] = rb
        parent[rc] = rb
    roots = np.array([find(i) for i in range(len(parent))])[same]  # each vertex's piece
    return [np.array([vertices[roots == r].min(0), vertices[roots == r].max(0)]) for r in np.unique(roots)]


def size_of(boxes: list[np.ndarray]) -> dict[str, float]:
    """One piece's size when the part is 2–12 alike pieces (tufting buttons and specks are not pieces), else the part's."""
    sizes = [b[1] - b[0] for b in boxes]
    biggest = max(float(np.linalg.norm(s)) for s in sizes)
    main = [s for s in sizes if np.linalg.norm(s) > biggest * 0.3]
    first = main[0]
    alike = 1 < len(main) <= 12 and all(np.all(np.abs(s - first) <= first * 0.15 + 0.005) for s in main)
    whole = np.max([b[1] for b in boxes], 0) - np.min([b[0] for b in boxes], 0)
    s = first if alike else whole
    return {"width": float(s[0]), "height": float(s[1]), "depth": float(s[2]), "count": len(main) if alike else 1}


def measure(glb: bytes, parts: list[dict[str, Any]], dimensions: dict[str, float] | None) -> list[dict[str, Any]]:
    """[{name, width, height, depth, count}] in metres for each part, in the file's order; parts are named from the
    request's part list (by id, else by order). Scaled per axis so the whole model is the requested size."""
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry.get(geometry)
        if mesh is None or not hasattr(mesh, "faces"):
            continue
        meshes.append((str(node), trimesh.transformations.transform_points(mesh.vertices, transform), np.asarray(mesh.faces)))
    if not meshes:
        return []
    all_points = np.vstack([v for _, v, _ in meshes])
    extent = all_points.max(0) - all_points.min(0)
    want = np.array([(dimensions or {}).get(k) or 0 for k in ("width", "height", "depth")], dtype=float)
    scale = np.where((want > 0) & (extent > 1e-9), want / np.maximum(extent, 1e-9), 1.0)
    names = {p.get("id"): p.get("name") for p in parts}
    out = []
    for i, (node, vertices, faces) in enumerate(meshes):
        name = names.get(node) or (parts[i].get("name") if i < len(parts) else None) or node
        if is_soft(name) or is_soft(node):
            continue
        out.append({"name": name, **size_of(_pieces(vertices * scale, faces))})
    return out
