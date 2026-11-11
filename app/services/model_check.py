"""Checks a model before it is saved, so a damaged one is rejected or flagged instead of shown as a success.

`check_glb` returns {"ok", "errors", "warnings", "parts"}. Errors are things a customer would see as broken (no
geometry, degenerate or non-finite parts, overlapping doors or drawers, floating parts, impossible sizes, parts that
cannot be edited alone); warnings are things worth a look (generic names, a missing handle).

ponytail: overlap and floating are box tests (axis-aligned bounds), exact for panel furniture and approximate for
curved pieces; a mesh-mesh intersection test is the upgrade if curved parts get false alarms.
"""

from __future__ import annotations

import io
from typing import Any

from app.services.part_names import is_generic

MIN_SIZE, MAX_SIZE = 0.2, 3.5  # metres: the whole model, on every axis
OVERLAP = 1e-6  # m³ two fronts may share (touching edges are fine)
FLOAT_GAP = 0.01  # m: a part this far from every other part and above the floor is floating
FRONTS = ("door", "drawer")


def _boxes_overlap(a, b) -> float:
    lo = [max(a[0][i], b[0][i]) for i in range(3)]
    hi = [min(a[1][i], b[1][i]) for i in range(3)]
    d = [hi[i] - lo[i] for i in range(3)]
    return d[0] * d[1] * d[2] if all(x > 0 for x in d) else 0.0


def _gap(a, b) -> float:
    return max(max(a[0][i] - b[1][i], b[0][i] - a[1][i], 0.0) for i in range(3))


def check_glb(data: bytes, family: str = "furniture") -> dict[str, Any]:
    import numpy as np
    import trimesh

    errors: list[str] = []
    warnings: list[str] = []
    try:
        scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene")
    except Exception as exc:  # noqa: BLE001 - an unreadable file is the error
        return {"ok": False, "errors": [f"The file cannot be read as a GLB ({str(exc)[:100]})."], "warnings": [], "parts": 0}

    parts: list[tuple[str, Any, Any]] = []  # name, world mesh, material
    for node in scene.graph.nodes_geometry:
        transform, geom = scene.graph[node]
        mesh = scene.geometry.get(geom)
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        world = mesh.copy()
        world.apply_transform(transform)
        parts.append((str(node), world, getattr(mesh.visual, "material", None)))
    if not parts:
        return {"ok": False, "errors": ["The model has no geometry."], "warnings": [], "parts": 0}

    for name, mesh, _ in parts:
        if not len(mesh.faces):
            errors.append(f"{name}: no faces.")
            continue
        if not np.isfinite(mesh.vertices).all():
            errors.append(f"{name}: has non-finite vertices.")
            continue
        if mesh.area <= 1e-8 or float(mesh.extents.max()) <= 1e-4:
            errors.append(f"{name}: degenerate (no area).")
            continue
        bad = 1 - float(mesh.nondegenerate_faces(height=1e-9).mean())
        if bad > 0.05:
            errors.append(f"{name}: {bad:.0%} of its triangles are degenerate.")

    lo = np.min([m.bounds[0] for _, m, _ in parts], axis=0)
    hi = np.max([m.bounds[1] for _, m, _ in parts], axis=0)
    size = hi - lo
    if (size < MIN_SIZE).any() or (size > MAX_SIZE).any():
        errors.append(f"Overall size {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f} m is not a real piece of furniture.")

    names = sorted(n for n, _, _ in parts)
    generic = [n for n in names if is_generic(n)]
    if generic:
        warnings.append(f"Parts without a meaningful name: {', '.join(generic[:6])}.")
    if len(parts) > 1:
        materials = [id(m) for _, _, m in parts if m is not None]
        if len(set(materials)) < len(materials):
            errors.append("Some parts share one material, so recolouring one would change the others.")

    boxes = {n: m.bounds.tolist() for n, m, _ in parts}
    fronts = [n for n in names if n.startswith(FRONTS)]
    for i, a in enumerate(fronts):
        for b in fronts[i + 1:]:
            if _boxes_overlap(boxes[a], boxes[b]) > OVERLAP:
                errors.append(f"{a} and {b} overlap.")
    if len(parts) > 1:
        for n in names:
            if boxes[n][0][1] - lo[1] > FLOAT_GAP and all(_gap(boxes[n], boxes[o]) > FLOAT_GAP for o in names if o != n):
                errors.append(f"{n} floats, touching no other part.")

    if family == "wardrobe" and len(parts) > 1:
        if not any(n.startswith(("wardrobe_body", "body", "frame")) for n in names):
            errors.append("No wardrobe body.")
        if not fronts:
            warnings.append("No doors or drawers found.")
        if fronts and not any(n.startswith("handle") for n in names):
            warnings.append("Doors and drawers without handles.")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "parts": len(parts)}
