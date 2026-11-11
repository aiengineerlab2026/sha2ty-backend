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


# ---- appearance: does the model look like its photo? ----
#
# ponytail: there is no GPU renderer on the server, so the model is "rendered" as a point-sampled orthographic front view
# with a z-buffer (base colours and textures, no lighting) and its colours compared with the photo's inside the
# product's box. It catches what went wrong before (a white model of a sage-green wardrobe, lost colour
# regions); it cannot judge shape. Upgrade: a headless render compared with SSIM, if shape mistakes need catching.

PALETTE = 5
MISSING_DE = 25  # CIE76 ΔE: a photo colour with no model colour this close is missing
MIN_SHARE = 0.12  # photo colours covering less than this are details, not the colour scheme


def _lab(rgb):
    """sRGB 0..255 (N × 3) → CIE Lab (D65)."""
    import numpy as np

    c = np.asarray(rgb, dtype=float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def _palette(rgb, weights, k: int = PALETTE) -> list[tuple[str, float, list[float]]]:
    """The main colours: weighted k-means in Lab, as (hex, share, lab), largest share first."""
    import numpy as np

    # Quantised to 8 levels a channel step, so a texture's many near-equal texels are one weighted colour.
    q = (np.asarray(rgb, dtype=float) // 8) * 8 + 4
    rgb, inverse = np.unique(q, axis=0, return_inverse=True)
    w = np.bincount(inverse.ravel(), weights=np.asarray(weights, dtype=float), minlength=len(rgb))
    lab = _lab(rgb)
    # Farthest-point start: the heaviest colour, then each colour farthest from those chosen (ignoring specks).
    big = w >= 0.01 * w.sum()
    chosen = [int(np.argmax(w))]
    while len(chosen) < min(k, int(big.sum())):
        far = np.min(((lab[:, None] - lab[chosen][None]) ** 2).sum(-1), axis=1) * big
        chosen.append(int(np.argmax(far)))
    centres = lab[chosen].copy()
    for _ in range(12):
        nearest = np.argmin(((lab[:, None] - centres[None]) ** 2).sum(-1), axis=1)
        for i in range(len(centres)):
            hit = nearest == i
            if w[hit].sum() > 0:
                centres[i] = np.average(lab[hit], axis=0, weights=w[hit])
    nearest = np.argmin(((lab[:, None] - centres[None]) ** 2).sum(-1), axis=1)
    out = []
    for i in range(len(centres)):
        hit = nearest == i
        share = float(w[hit].sum() / w.sum())
        if share > 0:
            mean = np.average(rgb[hit], axis=0, weights=w[hit])
            out.append(("#%02x%02x%02x" % tuple(int(round(v)) for v in mean), share, _lab([mean])[0].tolist()))
    return sorted(out, key=lambda p: -p[1])


def _photo_colours(photo: bytes, bbox: list[float] | None):
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(photo)).convert("RGB")
    x0, y0, x1, y1 = bbox if bbox and len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1] else (0.1, 0.1, 0.9, 0.9)
    # The middle of the product's box: its edges are mostly floor and wall.
    dx, dy = (x1 - x0) * 0.12, (y1 - y0) * 0.08
    w, h = img.size
    crop = img.crop((int((x0 + dx) * w), int((y0 + dy) * h), max(int((x1 - dx) * w), int((x0 + dx) * w) + 1), max(int((y1 - dy) * h), int((y0 + dy) * h) + 1)))
    px = np.asarray(crop.resize((48, 48)), dtype=float).reshape(-1, 3)
    return px, np.ones(len(px))


def _face_colours(mesh):
    """Each face's base colour: its texture at the face's UV centre × the material's factor, or vertex / flat colour."""
    import numpy as np
    import trimesh

    visual = mesh.visual
    if isinstance(visual, trimesh.visual.ColorVisuals):
        return visual.face_colors[:, :3].astype(float)
    mat = getattr(visual, "material", None)
    factor = getattr(mat, "baseColorFactor", None)
    factor = np.asarray(factor if factor is not None else getattr(mat, "main_color", [255, 255, 255, 255]), dtype=float)[:3]
    factor = factor / 255 if factor.max() > 1 else factor
    image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    uv = getattr(visual, "uv", None)
    if image is not None and uv is not None and len(uv) == len(mesh.vertices):
        tex = np.asarray(image.convert("RGB"), dtype=float)
        th, tw = tex.shape[:2]
        fuv = uv[mesh.faces].mean(axis=1) % 1.0  # trimesh's UVs: origin bottom left
        return tex[((1 - fuv[:, 1]) * (th - 1)).astype(int), (fuv[:, 0] * (tw - 1)).astype(int)] * factor
    return np.tile(factor * 255, (len(mesh.faces), 1))


RENDER = 96  # columns of the front-view render
SAMPLES = 120_000


def _model_colours(scene):
    """What a camera in front (+z, orthographic) sees: points sampled over every surface, the nearest one per pixel
    (a z-buffer), so a back panel behind closed doors does not count."""
    import numpy as np
    import trimesh

    pts, cols = [], []
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geom = scene.graph[node]
        mesh = scene.geometry.get(geom)
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) and mesh.area > 0:
            meshes.append((mesh, np.asarray(transform)))
    total = sum(m.area for m, _ in meshes)
    for mesh, transform in meshes:
        n = max(200, int(SAMPLES * mesh.area / total))
        p, faces = trimesh.sample.sample_surface(mesh, n, seed=0)
        pts.append(trimesh.transform_points(p, transform))
        cols.append(_face_colours(mesh)[faces])
    if not pts:
        return None, None
    pts, cols = np.vstack(pts), np.vstack(cols)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    rows = max(1, int(RENDER * span[1] / span[0]))
    ix = np.minimum(((pts[:, 0] - lo[0]) / span[0] * RENDER).astype(int), RENDER - 1)
    iy = np.minimum(((pts[:, 1] - lo[1]) / span[1] * rows).astype(int), rows - 1)
    cell = iy * RENDER + ix
    order = np.lexsort((pts[:, 2], cell))  # by pixel, nearest (largest z) last
    last = np.flatnonzero(np.r_[cell[order][1:] != cell[order][:-1], True])
    seen = cols[order[last]]
    return seen, np.ones(len(seen))


def _white_share(palette) -> float:
    return sum(share for _, share, (l, a, b) in palette if l > 88 and (a * a + b * b) ** 0.5 < 10)


def appearance(glb: bytes, photo: bytes, bbox: list[float] | None = None) -> dict[str, Any]:
    """{"ok", "issues", "reference": [{hex, share}], "model": [{hex, share}]}: the photo's colour scheme against the
    model's front. Unreadable inputs are not judged (ok, with a note)."""
    import numpy as np
    import trimesh

    try:
        ref_px, ref_w = _photo_colours(photo, bbox)
        scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
        mod_px, mod_w = _model_colours(scene)
    except Exception as exc:  # noqa: BLE001 - nothing to compare is not a failed model
        return {"ok": True, "issues": [], "note": f"Appearance not checked ({str(exc)[:80]})."}
    if mod_px is None:
        return {"ok": True, "issues": [], "note": "Appearance not checked (no front-facing surface)."}
    ref, mod = _palette(ref_px, ref_w), _palette(mod_px, mod_w)
    issues = []
    if _white_share(mod) > 0.6 and _white_share(ref) < 0.35:
        issues.append(f"The model is almost entirely white ({_white_share(mod):.0%} of its front) but the photo is not.")
    shown = [p for p in mod if p[1] >= 0.03]
    for hex_, share, lab in ref:
        if share < MIN_SHARE:
            continue
        de = min(float(np.linalg.norm(np.subtract(lab, m[2]))) for m in shown)
        if de > MISSING_DE:
            issues.append(f"The photo's {hex_} ({share:.0%} of the product) is missing from the model (nearest colour ΔE {de:.0f}).")
    chroma = lambda pal: sum(s * (l[1] ** 2 + l[2] ** 2) ** 0.5 for _, s, l in pal)  # noqa: E731
    if chroma(ref) > 12 and chroma(mod) < 0.45 * chroma(ref):
        issues.append(f"The model's colours are washed out (saturation {chroma(mod):.0f} against the photo's {chroma(ref):.0f}).")
    return {
        "ok": not issues,
        "issues": issues,
        "reference": [{"hex": h, "share": round(s, 3)} for h, s, _ in ref],
        "model": [{"hex": h, "share": round(s, 3)} for h, s, _ in mod],
    }


def with_appearance(report: dict[str, Any], glb: bytes, photo: bytes | None, bbox: list[float] | None = None) -> dict[str, Any]:
    """The structural report with the appearance check added: a model that does not look like its photo is flagged."""
    if not photo:
        return report
    look = appearance(glb, photo, bbox)
    return {**report, "ok": report["ok"] and look["ok"], "errors": report["errors"] + look["issues"], "appearance": look}


# ---- states: an open and a closed model of one product ----

FIXED_TOLERANCE = 1e-3  # m: a part that does not move between states sits where it was
MOVED = 0.05  # m: an open front has moved at least this far


def _parts(glb: bytes) -> dict[str, Any]:
    import trimesh

    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    out = {}
    for node in scene.graph.nodes_geometry:
        transform, geom = scene.graph[node]
        mesh = scene.geometry[geom].copy()
        mesh.apply_transform(transform)
        out[str(node)] = mesh.bounds
    return out


def check_states(closed: bytes, open_: bytes) -> dict[str, Any]:
    """Both states are one product: the same parts, the carcass and the inside exactly where they were (same size,
    orientation and origin), and the open state really open: its doors and drawers moved, not hidden or missing."""
    import numpy as np

    errors: list[str] = []
    warnings: list[str] = []
    try:
        a, b = _parts(closed), _parts(open_)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "errors": [f"The states cannot be compared ({str(exc)[:80]})."], "warnings": []}
    if set(a) != set(b):
        missing, extra = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        errors.append("The states do not have the same parts" + (f"; open lacks {', '.join(missing[:5])}" if missing else "") + (f"; open adds {', '.join(extra[:5])}" if extra else "") + ".")
    common = sorted(set(a) & set(b))
    fronts = [n for n in common if n.startswith(FRONTS)]
    fixed = [n for n in common if not n.startswith((*FRONTS, "handle", "decor"))]
    for n in fixed:
        if np.abs(a[n] - b[n]).max() > FIXED_TOLERANCE:
            errors.append(f"{n} is not where it is in the closed state (different size, orientation or origin).")
    moved = [n for n in fronts if np.abs(a[n] - b[n]).max() > MOVED]
    if fronts and not moved:
        errors.append("The open state's doors and drawers are where they are when closed: it is not open.")
    elif len(moved) < len(fronts):
        warnings.append(f"Not opened: {', '.join(n for n in fronts if n not in moved)}.")
    if not any(n.startswith(("shelf", "rail")) for n in common):
        warnings.append("Nothing inside to see when open (no shelves or hanging rail).")
    return {"ok": not errors, "errors": errors, "warnings": warnings}
