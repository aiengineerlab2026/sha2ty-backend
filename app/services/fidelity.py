"""Where visual fidelity is lost: each stage of photo → visual model checked against the one before it.

  1. cleaning    original photo  vs cleaned photo   (colours, proportions of the product)   only when cleaned
  2. provider    original photo  vs raw provider GLB (front-view colour render, model_check.appearance)
  3. processing  raw GLB         vs processed GLB   (size, triangles, front-view colours)
  4. materials   raw GLB         vs processed GLB   (texture maps and PBR extensions kept)
  5. structural  the structural model's own checks  (built from the spec, closed solids)

Each stage is ok / not ok / not judged (None) with its issues; `lostAt` names the first stage that failed, and a
model with any failed stage is quality "low": flagged for review, never shown as a clean success.

Beyond colour palettes, each image pair is compared as camera-aligned images: the GLB rendered from the front
(orthographic, +z, y up: `front_render`) against the product's box in the photo, cell by cell on an 8 × 8 grid
(where the major components and colour regions sit) and by edge structure (gradient correlation).

ponytail: a point-sampled z-buffer render without lighting, no GPU or SSIM. A photo taken from three-quarters
does not line up exactly with a front render, so layout is a diagnostic with a loose threshold; a learned
perceptual metric is the upgrade if shape mistakes need finer judging.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np

from app.services import model_check

logger = logging.getLogger(__name__)

SIZE_TOLERANCE = 0.02  # processed bounding box may differ from the raw one by this share
TRIANGLE_LOSS = 0.05  # processed may have this share fewer triangles (degenerate ones removed)
ASPECT_TOLERANCE = 0.12  # cleaned product's aspect ratio against the photo's: more and the cleaned image is rejected
GRID = 8  # layout cells per side
CELL_DE = 25  # a cell whose mean colour is this far (CIE76 ΔE) looks different
CLEAN_LAYOUT_SHARE = 0.35  # more cells than this changed by cleaning: components changed, not just objects removed
CLEAN_STRUCTURE_MIN = 0.4  # edge-structure correlation below this: cleaning redrew the furniture
MODEL_ASPECT_TOLERANCE = 0.12  # the provider model's width : height against the photo reading's
MODEL_LAYOUT_SHARE = 0.5  # front render cells unlike the photo (loose: a three-quarter photo is not a front view)
MAP_KINDS = ("baseColorTexture", "metallicRoughnessTexture", "normalTexture", "occlusionTexture", "emissiveTexture")


def _missing_colours(ref: list, other: list, what: str, against: str) -> list[str]:
    shown = [p for p in other if p[1] >= 0.03] or other
    issues = []
    for hex_, share, lab in ref:
        if share < model_check.MIN_SHARE or not shown:
            continue
        de = min(float(np.linalg.norm(np.subtract(lab, m[2]))) for m in shown)
        if de > model_check.MISSING_DE:
            issues.append(f"{what}'s {hex_} ({share:.0%}) is missing from {against} (nearest ΔE {de:.0f}).")
    return issues


def _foreground_box(image: bytes) -> tuple[list[float] | None, float | None]:
    """The product's box in a cleaned photo (not transparent, not the plain white background), normalised, and its
    aspect ratio in pixels."""
    from PIL import Image

    img = Image.open(io.BytesIO(image)).convert("RGBA")
    a = np.asarray(img, dtype=float)
    mask = (a[..., 3] > 16) & ~((a[..., :3] > 242).all(axis=-1))
    if mask.sum() < 50:
        return None, None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    box = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
    return box, (xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1)


def _crop(image: bytes, box: list[float] | None) -> tuple[np.ndarray, np.ndarray]:
    """The product's box of an image as RGB (0..255) and its mask (not transparent, not plain white background)."""
    from PIL import Image

    img = Image.open(io.BytesIO(image)).convert("RGBA")
    if box and len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
        w, h = img.size
        img = img.crop((int(box[0] * w), int(box[1] * h), max(int(box[2] * w), int(box[0] * w) + 1), max(int(box[3] * h), int(box[1] * h) + 1)))
    a = np.asarray(img.resize((64, 64)), dtype=float)
    return a[..., :3], a[..., 3] > 16


def _cells(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean Lab per grid cell and whether the cell has enough of the product in it."""
    h, w = mask.shape
    labs, valid = np.zeros((GRID, GRID, 3)), np.zeros((GRID, GRID), bool)
    for i in range(GRID):
        for j in range(GRID):
            sl = (slice(i * h // GRID, (i + 1) * h // GRID), slice(j * w // GRID, (j + 1) * w // GRID))
            m = mask[sl]
            if m.mean() > 0.3:
                labs[i, j] = model_check._lab(rgb[sl][m].mean(axis=0, keepdims=True))[0]
                valid[i, j] = True
    return labs, valid


def layout_difference(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float | None:
    """Share of the grid cells (product in both) whose colour differs: where the components and colour regions sit."""
    la, va = _cells(*a)
    lb, vb = _cells(*b)
    both = va & vb
    if both.sum() < GRID:
        return None
    de = np.linalg.norm(la - lb, axis=-1)[both]
    return float((de > CELL_DE).mean())


def structure_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Correlation of the two images' edge maps (gradient magnitude of luminance): 1 the same edges, 0 unrelated."""
    def edges(rgb: np.ndarray) -> np.ndarray:
        g = rgb @ np.array([0.299, 0.587, 0.114])
        gy, gx = np.gradient(g)
        return np.hypot(gx, gy).ravel()

    ea, eb = edges(a), edges(b)
    if ea.std() < 1e-6 or eb.std() < 1e-6:
        return None
    return float(np.corrcoef(ea, eb)[0, 1])


def compare_images(original: bytes, cleaned: bytes, bbox: list[float] | None, removal: bool = False) -> dict[str, Any]:
    """Stage 1: did cleaning keep the product's colours, proportions and visible components? A failed stage means the
    cleaned image must not be meshed (vendor.conditioned_image rejects it before Meshy sees it).

    removal=True (Meshy Image to Image, no vision reading): cleaning is EXPECTED to take whole colour regions away
    (bedding, pillows, clutter), so a missing colour is a note, not a failure, and the layout / edge thresholds are
    looser. What still fails: an unreadable or tiny image, no furniture left, re-framing, and redrawn structure."""
    if removal:
        return _compare_removal(original, cleaned, bbox)
    from PIL import Image

    try:
        fg, aspect = _foreground_box(cleaned)
        ref = model_check._palette(*model_check._photo_colours(original, bbox))
        out = model_check._palette(*model_check._photo_colours(cleaned, fg))
        w, h = Image.open(io.BytesIO(original)).size
        before, after = _crop(original, bbox), _crop(cleaned, fg)
    except Exception as exc:  # noqa: BLE001 - nothing to compare is not judged
        return {"stage": "cleaning", "ok": None, "issues": [], "notes": [f"Not compared ({str(exc)[:80]})."]}
    issues = _missing_colours(ref, out, "The photo", "the cleaned image")
    metrics: dict[str, Any] = {}
    if bbox and aspect:
        photo_aspect = (bbox[2] - bbox[0]) * w / max((bbox[3] - bbox[1]) * h, 1e-6)
        change = abs(aspect - photo_aspect) / photo_aspect
        metrics["aspect"] = [round(photo_aspect, 3), round(float(aspect), 3)]
        if change > ASPECT_TOLERANCE:
            issues.append(f"Cleaning changed the product's proportions (width : height {photo_aspect:.2f} → {aspect:.2f}, {change:.0%}).")
    # The original's box has the room around the product: compare every cell of the box (mask all), the cleaned one's
    # product only.
    layout = layout_difference((before[0], np.ones_like(before[1])), after)
    structure = structure_correlation(before[0], after[0])
    metrics.update(layout=layout, structure=structure)
    if layout is not None and layout > CLEAN_LAYOUT_SHARE:
        issues.append(f"Cleaning changed the product's visible layout: {layout:.0%} of it looks different, more than removed objects explain.")
    if structure is not None and structure < CLEAN_STRUCTURE_MIN:
        issues.append(f"Cleaning redrew the furniture's edges and components (edge match {structure:.2f}).")
    return {"stage": "cleaning", "ok": not issues, "issues": issues, "metrics": metrics,
            "reference": [h for h, _, _ in ref], "cleaned": [h for h, _, _ in out]}


# Removal-mode thresholds (Meshy cleaning): legitimate removal of bedding / clutter changes many cells' colour, so only
# far larger changes count as the furniture being redrawn. ponytail: global constants, per-family tuning if needed.
REMOVAL_LAYOUT_SHARE = 0.65
REMOVAL_STRUCTURE_MIN = 0.2
REMOVAL_FRAME_TOLERANCE = 0.15  # canvas width:height change allowed (Meshy outputs fixed aspect ratios)
REMOVAL_MIN_SIDE = 256  # px: smaller is not a usable reconstruction input
REMOVAL_MIN_PRODUCT = 0.04  # the furniture's share of the canvas: less means it was removed or shrunk away
REMOVAL_ASPECT_TOLERANCE = 0.2  # looser than ASPECT_TOLERANCE: removed pillows above a headboard change the box a little


def _compare_removal(original: bytes, cleaned: bytes, bbox: list[float] | None) -> dict[str, Any]:
    from PIL import Image

    def out(issues: list[str], notes: list[str], metrics: dict[str, Any] | None = None, judged: bool = True) -> dict[str, Any]:
        # judged=False: not compared (ok None), which the caller must never report as a passed check
        return {"stage": "cleaning", "mode": "removal", "ok": (not issues) if judged else None, "issues": issues, "notes": notes, "metrics": metrics or {}}

    try:
        img = Image.open(io.BytesIO(cleaned))
        img.load()
        cw, ch = img.size
    except Exception:  # noqa: BLE001 - an image that cannot be decoded is not meshed
        return out(["The cleaned image is not a readable image."], [])
    if min(cw, ch) < REMOVAL_MIN_SIDE:
        return out([f"The cleaned image is too small ({cw}×{ch} px)."], [])
    fg, _ = _foreground_box(cleaned)
    if fg is None or (fg[2] - fg[0]) * (fg[3] - fg[1]) < REMOVAL_MIN_PRODUCT:
        return out(["The cleaned image has no furniture left in it (or almost none)."], [])
    try:
        w, h = Image.open(io.BytesIO(original)).size
    except Exception:  # noqa: BLE001
        return out([], ["The original photo could not be read: the cleaned image was not compared with it."], judged=False)
    frame = abs((cw / ch) - (w / h)) / (w / h)
    metrics: dict[str, Any] = {"size": [cw, ch], "frame": round(frame, 3), "productBox": [round(v, 3) for v in fg]}
    if frame > REMOVAL_FRAME_TOLERANCE:
        # Different canvas proportions: positions cannot be compared cell by cell. Not judged rather than guessed.
        return out([], [f"The cleaned image is framed differently ({w}×{h} → {cw}×{ch}): layout not compared."], metrics, judged=False)
    # Same framing: the furniture sits where it sat. With no box from a photo reading, the cleaned image's own product
    # box is the region compared in both (the original's still has the removed objects in it).
    region = bbox or fg
    before, after = _crop(original, region), _crop(cleaned, fg)
    layout = layout_difference((before[0], np.ones_like(before[1])), after)
    structure = structure_correlation(before[0], after[0])
    metrics.update(layout=layout, structure=structure)
    issues, notes = [], []
    if bbox and len(bbox) == 4:  # a reading's box of the product in the original: its proportions must survive
        before_aspect = (bbox[2] - bbox[0]) * w / max((bbox[3] - bbox[1]) * h, 1e-6)
        after_aspect = (fg[2] - fg[0]) * cw / max((fg[3] - fg[1]) * ch, 1e-6)
        change = abs(after_aspect - before_aspect) / before_aspect
        metrics["aspect"] = [round(before_aspect, 3), round(after_aspect, 3)]
        if change > REMOVAL_ASPECT_TOLERANCE:
            issues.append(f"Cleaning changed the furniture's proportions (width : height {before_aspect:.2f} → {after_aspect:.2f}, {change:.0%}).")
    if layout is not None and layout > REMOVAL_LAYOUT_SHARE:
        issues.append(f"Cleaning changed {layout:.0%} of the furniture's visible layout: more than removing accessories explains.")
    if structure is not None and structure < REMOVAL_STRUCTURE_MIN:
        issues.append(f"Cleaning redrew the furniture's edges and components (edge match {structure:.2f}).")
    ref = model_check._palette(*model_check._photo_colours(original, region))
    got = model_check._palette(*model_check._photo_colours(cleaned, fg))
    notes += [f"{m} (expected when accessories or bedding were removed)" for m in _missing_colours(ref, got, "The photo", "the cleaned image")]
    return out(issues, notes, metrics)


RENDER_COLS = 96


def front_render(glb: bytes, cols: int = RENDER_COLS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The GLB from the front: orthographic along -z (glTF's front is +z), y up, base colours and textures, no
    lighting. (RGB rows top → down, the product mask, the model's extents x y z.) The same camera for every model."""
    import trimesh

    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    pts, cols_ = [], []
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geom = scene.graph[node]
        mesh = scene.geometry.get(geom)
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) and mesh.area > 0:
            meshes.append((mesh, np.asarray(transform)))
    total = sum(m.area for m, _ in meshes) or 1.0
    for mesh, transform in meshes:
        n = max(400, int(max(200_000, cols * cols * 20) * mesh.area / total))
        p, faces = trimesh.sample.sample_surface(mesh, n, seed=0)
        pts.append(trimesh.transform_points(p, transform))
        cols_.append(model_check._face_colours(mesh)[faces])
    if not pts:
        raise ValueError("no surface to render")
    pts, colours = np.vstack(pts), np.vstack(cols_)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    rows = max(1, int(round(cols * span[1] / span[0])))
    ix = np.minimum(((pts[:, 0] - lo[0]) / span[0] * cols).astype(int), cols - 1)
    iy = np.minimum(((hi[1] - pts[:, 1]) / span[1] * rows).astype(int), rows - 1)  # row 0 = top
    cell = iy * cols + ix
    order = np.lexsort((pts[:, 2], cell))  # nearest to the camera (largest z) last
    last = np.flatnonzero(np.r_[cell[order][1:] != cell[order][:-1], True])
    image, mask = np.zeros((rows * cols, 3)), np.zeros(rows * cols, bool)
    image[cell[order[last]]] = colours[order[last]]
    mask[cell[order[last]]] = True
    return image.reshape(rows, cols, 3), mask.reshape(rows, cols), span


def render_png(glb: bytes, cols: int = 256) -> bytes:
    """The front render as a PNG (white where there is no model), for side-by-side comparisons."""
    from PIL import Image

    rgb, mask, _ = front_render(glb, cols)
    rgb[~mask] = 255
    buf = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _resized(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((64, 64))
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((64, 64))
    return np.asarray(img, dtype=float), np.asarray(m) > 127


def proportions(span: np.ndarray, expected: list[float] | None) -> tuple[list[str], dict[str, Any]]:
    """The model's width : height : depth against the photo reading's (`expected` width : depth : height, largest 1),
    and whether an axis looks swapped (the reading's width shows up as the model's depth, or it lies on its side)."""
    if not expected or len(expected) != 3 or min(expected) <= 0:
        return [], {}
    w, d, h = expected
    x, y, z = (float(v) for v in span)
    issues: list[str] = []
    model_aspect, photo_aspect = x / y, w / h
    change = model_aspect / photo_aspect - 1
    if abs(change) > MODEL_ASPECT_TOLERANCE:
        issues.append(f"The model is {abs(change):.0%} {'wider' if change > 0 else 'narrower'} than the photo for its height "
                      f"(width : height {model_aspect:.2f}, photo {photo_aspect:.2f}).")
    target = np.array([w, h, d]) / max(w, h, d)
    got = np.array([x, y, z]) / max(x, y, z)
    fits = {name: float(np.abs(np.log(np.maximum(got[list(perm)], 1e-6) / np.maximum(target, 1e-6))).sum())
            for name, perm in (("as is", (0, 1, 2)), ("width and depth swapped", (2, 1, 0)), ("on its side", (1, 0, 2)), ("on its back", (0, 2, 1)))}
    best = min(fits, key=fits.get)
    if best != "as is" and fits[best] < 0.6 * fits["as is"]:
        issues.append(f"The model's axes look wrong ({best}): its width × height × depth is {x:.2f} × {y:.2f} × {z:.2f}.")
    return issues, {"modelWHD": [round(x / y, 3), 1.0, round(z / y, 3)], "photoWHD": [round(w / h, 3), 1.0, round(d / h, 3)], "axisFit": {k: round(v, 3) for k, v in fits.items()}}


def provider(raw_glb: bytes, original: bytes | None, bbox: list[float] | None, cleaned: bytes | None = None,
             expected: list[float] | None = None, layout: bool = False) -> dict[str, Any]:
    """Stage 2: the raw provider model against the original photo. When the photo was cleaned, a colour that only the
    removed objects had is not the provider's loss: the colours pass if they match either image. `layout`: also the
    camera-aligned render (proportions, axes, where the components sit), for Meshy."""
    if not original:
        return {"stage": "provider", "ok": None, "issues": [], "notes": ["No photo to compare with."]}
    look = model_check.appearance(raw_glb, original, bbox)
    result: dict[str, Any] = {"stage": "provider", "ok": look["ok"], "issues": list(look["issues"]), "reference": look.get("reference"), "model": look.get("model")}
    if look.get("note"):
        result.update(ok=None, notes=[look["note"]])
    if not look["ok"] and cleaned:
        fg, _ = _foreground_box(cleaned)
        again = model_check.appearance(raw_glb, cleaned, fg)
        if again["ok"]:
            result.update(ok=True, issues=[], notes=["Matches the cleaned image; the photo's other colours were the removed objects."])
    if not layout:
        return result
    try:
        rgb, mask, span = front_render(raw_glb)
        ref = _crop(cleaned, _foreground_box(cleaned)[0]) if cleaned else _crop(original, bbox)
    except Exception as exc:  # noqa: BLE001 - an unreadable model or photo is not judged on layout
        result.setdefault("notes", []).append(f"Layout not compared ({str(exc)[:80]}).")
        return result
    shape_issues, metrics = proportions(span, expected)
    share = layout_difference(_resized(rgb, mask), (ref[0], ref[1] if cleaned else np.ones_like(ref[1])))
    metrics["layout"] = share
    if share is not None and share > MODEL_LAYOUT_SHARE:
        shape_issues.append(f"Major components are not where the photo shows them: {share:.0%} of the front view differs.")
    result["metrics"] = metrics
    if shape_issues:
        result.update(ok=False, issues=[*result["issues"], *shape_issues])
    return result


def mesh_count(glb: bytes) -> int:
    """Independent meshes in the file: geometry instances with faces (one fused mesh is 1)."""
    import trimesh

    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    return sum(1 for node in scene.graph.nodes_geometry if len(getattr(scene.geometry.get(scene.graph[node][1]), "faces", [])))


def split_check(raw: bytes, split: bytes | None, task: dict[str, Any]) -> dict[str, Any]:
    """Meshy Auto Split's result, accepted only when it really is the same model in several meshes: SUCCEEDED, the
    part names read (not prompt_ignored), more than one mesh in the file, the raw model's size and triangles, and
    its texture maps (keep_texture). Otherwise the unsplit model stays the visual one and `reasons` say why."""
    report: dict[str, Any] = {"stage": "split", **{k: task.get(k) for k in ("status", "taskId", "partCount", "promptIgnored", "error", "prompt")}}
    reasons: list[str] = []
    try:
        report["meshesBefore"] = mesh_count(raw)
    except Exception:  # noqa: BLE001
        report["meshesBefore"] = None
    if task.get("status") != "SUCCEEDED" or split is None:
        reasons.append(f"Auto Split did not finish ({task.get('status') or 'no result'}{': ' + str(task.get('error')) if task.get('error') else ''}).")
    else:
        if task.get("promptIgnored"):
            reasons.append("Auto Split ignored the part names (prompt_ignored): its cut would follow no real component.")
        try:
            report["meshesAfter"] = mesh_count(split)
        except Exception as exc:  # noqa: BLE001 - an unreadable split file is a failed split
            report["meshesAfter"] = None
            reasons.append(f"Auto Split's file cannot be read ({str(exc)[:80]}).")
            report.update(ok=False, accepted=False, issues=reasons)
            return report
        if report["meshesAfter"] < 2:
            reasons.append("Auto Split returned one fused mesh.")
        for stage in compare_glbs(raw, split):
            reasons += [f"After splitting: {issue}" for issue in stage["issues"] if "triangles" not in issue or stage["stage"] != "processing"]
            if stage["stage"] == "processing":
                tris = stage.get("metrics", {}).get("triangles") or [0, 0]
                if tris[0] and abs(tris[1] - tris[0]) / tris[0] > 0.1:
                    reasons.append(f"After splitting: the triangle count changed {tris[0]} → {tris[1]}: the surface was rebuilt.")
    report.update(ok=not reasons, accepted=not reasons, issues=reasons)
    return report


def _maps(doc: dict[str, Any]) -> tuple[set[str], set[str], int]:
    kinds: set[str] = set()
    for mat in doc.get("materials") or []:
        pbr = mat.get("pbrMetallicRoughness") or {}
        for k in MAP_KINDS:
            if k in pbr or k in mat:
                kinds.add(k)
    return kinds, set(doc.get("extensionsUsed") or []), len(doc.get("images") or [])


def _texture_side(doc: dict[str, Any], views: list[bytes]) -> int:
    from PIL import Image

    side = 0
    for image in doc.get("images") or []:
        view = image.get("bufferView")
        if view is None or view >= len(views):
            continue
        try:
            side = max(side, *Image.open(io.BytesIO(views[view])).size)
        except Exception:  # noqa: BLE001 - an unreadable image counts as none
            continue
    return side


def _geometry(glb: bytes) -> tuple[np.ndarray, int, list]:
    import trimesh

    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    tris = sum(len(g.faces) for g in scene.geometry.values() if hasattr(g, "faces"))
    px, w = model_check._model_colours(scene)
    return np.asarray(scene.extents, dtype=float), tris, model_check._palette(px, w) if px is not None else []


def compare_glbs(raw: bytes, processed: bytes) -> list[dict[str, Any]]:
    """Stages 3 and 4: what processing did to the provider's model (it must not change how it looks)."""
    from app.services.glb_size import _views, read_glb

    try:
        raw_doc, raw_bin = read_glb(raw)
        out_doc, out_bin = read_glb(processed)
        raw_ext, raw_tris, raw_pal = _geometry(raw)
        out_ext, out_tris, out_pal = _geometry(processed)
    except Exception as exc:  # noqa: BLE001
        note = [f"Not compared ({str(exc)[:80]})."]
        return [{"stage": "processing", "ok": None, "issues": [], "notes": note}, {"stage": "materials", "ok": None, "issues": [], "notes": note}]
    geo: list[str] = []
    scale = raw_ext.max() / max(out_ext.max(), 1e-9)  # a uniform rescale to metres is not a change of shape
    shape = np.abs(out_ext * scale - raw_ext) / np.maximum(raw_ext, 1e-9)
    if float(shape.max()) > SIZE_TOLERANCE:
        geo.append(f"The processed model's proportions differ from the provider's ({float(shape.max()):.0%} on one axis).")
    if raw_tris and out_tris < raw_tris * (1 - TRIANGLE_LOSS):
        geo.append(f"Processing removed {1 - out_tris / raw_tris:.0%} of the triangles ({raw_tris} → {out_tris}).")
    geo += _missing_colours(raw_pal, out_pal, "The provider model", "the processed model")
    raw_kinds, raw_exts, raw_images = _maps(raw_doc)
    out_kinds, out_exts, out_images = _maps(out_doc)
    mats: list[str] = []
    notes: list[str] = []
    flattened = [m.get("name") or "" for m in out_doc.get("materials") or [] if ((m.get("extras") or {}).get("sha2ty") or {}).get("flattened")]
    for k in sorted(raw_kinds - out_kinds):
        if k == "baseColorTexture" and flattened:
            continue  # replaced on purpose (surface_finish): said in the notes below, not a loss
        mats.append(f"The processed model lost its {k.replace('Texture', '')} map.")
    if flattened:
        notes.append(f"{len(flattened)} material(s) had photographed lighting tiled across them and now show their own even colour.")
    for e in sorted(raw_exts - out_exts - {"KHR_mesh_quantization", "KHR_texture_transform"}):
        mats.append(f"The processed model lost the {e} material extension.")
    if out_images < raw_images:
        mats.append(f"Textures dropped: {raw_images} → {out_images} images.")
    raw_side, out_side = _texture_side(raw_doc, _views(raw_doc, raw_bin)), _texture_side(out_doc, _views(out_doc, out_bin))
    if raw_side and out_side and out_side < raw_side:
        notes.append(f"Textures re-encoded from {raw_side} px to {out_side} px to fit the 8 MB file target.")
    meshes = [mesh_count(raw), mesh_count(processed)]
    return [
        {"stage": "processing", "ok": not geo, "issues": geo, "metrics": {"triangles": [raw_tris, out_tris], "meshes": meshes, "extents": [raw_ext.round(4).tolist(), out_ext.round(4).tolist()]}},
        {"stage": "materials", "ok": not mats, "issues": mats, "notes": notes, "metrics": {"maps": sorted(out_kinds), "images": [raw_images, out_images], "textureSide": [raw_side, out_side]}},
    ]


def structural(report: dict[str, Any] | None) -> dict[str, Any]:
    """Stage 5: the structural model's own checks (it never changes the visual one)."""
    if not report or not report.get("built"):
        return {"stage": "structural", "ok": None, "issues": [], "notes": list((report or {}).get("uncertain") or ["Not built."])}
    checks = report.get("checks") or {}
    issues = [w for w in checks.get("warnings") or [] if "occupy the same space" in w or "not a closed solid" in w]
    return {"stage": "structural", "ok": bool(checks.get("closed")) and not issues, "issues": issues, "notes": list(report.get("uncertain") or [])}


def report(stages: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [s for s in stages if s.get("ok") is False]
    # A rejected split or cleaned image was not used: reported, but the visual model did not lose fidelity by it.
    visual_failed = [s for s in failed if s["stage"] not in ("structural", "split") and not s.get("rejected")]
    return {
        "version": 1,
        "ok": not visual_failed,
        "quality": "low" if visual_failed else "ok",
        "lostAt": visual_failed[0]["stage"] if visual_failed else None,
        "structuralOk": next((s["ok"] for s in stages if s["stage"] == "structural"), None),
        "stages": stages,
    }
