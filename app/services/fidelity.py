"""Where visual fidelity is lost: each stage of photo → visual model checked against the one before it.

  1. cleaning    original photo  vs cleaned photo   (colours, proportions of the product)   only when cleaned
  2. provider    original photo  vs raw provider GLB (front-view colour render, model_check.appearance)
  3. processing  raw GLB         vs processed GLB   (size, triangles, front-view colours)
  4. materials   raw GLB         vs processed GLB   (texture maps and PBR extensions kept)
  5. structural  the structural model's own checks  (built from the spec, closed solids)

Each stage is ok / not ok / not judged (None) with its issues; `lostAt` names the first stage that failed, and a
model with any failed stage is quality "low": flagged for review, never shown as a clean success.

ponytail: colour palettes and box sizes, no GPU render or SSIM (same ceiling as model_check.appearance: it catches
lost colours, lost maps, resized or decimated models; it cannot judge a subtly wrong shape).
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
ASPECT_TOLERANCE = 0.2  # cleaned product's aspect ratio against the photo's
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


def compare_images(original: bytes, cleaned: bytes, bbox: list[float] | None) -> dict[str, Any]:
    """Stage 1: did cleaning keep the product's colours and proportions?"""
    from PIL import Image

    try:
        fg, aspect = _foreground_box(cleaned)
        ref = model_check._palette(*model_check._photo_colours(original, bbox))
        out = model_check._palette(*model_check._photo_colours(cleaned, fg))
        w, h = Image.open(io.BytesIO(original)).size
    except Exception as exc:  # noqa: BLE001 - nothing to compare is not judged
        return {"stage": "cleaning", "ok": None, "issues": [], "notes": [f"Not compared ({str(exc)[:80]})."]}
    issues = _missing_colours(ref, out, "The photo", "the cleaned image")
    if bbox and aspect:
        photo_aspect = (bbox[2] - bbox[0]) * w / max((bbox[3] - bbox[1]) * h, 1e-6)
        change = abs(aspect - photo_aspect) / photo_aspect
        if change > ASPECT_TOLERANCE:
            issues.append(f"Cleaning changed the product's proportions (width : height {photo_aspect:.2f} → {aspect:.2f}).")
    return {"stage": "cleaning", "ok": not issues, "issues": issues,
            "reference": [h for h, _, _ in ref], "cleaned": [h for h, _, _ in out]}


def provider(raw_glb: bytes, original: bytes | None, bbox: list[float] | None, cleaned: bytes | None = None) -> dict[str, Any]:
    """Stage 2: the raw provider model's front against the original photo. When the photo was cleaned, a colour that
    only the removed objects had is not the provider's loss: the model passes if it matches either image."""
    if not original:
        return {"stage": "provider", "ok": None, "issues": [], "notes": ["No photo to compare with."]}
    look = model_check.appearance(raw_glb, original, bbox)
    result = {"stage": "provider", "ok": look["ok"], "issues": look["issues"], "reference": look.get("reference"), "model": look.get("model")}
    if look.get("note"):
        result.update(ok=None, notes=[look["note"]])
    if not look["ok"] and cleaned:
        fg, _ = _foreground_box(cleaned)
        again = model_check.appearance(raw_glb, cleaned, fg)
        if again["ok"]:
            result.update(ok=True, notes=["Matches the cleaned image; the photo's other colours were the removed objects."])
    return result


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
    for k in sorted(raw_kinds - out_kinds):
        mats.append(f"The processed model lost its {k.replace('Texture', '')} map.")
    for e in sorted(raw_exts - out_exts - {"KHR_mesh_quantization", "KHR_texture_transform"}):
        mats.append(f"The processed model lost the {e} material extension.")
    if out_images < raw_images:
        mats.append(f"Textures dropped: {raw_images} → {out_images} images.")
    raw_side, out_side = _texture_side(raw_doc, _views(raw_doc, raw_bin)), _texture_side(out_doc, _views(out_doc, out_bin))
    if raw_side and out_side and out_side < raw_side:
        notes.append(f"Textures re-encoded from {raw_side} px to {out_side} px to fit the 8 MB file target.")
    return [
        {"stage": "processing", "ok": not geo, "issues": geo, "metrics": {"triangles": [raw_tris, out_tris], "extents": [raw_ext.round(4).tolist(), out_ext.round(4).tolist()]}},
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
    visual_failed = [s for s in failed if s["stage"] != "structural"]
    return {
        "version": 1,
        "ok": not visual_failed,
        "quality": "low" if visual_failed else "ok",
        "lostAt": visual_failed[0]["stage"] if visual_failed else None,
        "structuralOk": next((s["ok"] for s in stages if s["stage"] == "structural"), None),
        "stages": stages,
    }
