"""Photographed lighting baked into a repeating surface texture, replaced by the surface's own even colour.

A generator that writes its own GLB (GPT Astra) can give a panel's material a small crop of the product photo as its
colour texture, repeated across the panel. That crop carries the photo's lighting (a highlight, the glow of display
lights), not the finish: tiled four or five times over a 2.9 m roof or back it shows as a grid of rectangular
light and dark patches. Measured on the wardrobe that showed it: such textures vary 0.46–0.48 in brightness at low
frequency (an 8 × 8 average, relative to their mean) where real wood, veneer and melamine swatches vary 0.002–0.035.

`flatten` finds a colour texture that is both REPEATED across a part (its UVs span more than `TILED` tiles: not a
UV atlas, which Meshy's are) and SHADED (low-frequency variation above `SHADED`), and gives the materials using it
their texture's median colour (times their own factor) instead. Nothing else changes: geometry, UVs, names, other
maps, PBR factors and every other texture stay as they are, and each changed material says so in its extras.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np

from app.services.glb_size import _views, read_glb, write_glb

logger = logging.getLogger(__name__)

TILED = 1.2  # UV span (in tiles) above which a texture is a repeated swatch, not an atlas
SHADED = 0.15  # low-frequency brightness variation above which a swatch carries photographed lighting
LUMA = np.array([0.2126, 0.7152, 0.0722])


def shading(image) -> float:
    """Brightness variation that survives an 8 × 8 average, relative to the mean: lighting, not grain."""
    from PIL import Image

    rgb = image.convert("RGB")
    coarse = np.asarray(rgb.resize((8, 8), Image.BOX), dtype=float) / 255 @ LUMA
    full = np.asarray(rgb, dtype=float) / 255 @ LUMA
    return float(coarse.std() / max(full.mean(), 1e-3))


def _linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _hex(linear: list[float]) -> str:
    srgb = [c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in linear]
    return "#" + "".join(f"{min(max(round(c * 255), 0), 255):02x}" for c in srgb)


def _accessor_span(doc: dict[str, Any], views: list[bytes], index: int) -> float:
    from app.services.realistic_materials import _accessor

    uv = _accessor(doc, views, index)
    return float(np.ptp(uv, axis=0).max()) if len(uv) else 0.0


def flatten(data: bytes) -> tuple[bytes, list[dict[str, Any]]]:
    """(the GLB, what was changed). The input comes back as it was when nothing qualifies or it cannot be read."""
    from PIL import Image

    try:
        doc, binary = read_glb(data)
        views = _views(doc, binary)
    except (ValueError, KeyError, IndexError):
        return data, []
    materials = doc.get("materials") or []
    textures = doc.get("textures") or []
    images = doc.get("images") or []
    # How far each colour texture is repeated over the parts that use it.
    span: dict[int, float] = {}
    for mesh in doc.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            mat = materials[prim["material"]] if isinstance(prim.get("material"), int) and prim["material"] < len(materials) else None
            tex = ((mat or {}).get("pbrMetallicRoughness") or {}).get("baseColorTexture")
            uv = prim.get("attributes", {}).get(f"TEXCOORD_{(tex or {}).get('texCoord', 0)}")
            if not tex or uv is None or tex["index"] >= len(textures):
                continue
            try:
                span[tex["index"]] = max(span.get(tex["index"], 0.0), _accessor_span(doc, views, uv))
            except (KeyError, IndexError, ValueError):
                continue
    flat: dict[int, tuple[list[float], float]] = {}
    for t, tiles in span.items():
        source = textures[t].get("source")
        view = images[source].get("bufferView") if isinstance(source, int) and source < len(images) else None
        if tiles <= TILED or view is None:
            continue
        try:
            image = Image.open(io.BytesIO(views[view]))
            score = shading(image)
        except Exception:  # noqa: BLE001 - an unreadable image is left as it is
            continue
        if score > SHADED:
            median = np.median(np.asarray(image.convert("RGB"), dtype=float).reshape(-1, 3), axis=0) / 255
            flat[t] = (_linear(median).round(4).tolist(), round(score, 3))
    changed: list[dict[str, Any]] = []
    for m in materials:
        pbr = m.get("pbrMetallicRoughness") or {}
        tex = pbr.get("baseColorTexture")
        if not tex or tex["index"] not in flat:
            continue
        colour, score = flat[tex["index"]]
        factor = pbr.get("baseColorFactor") or [1, 1, 1, 1]
        pbr["baseColorFactor"] = [round(colour[i] * factor[i], 4) for i in range(3)] + [factor[3] if len(factor) > 3 else 1]
        del pbr["baseColorTexture"]
        m.setdefault("extras", {})["sha2ty"] = {"flattened": "photographed lighting in a repeated colour texture", "shading": score}
        changed.append({"material": m.get("name") or "", "colour": _hex(pbr["baseColorFactor"][:3]), "shading": score, "tiles": round(span[tex["index"]], 1)})
    if not changed:
        return data, []
    logger.info("Surface finish: %d material(s) had photographed lighting tiled across them: %s", len(changed), [c["material"] for c in changed])
    return write_glb(doc, views), changed
