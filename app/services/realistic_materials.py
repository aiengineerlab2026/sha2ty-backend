"""Realistic materials for an existing 3D model (the mobile AR viewer's optional mode).

The model's geometry, sizes and colours are never touched: only each glTF material's physical properties change
(roughness, metalness, and the sheen / clearcoat extensions the app's renderer, Filament gltfio, supports). The
steps:

1. `regions`: one region per glTF material the meshes use (named or unnamed meshes alike), with its colour, its
   share of the surface, where it sits on the model and whether it has textures of its own.
2. `render`: two simple views of the model (a shaded look and a numbered region map), drawn here from sampled
   surface points (no GPU), so the vision model sees what each region is, not only its name.
3. The vision model the admin configured (vision.current_model) returns a `MaterialProfile` (one material class,
   finish and confidence per region), validated by `parse`; anything it is unsure of (or any region it skips) keeps
   its original look.
4. `apply`: the project's own material presets (frontend/lib/vendor/furnitureBuilder.ts FINISH) are written into
   the materials. Base colours and every texture (colour, normal, metallic-roughness) are kept: a part that already
   has a metallic-roughness map keeps its measured roughness and metalness, and only gains sheen / clearcoat.

The result is cached per file version (the SHA-256 of the GLB's bytes, so any change to the model is a new key)
and PROFILE_VERSION (bump it when the presets or the prompt change).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, ValidationError

from app.services.glb_size import _SIZES, _TYPES, _views, read_glb, write_glb

logger = logging.getLogger(__name__)

PROFILE_VERSION = 1
MAX_REGIONS = 24  # more materials than this: only the largest are asked about, the rest keep their look
MIN_CONFIDENCE = 0.6
POINTS = 220_000  # surface samples for the two views (all regions together)
VIEW = 448  # pixels per view

Kind = Literal[
    "fabric", "velvet", "leather", "boucle", "wood", "metal", "brass", "marble", "lacquer",
    "glass", "plastic", "stone", "ceramic", "rubber", "painted", "unknown",
]

# How each material catches light. The first nine are the furniture builder's FINISH table
# (frontend/lib/vendor/furnitureBuilder.ts): keep both in step. The rest cover other parts furniture has.
PRESETS: dict[str, dict[str, float]] = {
    "fabric": {"roughness": 0.92, "metalness": 0, "sheen": 0.4},
    "velvet": {"roughness": 0.75, "metalness": 0, "sheen": 1},
    "leather": {"roughness": 0.5, "metalness": 0, "clearcoat": 0.25},
    "boucle": {"roughness": 1, "metalness": 0, "sheen": 0.6},
    "wood": {"roughness": 0.55, "metalness": 0, "clearcoat": 0.15},
    "metal": {"roughness": 0.35, "metalness": 0.85},
    "brass": {"roughness": 0.3, "metalness": 1},
    "marble": {"roughness": 0.12, "metalness": 0, "clearcoat": 0.8},
    "lacquer": {"roughness": 0.2, "metalness": 0, "clearcoat": 1},
    "glass": {"roughness": 0.05, "metalness": 0, "clearcoat": 1},
    "plastic": {"roughness": 0.4, "metalness": 0},
    "stone": {"roughness": 0.6, "metalness": 0},
    "ceramic": {"roughness": 0.15, "metalness": 0, "clearcoat": 0.6},
    "rubber": {"roughness": 0.95, "metalness": 0},
    "painted": {"roughness": 0.45, "metalness": 0},
}
FINISH_SHIFT = {"matte": 0.15, "satin": 0.0, "gloss": -0.2}  # added to the preset's roughness


class RegionMaterial(BaseModel):
    id: str = Field(description="The region's id, exactly as listed (R1, R2, …).")
    material: Kind = Field(description="What the region is made of; unknown when it cannot be told.")
    finish: Literal["matte", "satin", "gloss"] = "satin"
    confidence: float = Field(ge=0, le=1, description="0–1: how sure, from the pictures, not from the name alone.")


class MaterialProfile(BaseModel):
    regions: list[RegionMaterial] = Field(default_factory=list)


@dataclass
class Region:
    """One part of the model that can take its own material: a mesh's primitives with one glTF material (several meshes
    sharing a material are separate regions; with too many meshes, one region per material)."""

    id: str  # R1, R2 … (by surface share)
    material: int  # index into doc["materials"]
    name: str
    meshes: list[str]
    color: tuple[float, float, float]  # its average linear colour (factor × texture, as drawn)
    textured: bool
    pbr_maps: bool  # has its own metallic-roughness texture
    unlit: bool
    prims: list[tuple[int, int]] = field(default_factory=list)  # (mesh, primitive) it covers
    share: float = 0.0  # of the model's surface
    band: str = ""  # bottom / middle / top of the model
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # sRGB 0–255 per point


class Unsupported(ValueError):
    """The file cannot be given realistic materials (no materials, no triangles, unreadable)."""


# ── Reading the file ─────────────────────────────────────────────────────────


def _accessor(doc: dict[str, Any], views: list[bytes], index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    if "bufferView" not in acc or acc.get("sparse"):
        raise Unsupported("sparse or empty accessor")
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_TYPES[acc["componentType"]])
    width = _SIZES[acc["type"]]
    count = acc["count"]
    blob = views[acc["bufferView"]]
    stride = view.get("byteStride") or dtype.itemsize * width
    start = acc.get("byteOffset", 0)
    rows = np.ndarray((count, width), dtype=dtype, buffer=blob, offset=start, strides=(stride, dtype.itemsize))
    out = rows.astype(np.float64)
    if acc.get("normalized") and dtype.kind in "iu":
        out /= float(np.iinfo(dtype).max)
        if dtype.kind == "i":
            out = np.maximum(out, -1.0)
    return out


def _local(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = node.get("translation", [0, 0, 0])
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = node.get("scale", [1, 1, 1])
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r * np.array(s)
    m[:3, 3] = t
    return m


def _instances(doc: dict[str, Any]) -> list[tuple[int, np.ndarray, str]]:
    """(mesh index, world matrix, node name) for every mesh in the default scene."""
    nodes = doc.get("nodes") or []
    scenes = doc.get("scenes") or [{"nodes": list(range(len(nodes)))}]
    roots = scenes[doc.get("scene", 0)].get("nodes", [])
    out: list[tuple[int, np.ndarray, str]] = []
    stack = [(i, np.eye(4)) for i in roots]
    seen = 0
    while stack and seen < 10_000:
        seen += 1
        i, parent = stack.pop()
        node = nodes[i]
        world = parent @ _local(node)
        if "mesh" in node:
            out.append((node["mesh"], world, node.get("name", "")))
        stack.extend((c, world) for c in node.get("children", []))
    return out


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _texture(doc: dict[str, Any], views: list[bytes], info: dict[str, Any] | None) -> np.ndarray | None:
    """An embedded colour texture as sRGB 0–255 (at most 512 px a side), or None."""
    try:
        source = doc["textures"][info["index"]]["source"]  # type: ignore[index]
        blob = views[doc["images"][source]["bufferView"]]
        with Image.open(io.BytesIO(blob)) as picture:
            picture = picture.convert("RGB")
            picture.thumbnail((512, 512))
            return np.asarray(picture, dtype=np.float64)
    except (KeyError, IndexError, TypeError, OSError, ValueError):
        return None


@dataclass
class _Piece:
    tris: np.ndarray  # (n, 3, 3) world positions
    uvs: np.ndarray | None  # (n, 3, 2)
    ref: tuple[int, int]
    label: str


def regions(data: bytes, rng: np.random.Generator | None = None) -> tuple[dict[str, Any], list[bytes], list[Region]]:
    """The file's material regions, with sampled, coloured surface points for the views. Raises Unsupported."""
    try:
        doc, binary = read_glb(data)
    except (ValueError, json.JSONDecodeError) as exc:
        raise Unsupported("not a GLB file") from exc
    views = _views(doc, binary)
    materials = doc.get("materials") or []
    if not materials:
        raise Unsupported("the model has no materials")
    rng = rng or np.random.default_rng(7)
    pieces: dict[tuple[int, int], list[_Piece]] = {}  # (material, mesh) → its primitives
    for mesh_index, world, node_name in _instances(doc):
        mesh = doc["meshes"][mesh_index]
        for prim_index, prim in enumerate(mesh.get("primitives", [])):
            attrs = prim.get("attributes", {})
            if prim.get("mode", 4) != 4 or prim.get("material", -1) not in range(len(materials)) or "POSITION" not in attrs:
                continue
            try:
                pos = _accessor(doc, views, attrs["POSITION"])[:, :3]
                idx = _accessor(doc, views, prim["indices"])[:, 0].astype(np.int64) if "indices" in prim else np.arange(len(pos))
                uv = _accessor(doc, views, attrs["TEXCOORD_0"])[:, :2] if "TEXCOORD_0" in attrs else None
            except (Unsupported, KeyError, IndexError, ValueError, TypeError):
                continue
            idx = idx[: len(idx) // 3 * 3]
            if not len(idx) or idx.max() >= len(pos) or (uv is not None and idx.max() >= len(uv)):
                continue
            world_pos = pos @ world[:3, :3].T + world[:3, 3]
            pieces.setdefault((prim["material"], mesh_index), []).append(_Piece(
                world_pos[idx].reshape(-1, 3, 3), uv[idx].reshape(-1, 3, 2) if uv is not None else None,
                (mesh_index, prim_index), mesh.get("name") or node_name,
            ))
    if not pieces:
        raise Unsupported("the model has no triangles with materials")
    if len(pieces) > MAX_REGIONS:  # too many meshes to ask about one by one: one region per material instead
        merged: dict[tuple[int, int], list[_Piece]] = {}
        for (m, _), group in pieces.items():
            merged.setdefault((m, -1), []).extend(group)
        pieces = merged

    every = np.concatenate([p.tris.reshape(-1, 3) for group in pieces.values() for p in group])
    low, high = every.min(axis=0), every.max(axis=0)
    area = {k: [0.5 * np.linalg.norm(np.cross(p.tris[:, 1] - p.tris[:, 0], p.tris[:, 2] - p.tris[:, 0]), axis=1) for p in g] for k, g in pieces.items()}
    total = float(sum(a.sum() for g in area.values() for a in g)) or 1.0
    textures: dict[int, np.ndarray | None] = {}

    found: list[Region] = []
    for key, group in pieces.items():
        m = key[0]
        mat = materials[m]
        pbr = mat.get("pbrMetallicRoughness", {})
        factor = np.array((pbr.get("baseColorFactor") or [1, 1, 1, 1])[:3], dtype=np.float64)
        if m not in textures:
            textures[m] = _texture(doc, views, pbr.get("baseColorTexture"))
        texture = textures[m]
        tris = np.concatenate([p.tris for p in group])
        uvs = np.concatenate([p.uvs for p in group]) if texture is not None and all(p.uvs is not None for p in group) else None
        a = np.concatenate(area[key])
        share = float(a.sum()) / total
        n = max(80, int(POINTS * share))
        pick = rng.choice(len(tris), size=n, p=a / a.sum()) if a.sum() > 0 else rng.integers(0, len(tris), size=n)
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        w = np.stack([1 - u - v, u, v], axis=1)[:, :, None]
        chosen = tris[pick]
        points = (chosen * w).sum(axis=1)
        normals = np.cross(chosen[:, 1] - chosen[:, 0], chosen[:, 2] - chosen[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        tint = _linear_to_srgb(factor)  # the factor multiplies in linear space; close enough for a picture
        if uvs is not None and texture is not None:
            at = (uvs[pick] * w).sum(axis=1) % 1.0
            h, wd = texture.shape[:2]
            colors = texture[(at[:, 1] * (h - 1)).astype(int), (at[:, 0] * (wd - 1)).astype(int)] * tint
        else:
            colors = np.tile(tint * 255, (n, 1))
        centre_y = float((points[:, 1].mean() - low[1]) / max(high[1] - low[1], 1e-9))
        labels = list(dict.fromkeys(p.label for p in group if p.label))
        found.append(Region(
            id="", material=m, name=mat.get("name", ""), meshes=labels[:6],
            color=tuple(float(c) for c in _srgb_to_linear(colors.mean(axis=0) / 255)),  # type: ignore[arg-type]
            textured="baseColorTexture" in pbr, pbr_maps="metallicRoughnessTexture" in pbr,
            unlit="KHR_materials_unlit" in (mat.get("extensions") or {}),
            prims=[p.ref for p in group], share=share,
            band="top" if centre_y > 0.66 else "bottom" if centre_y < 0.33 else "middle",
            points=points, normals=normals, colors=np.clip(colors, 0, 255),
        ))
    found.sort(key=lambda r: -r.share)
    for i, r in enumerate(found, 1):
        r.id = f"R{i}"
    return doc, views, found


# ── Two views for the vision model ───────────────────────────────────────────

MAP_COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
              (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
              (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0), (100, 100, 255), (255, 100, 100)]


def _view(found: list[Region], yaw: float, colours: list[tuple[int, int, int]] | None, shade: bool) -> tuple[Image.Image, dict[str, tuple[int, int]]]:
    """`colours`: one flat colour per region (the map); None: each point's own colour (the look)."""
    pts = np.concatenate([r.points for r in found])
    nrm = np.concatenate([r.normals for r in found])
    owner = np.concatenate([np.full(len(r.points), i) for i, r in enumerate(found)])
    pitch = math.radians(20)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rot = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.eye(3)
    rot = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ rot
    cam = (pts - pts.mean(axis=0)) @ rot.T
    n_cam = nrm @ rot.T
    span = max(np.ptp(cam[:, 0]), np.ptp(cam[:, 1]), 1e-9)
    scale = (VIEW - 40) / span
    px = np.clip(((cam[:, 0] - cam[:, 0].min()) * scale + (VIEW - np.ptp(cam[:, 0]) * scale) / 2).astype(int), 0, VIEW - 3)
    py = np.clip((VIEW - 1 - ((cam[:, 1] - cam[:, 1].min()) * scale + (VIEW - np.ptp(cam[:, 1]) * scale) / 2)).astype(int), 0, VIEW - 3)
    order = np.argsort(cam[:, 2])  # far → near: the nearest point is written last and wins
    base = np.array(colours, dtype=np.float64)[owner] if colours is not None else np.concatenate([r.colors for r in found])
    if shade:
        light = np.array([0.3, 0.6, 0.75])
        lambert = np.abs(n_cam @ (light / np.linalg.norm(light)))  # two-sided: sampled faces may point either way
        base = base * (0.35 + 0.65 * lambert)[:, None]
    image = np.full((VIEW, VIEW, 3), 245, dtype=np.uint8)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)):  # 3×3 splats close the gaps
        image[py[order] + dy, px[order] + dx] = np.clip(base[order], 0, 255).astype(np.uint8)
    seen = np.full((VIEW, VIEW), -1)
    seen[py[order], px[order]] = owner[order]
    centres: dict[str, tuple[int, int]] = {}
    for i, r in enumerate(found):
        ys, xs = np.nonzero(seen == i)
        if len(xs) >= 25:
            centres[r.id] = (int(np.median(xs)), int(np.median(ys)))
    return Image.fromarray(image), centres


def render(found: list[Region]) -> tuple[bytes, bytes]:
    """PNGs of (look, region map), each front three-quarter and back three-quarter side by side."""
    regionmap = [MAP_COLORS[i % len(MAP_COLORS)] for i in range(len(found))]
    out = []
    for colours, shade, labels in ((None, True, False), (regionmap, False, True)):
        sheet = Image.new("RGB", (VIEW * 2, VIEW), (245, 245, 245))
        for side, yaw in enumerate((math.radians(35), math.radians(215))):
            picture, centres = _view(found, yaw, colours, shade)  # type: ignore[arg-type]
            if labels:
                draw = ImageDraw.Draw(picture)
                for rid, (x, y) in centres.items():
                    draw.ellipse((x - 13, y - 9, x + 13, y + 9), fill=(255, 255, 255), outline=(0, 0, 0))
                    draw.text((x - 9, y - 6), rid, fill=(0, 0, 0))
            sheet.paste(picture, (side * VIEW, 0))
        buffer = io.BytesIO()
        sheet.save(buffer, format="PNG")
        out.append(buffer.getvalue())
    return out[0], out[1]


# ── Asking the vision model ──────────────────────────────────────────────────


def prompt(found: list[Region], with_photo: bool) -> str:
    lines = []
    for r in found[:MAX_REGIONS]:
        hex_colour = "#%02x%02x%02x" % tuple(int(c) for c in _linear_to_srgb(np.array(r.color)) * 255)
        lines.append(
            f"- {r.id}: colour {hex_colour}, {r.share * 100:.0f}% of the surface, {r.band} of the piece"
            f"{', has its own texture' if r.textured else ''}"
            f"{f', material name “{r.name}”' if r.name else ''}{f', mesh names: {', '.join(r.meshes)}' if r.meshes else ''}"
        )
    return (
        "You see a 3D furniture model. Picture 1 shows its colours (front and back three-quarter views); picture 2 shows the same views "
        "with each material region in a flat false colour and labelled with its id"
        + (". Picture 3 is a photo of the real product" if with_photo else "")
        + ". For every region below, say what physical material it most likely is, and its finish. Judge from the pictures and the "
        "region's shape and position; names can be wrong or generic. Use \"unknown\" with a low confidence when you cannot tell. "
        "Do not invent regions.\n\nRegions:\n" + "\n".join(lines)
    )


def parse(text: str | None, ids: set[str]) -> MaterialProfile | None:
    """The model's answer, validated: unknown ids and duplicates dropped; None when it is not a profile at all."""
    if not text:
        return None
    try:
        profile = MaterialProfile.model_validate_json(text.strip().removeprefix("```json").removesuffix("```"))
    except ValidationError as exc:
        logger.warning("Material profile invalid: %s", str(exc)[:300])
        return None
    seen: set[str] = set()
    kept = []
    for r in profile.regions:
        if r.id in ids and r.id not in seen:
            seen.add(r.id)
            kept.append(r)
    return MaterialProfile(regions=kept)


# ── Writing the materials ────────────────────────────────────────────────────


def apply(doc: dict[str, Any], found: list[Region], profile: MaterialProfile) -> list[dict[str, Any]]:
    """Writes the presets into `doc`'s materials (in place); returns what was done per region, for the app and the cache.
    Unsure, unknown, unlit or unlisted regions keep their original material."""
    by_id = {r.id: r for r in found}
    used: set[str] = set()
    report: list[dict[str, Any]] = []
    for answer in profile.regions:
        region = by_id[answer.id]
        entry = {"id": answer.id, "material": answer.material, "finish": answer.finish, "confidence": round(answer.confidence, 2), "applied": False}
        report.append(entry)
        if answer.material == "unknown" or answer.confidence < MIN_CONFIDENCE or region.unlit:
            continue
        preset = PRESETS[answer.material]
        if sum(r.material == region.material for r in found) > 1:
            # Other parts share this material: this part gets its own copy (same textures), so they keep theirs.
            # Only the primitives' material index changes; their geometry is untouched.
            copy = json.loads(json.dumps(doc["materials"][region.material]))
            copy["name"] = f"{copy.get('name') or 'material'} ({region.id})"
            doc["materials"].append(copy)
            for mesh, prim in region.prims:
                doc["meshes"][mesh]["primitives"][prim]["material"] = len(doc["materials"]) - 1
            region.material = len(doc["materials"]) - 1
        material = doc["materials"][region.material]
        pbr = material.setdefault("pbrMetallicRoughness", {})
        if not region.pbr_maps:  # a metallic-roughness map of its own is kept as measured
            pbr["roughnessFactor"] = round(min(1.0, max(0.04, preset["roughness"] + FINISH_SHIFT[answer.finish])), 3)
            pbr["metallicFactor"] = preset["metalness"]
        ext = material.setdefault("extensions", {})
        if preset.get("sheen"):
            tint = [round(min(1.0, c * preset["sheen"] + 0.05), 3) for c in region.color]
            ext["KHR_materials_sheen"] = {"sheenColorFactor": tint, "sheenRoughnessFactor": 0.5}
            used.add("KHR_materials_sheen")
        if preset.get("clearcoat"):
            ext["KHR_materials_clearcoat"] = {"clearcoatFactor": preset["clearcoat"], "clearcoatRoughnessFactor": 0.1 if answer.finish == "gloss" else 0.25}
            used.add("KHR_materials_clearcoat")
        if not ext:
            material.pop("extensions")
        entry["applied"] = True
    if used:
        doc["extensionsUsed"] = sorted(set(doc.get("extensionsUsed", [])) | used)
    return report


def realistic_glb(doc: dict[str, Any], views: list[bytes]) -> bytes:
    """The file again with its new materials: the same buffer views (geometry, textures) byte for byte."""
    return write_glb(doc, views)


def cache_key(data: bytes) -> str:
    return "realistic:" + hashlib.sha256(data + f"|v{PROFILE_VERSION}".encode()).hexdigest()


# ── The whole step, with its cache ───────────────────────────────────────────

Ask = Callable[[list[dict[str, Any]], dict[str, Any]], Awaitable[str | None]]


async def make(
    data: bytes, photo: bytes | None, ask: Ask,
    cache_get: Callable[[str], Awaitable[dict[str, Any] | None]],
    cache_put: Callable[[str, dict[str, Any]], Awaitable[None]],
    upload: Callable[[str, bytes], str],
) -> dict[str, Any]:
    """{"status": "ready", "glb_url", "regions"} | {"status": "unchanged" | "unsupported" | "failed", "reason"}.
    Ready and unchanged results are cached; failures are not (the next open tries again)."""
    key = cache_key(data)
    hit = await cache_get(key)
    if hit and hit.get("status") in ("ready", "unchanged"):
        return {**hit, "cached": True}
    try:
        doc, views, found = await asyncio.to_thread(regions, data)  # CPU work: off the event loop
    except Unsupported as exc:
        return {"status": "unsupported", "reason": str(exc)}
    asked = [r for r in found if not r.unlit][:MAX_REGIONS]
    if not asked:
        return {"status": "unsupported", "reason": "the model's materials are unlit"}
    look, regionmap = await asyncio.to_thread(render, found)

    def picture(png: bytes, mime: str = "image/png") -> dict[str, Any]:
        return {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(png).decode()}"}

    content = [{"type": "input_text", "text": prompt(asked, photo is not None)}, picture(look), picture(regionmap)]
    if photo:
        content.append(picture(photo, "image/jpeg" if photo[:2] == b"\xff\xd8" else "image/png"))
    profile = parse(await ask(content, MaterialProfile.model_json_schema()), {r.id for r in asked})
    if profile is None:
        return {"status": "failed", "reason": "the vision model gave no usable answer"}
    report = apply(doc, found, profile)
    if not any(r["applied"] for r in report):
        result = {"status": "unchanged", "reason": "no part could be told apart with enough confidence", "regions": report}
    else:
        glb = await asyncio.to_thread(realistic_glb, doc, views)
        url = await asyncio.to_thread(upload, f"realistic/{key.split(':')[1]}.glb", glb)
        result = {"status": "ready", "glb_url": url, "regions": report}
    await cache_put(key, result)
    return result
