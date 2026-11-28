"""Hybrid furniture reconstruction: the visual model built from the photo's structured spec, not from a mesh generator.

  original photo ─ vision model (furniture_spec: 1 call, cached) ─▶ components (box in the photo, size, material,
                                                                    decoration, complexity, relations, motion)
      ├─ structure   structural.build → carpenter boards: flat, straight, closed solids, one board thickness
      ├─ appearance  each part textured with ITS OWN region of the original photo (perspective as photographed),
      │              the parts mounted on it (handles, knobs) painted out so they are not baked into the panel;
      │              glass and mirrors keep their material, never a photo of their reflections
      ├─ decoration  fluted fronts as real ribs; other regular decoration (grooves, raised / recessed / arched
      │              panels) shown by the photo's own pixels on the flat panel; complex (organic) parts reported
      ├─ movement    animation.configure from the spec's motion, with each door's handles attached by relation AND
      │              geometry (never the nearest box); estimated interior marked as estimated
      └─ checks      closed, flat, thickness, floating, handle ownership: structured errors / warnings
  → one GLB: one named node and mesh per physical part, partId in its extras, metres, y up, +z front.

Modes (config GENERATION_MODE or the vendor's choice): "deterministic" and "hybrid" use this engine; "hybrid" is
meant to send complex parts to a mesh provider one by one, which is not implemented yet: such parts are built as a
photo-textured block and listed in `limits`. "meshy" is the existing generator pipeline (Meshy / GPT Astra / Sol).
A photo whose furniture is mostly complex or upholstered is not rebuilt from boards: the caller falls back to the
generator and says why.

ponytail: boxes only (no CadQuery / boolean engine installed): grooves and arches are texture, not cut geometry.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any

import numpy as np

from app.services import animation, carpenter, photo_tools, solids, structural
from app.services.generation_prompt import furniture_family
from app.services.glb_size import write_glb

logger = logging.getLogger(__name__)

MODES = ("hybrid", "deterministic", "meshy")
TEXTURE_MAX = 512  # px per part texture: the photo region of one part is rarely larger
COMPLEX_SHARE_MAX = 0.6  # complex parts covering more of the front than this: one whole-model generation is cheaper
COMPLEX_MAX = 4  # distinct complex components sent to the mesh provider (each one a paid task); more: whole model
FLUTE_PITCH = 0.03  # m between flutes
TOUCH = 0.02  # m: a part this close to another is held by it (a reveal read from a photo is about this wide)
HANDLE_WORDS = ("handle", "knob", "pull", "grip")
PANE_CLASSES = ("glass", "mirror")


def _words(c: dict[str, Any]) -> str:
    return f"{c.get('id') or ''} {c.get('name') or ''}".lower().replace("_", " ")


def _area(box: list[float] | None) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) if box and len(box) == 4 else 0.0


def is_complex(c: dict[str, Any]) -> bool:
    """A part whose SHAPE boards cannot make: organic, carved, sculpted or upholstered. It goes to the mesh provider."""
    return c.get("complexity") == "complex" or c.get("decoration") in ("carved", "upholstered") or str(c.get("materialClass") or "") in ("fabric", "leather")


def group_key(c: dict[str, Any], size: np.ndarray) -> str:
    """Repeated components (four identical carved feet, two matching cushions) share one generated mesh: the same
    decoration, the same name without its side or number, and the same size to the centimetre."""
    stem = re.sub(r"\b(left|right|front|back|rear|top|bottom|upper|lower|inner|outer|\d+)\b|[_\d]+", " ", str(c.get("name") or c.get("id")).lower())
    return f"{' '.join(stem.split())}|{c.get('decoration')}|{c.get('materialClass')}|" + "x".join(str(int(round(v * 100))) for v in sorted(size)[::-1])


def unsuitable(analysis: dict[str, Any] | None) -> str | None:
    """Why this furniture should go to the whole-model generator instead (None: rebuild it here). Complex parts
    covering most of it (a sofa that is all cushion), or too many distinct ones to generate one by one."""
    comps = [c for c in (analysis or {}).get("components") or [] if c.get("box")]
    if not comps:
        return "The photo reading found no located components."
    # The whole body (a part covering most of the object's box) says nothing about complexity; the largest part of a
    # table, its top, does: it is a board, and its legs are what may be sculpted.
    parts = [c for c in comps if _area(c.get("box")) < 0.64] or comps
    total = sum(_area(c["box"]) for c in parts) or 1.0
    soft = sum(_area(c["box"]) for c in parts if is_complex(c))
    if soft / total > COMPLEX_SHARE_MAX:
        return f"{soft / total:.0%} of its visible parts are upholstered, carved or sculpted: one whole-model generation suits it better."
    distinct = {group_key(c, np.array([c["box"][2] - c["box"][0], c["box"][3] - c["box"][1], 0])) for c in parts if is_complex(c)}
    if len(distinct) > COMPLEX_MAX:
        return f"{len(distinct)} different sculpted or upholstered parts: more than {COMPLEX_MAX} separate generations would cost more than one."
    return None


# ---- appearance: each part's own pixels from the photo ----


def _photo(photo: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(photo)).convert("RGB")


def _rect(obj: list[float], box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """A component's box (fractions of the object's box) in photo pixels."""
    W, H = size
    ox0, oy0, ox1, oy1 = obj
    x0, y0, x1, y1 = box
    r = (int((ox0 + x0 * (ox1 - ox0)) * W), int((oy0 + y0 * (oy1 - oy0)) * H), int((ox0 + x1 * (ox1 - ox0)) * W), int((oy0 + y1 * (oy1 - oy0)) * H))
    return r if r[2] - r[0] >= 4 and r[3] - r[1] >= 4 else None


FURNITURE_SHARE_MIN = 0.6  # a part region less furniture than this (a side seen edge-on, the room around it): its colour only


def part_texture(img, obj: list[float], box: list[float], mounted: list[list[float]], silhouette: np.ndarray | None = None) -> tuple[bytes | None, list[float]] | None:
    """(JPEG of the part's region with what is mounted on it painted out, its median colour 0..1) or None. Pixels outside
    the furniture's silhouette (the wall behind a thin side, the ceiling over the top) are never used: they are filled
    from the part's own pixels, and a region that is mostly not furniture gives its colour only (texture None)."""
    import cv2
    from PIL import Image

    from app.services import photo_tools

    rect = _rect(obj, box, img.size)
    if rect is None:
        return None
    crop, warped = photo_tools.rectified(np.asarray(img), rect)  # straightened when the photo shows it at an angle
    crop = crop.copy()
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if silhouette is not None and not warped:
        outside = silhouette[rect[1]:rect[3], rect[0]:rect[2]] == 0
        if outside.shape == (h, w) and outside.any():
            inside = ~outside
            if inside.mean() < FURNITURE_SHARE_MIN:
                colour = (np.median(crop[inside].reshape(-1, 3), axis=0) / 255.0).tolist() if inside.any() else None
                return (None, colour) if colour else None
            mask[outside] = 255  # filled from the furniture's own pixels below
    bw, bh = box[2] - box[0], box[3] - box[1]
    for m in mounted:  # handles, knobs, mounts: fractions of the object box → fractions of this part
        mx0, my0 = (m[0] - box[0]) / bw, (m[1] - box[1]) / bh
        mx1, my1 = (m[2] - box[0]) / bw, (m[3] - box[1]) / bh
        pad_x, pad_y = 0.15 * (mx1 - mx0) + 2 / w, 0.15 * (my1 - my0) + 2 / h
        x0, y0 = max(0, int((mx0 - pad_x) * w)), max(0, int((my0 - pad_y) * h))
        x1, y1 = min(w, int((mx1 + pad_x) * w) + 1), min(h, int((my1 + pad_y) * h) + 1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    if mask.any() and mask.mean() < 0.5 * 255:
        crop = cv2.inpaint(crop, mask, 5, cv2.INPAINT_TELEA)
    elif mask.mean() >= 0.5 * 255:
        return None, (np.median(crop[mask == 0].reshape(-1, 3), axis=0) / 255.0).tolist() if (mask == 0).any() else None
    colour = (np.median(crop.reshape(-1, 3), axis=0) / 255.0).tolist()
    out = Image.fromarray(crop)
    out.thumbnail((TEXTURE_MAX, TEXTURE_MAX))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=88)
    return buf.getvalue(), colour


# ---- geometry ----


def box_mesh(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A closed box with flat faces: 24 vertices (4 per face, own normals), UVs projected from the front (x, y)."""
    faces = [((0, 1), 2, 1), ((0, 1), 2, -1), ((2, 1), 0, 1), ((2, 1), 0, -1), ((0, 2), 1, 1), ((0, 2), 1, -1)]
    pos, nrm, idx = [], [], []
    for (a, b), axis, sign in faces:
        base = len(pos)
        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p = np.empty(3)
            p[axis] = hi[axis] if sign > 0 else lo[axis]
            p[a] = lo[a] + u * (hi[a] - lo[a])
            p[b] = lo[b] + v * (hi[b] - lo[b])
            pos.append(p)
            n = np.zeros(3)
            n[axis] = sign
            nrm.append(n)
        quad = [base, base + 1, base + 2, base, base + 2, base + 3]
        # Counter-clockwise seen from outside.
        e1, e2 = pos[base + 1] - pos[base], pos[base + 2] - pos[base]
        if np.dot(np.cross(e1, e2), nrm[base]) < 0:
            quad = [base, base + 2, base + 1, base, base + 3, base + 2]
        idx += quad
    pos, nrm = np.array(pos), np.array(nrm)
    return pos, nrm, _uv(pos, lo, hi), np.array(idx)


def _uv(pos: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.maximum(hi - lo, 1e-9)
    return np.column_stack([(pos[:, 0] - lo[0]) / span[0], 1 - (pos[:, 1] - lo[1]) / span[1]])


def flutes(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Vertical half-round ribs standing on the front face (+z) of a panel: a fluted front as geometry."""
    width = hi[0] - lo[0]
    count = int(width // FLUTE_PITCH)
    if count < 3:
        return None
    r = 0.45 * width / count
    angles = np.linspace(0, np.pi, 9)  # the front half of the round, left edge to right edge
    pos, nrm, idx = [], [], []
    for i in range(count):
        cx = lo[0] + (i + 0.5) * width / count
        base = len(pos)
        for a in angles:
            n = np.array([-np.cos(a), 0.0, np.sin(a)])
            for y in (lo[1], hi[1]):
                pos.append([cx + r * n[0], y, hi[2] + r * n[2]])
                nrm.append(n)
        for k in range(len(angles) - 1):
            a0, a1, b0, b1 = base + 2 * k, base + 2 * k + 1, base + 2 * k + 2, base + 2 * k + 3
            idx += [a0, b0, b1, a0, b1, a1]
    p = np.array(pos)
    return p, np.array(nrm), _uv(p, lo, hi), np.array(idx)


def _merge(*meshes):
    pos, nrm, uv, idx, off = [], [], [], [], 0
    for m in meshes:
        if m is None:
            continue
        pos.append(m[0]), nrm.append(m[1]), uv.append(m[2]), idx.append(m[3] + off)
        off += len(m[0])
    return np.concatenate(pos), np.concatenate(nrm), np.concatenate(uv), np.concatenate(idx)


def write(parts: list[dict[str, Any]]) -> bytes:
    """One node and mesh per part; its texture (the photo region) or colour; partId and how it was made in extras."""
    views: list[bytes] = []
    accessors, meshes, nodes, materials, images, textures = [], [], [], [], [], []

    def view(blob: bytes, target: int | None = None) -> int:
        views.append(blob)
        return len(views) - 1

    for part in parts:
        pos, nrm, uv, idx = part["mesh"]
        a0 = len(accessors)
        for data, kind in ((pos, "VEC3"), (nrm, "VEC3"), (uv, "VEC2")):
            arr = np.ascontiguousarray(data, dtype=np.float32)
            acc = {"bufferView": view(arr.tobytes()), "componentType": 5126, "count": len(arr), "type": kind}
            if kind == "VEC3" and data is pos:
                acc.update(min=arr.min(0).tolist(), max=arr.max(0).tolist())
            accessors.append(acc)
        accessors.append({"bufferView": view(np.ascontiguousarray(idx, dtype=np.uint32).tobytes()), "componentType": 5125, "count": int(idx.size), "type": "SCALAR"})
        pbr: dict[str, Any] = {"baseColorFactor": [*part["colour"], part.get("alpha", 1.0)], "metallicFactor": part["metallic"], "roughnessFactor": part["roughness"]}
        if part.get("texture"):
            images.append({"bufferView": view(part["texture"]), "mimeType": "image/jpeg"})
            textures.append({"source": len(images) - 1, "sampler": 0})
            pbr = {**pbr, "baseColorFactor": [1, 1, 1, 1], "baseColorTexture": {"index": len(textures) - 1}}
        material = {"name": part["name"], "pbrMetallicRoughness": pbr, **({"alphaMode": "BLEND"} if part.get("alpha", 1.0) < 1 else {})}
        materials.append(material)
        meshes.append({"name": part["name"], "primitives": [{"attributes": {"POSITION": a0, "NORMAL": a0 + 1, "TEXCOORD_0": a0 + 2}, "indices": a0 + 3, "material": len(materials) - 1}]})
        nodes.append({"name": part["name"], "mesh": len(meshes) - 1, "extras": part["extras"]})
    doc: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "sha2ty hybrid"}, "scene": 0,
        "scenes": [{"nodes": [len(nodes)]}],
        "nodes": nodes + [{"name": "Furniture", "children": list(range(len(nodes)))}],
        "meshes": meshes, "materials": materials, "accessors": accessors, "buffers": [{}],
        "bufferViews": [{} for _ in views],
        **({"images": images, "textures": textures, "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]} if images else {}),
    }
    return write_glb(doc, views)


# ---- the build ----


def _finish(spec: dict[str, Any] | None, cls: str) -> tuple[float, float]:
    """(metallic, roughness) from the part's class and finish words."""
    finish = f"{(spec or {}).get('finish') or ''}".lower()
    rough = 0.25 if "gloss" in finish or "lacquer" in finish else 0.45 if "satin" in finish else 0.7
    if cls == "metal":
        return 1.0, 0.35 if "brush" in finish else 0.25
    if cls == "mirror":
        return 1.0, 0.03
    if cls == "glass":
        return 0.0, 0.05
    return 0.0, rough


def _hex(value: str | None) -> list[float] | None:
    if not value or not re.fullmatch(r"#?[0-9a-fA-F]{6}", value.strip()):
        return None
    v = value.strip().lstrip("#")
    return [int(v[i: i + 2], 16) / 255 for i in (0, 2, 4)]


def _inside(inner: list[float], outer: list[float], share: float = 0.8) -> bool:
    w = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return w * h >= share * max(_area(inner), 1e-9)


def with_vendor_size(details: dict[str, Any], width: float | None, height: float | None, depth: float | None) -> dict[str, Any]:
    """The vendor's measurements over the photo's estimate (each one given wins; the rest stay estimated)."""
    given = {k: v for k, v in (("width_m", width), ("height_m", height), ("depth_m", depth)) if v}
    if not given:
        return details
    complete = len(given) == 3
    return {**details, **given, "dimension_source": "vendor" if complete else details.get("dimension_source"),
            "vendorSize": sorted(given)}


def build(photo: bytes, spec_doc: dict[str, Any] | None, category: str | None, kind: str | None, mode: str = "hybrid",
          size: tuple[float | None, float | None, float | None] = (None, None, None)) -> tuple[bytes | None, dict[str, Any]]:
    """(the GLB, its part document) or (None, {"reason": why not}) when this photo cannot be rebuilt here. `size`:
    the vendor's width, height and depth in metres, which win over the photo's estimate."""
    details = with_vendor_size((spec_doc or {}).get("details") or {}, *size)
    analysis = (spec_doc or {}).get("analysis") or {}
    why = unsuitable(analysis)
    if why:
        return None, {"reason": why}
    _, report = structural.build(details, analysis, category, kind)
    if not report.get("built"):
        return None, {"reason": " ".join(report.get("uncertain") or []) or "The structure could not be rebuilt from the photo."}
    spec_by_id = {c["id"]: c for c in analysis.get("components") or []}
    obj = details.get("bbox") if isinstance(details.get("bbox"), list) and len(details.get("bbox")) == 4 else [0.0, 0.0, 1.0, 1.0]
    img = _photo(photo)
    materials = report.get("materials") or {}
    # Real positions: carpenter's centimetres from the envelope's low corner → metres, centred on x and z, floor at y 0.
    raw = report["components"]
    lows = np.array([[p["positionCm"][k] - p["sizeCm"][d] / 2 for k, d in (("x", "width"), ("y", "height"), ("z", "depth"))] for p in raw]) / 100
    highs = lows + np.array([[p["sizeCm"]["width"], p["sizeCm"]["height"], p["sizeCm"]["depth"]] for p in raw]) / 100
    lo_all, hi_all = lows.min(0), highs.max(0)
    size_m = report.get("sizeM") or {}
    if details.get("dimension_source") == "vendor" and size_m:
        # The vendor's measurements are the model's: the parts read from the photo (which stop a little short of the
        # object's edges) are stretched along each axis to fill them exactly.
        want = np.array([size_m.get("width") or 0, size_m.get("height") or 0, size_m.get("depth") or 0], float)
        span = np.maximum(hi_all - lo_all, 1e-9)
        k = np.where(want > 0, want / span, 1.0)
        lows, highs = lo_all + (lows - lo_all) * k, lo_all + (highs - lo_all) * k
        lo_all, hi_all = lows.min(0), highs.max(0)
    shift = np.array([-(lo_all[0] + hi_all[0]) / 2, -lo_all[1], -(lo_all[2] + hi_all[2]) / 2])
    built, limits, decorations = [], [], []
    rgb = np.asarray(img)
    silhouette = photo_tools.silhouette(rgb, obj)
    bands = line_bands(spec_by_id, rgb, obj)
    for p, lo, hi in zip(raw, lows + shift, highs + shift):
        spec = spec_by_id.get(p.get("specId") or "")
        if spec is not None and is_complex(spec) and p.get("source") != "inferred":
            continue  # its shape is generated on its own (pending_components), never a block in its place
        mspec = materials.get(p.get("specId") or "") or {}
        cls = str(mspec.get("class") or (spec or {}).get("materialClass") or p.get("materialCategory") or "wood")
        metallic, rough = _finish(spec, cls)
        colour = _hex((spec or {}).get("color")) or _hex(mspec.get("color")) or [0.72, 0.58, 0.4]
        texture = None
        visible = bool(spec and spec.get("box") and spec.get("visibility") != "inferred" and p.get("source") != "inferred")
        own = visible and spec is not None and not (_area(spec["box"]) >= 0.64 and p.get("source") == "inferred")
        if own and cls not in PANE_CLASSES:
            mounted = [o["box"] for o in spec_by_id.values() if o is not spec and o.get("box") and _inside(o["box"], spec["box"])
                       and _area(o["box"]) < 0.5 * _area(spec["box"])]
            got = part_texture(img, obj, spec["box"], mounted, silhouette)
            if got:
                texture, colour = got[0], got[1] or colour
        elif spec is not None and spec.get("box") and cls not in PANE_CLASSES:
            rect = _rect(obj, spec["box"], img.size)  # an inferred panel of a seen body: the body's own colour
            if rect:
                colour = (np.median(np.asarray(img.crop(rect)).reshape(-1, 3), axis=0) / 255.0).tolist()
        deco = (spec or {}).get("decoration") or "none"
        if deco in solids.DECORATIONS and (hi - lo).min() < 0.1:
            rect = _rect(obj, spec["box"], img.size) if spec and spec.get("box") else None
            band = None
            if deco in ("grooved", "fluted"):
                # Lines are built only where the photo shows them: the band of the part (e.g. the lower third of a door)
                # with a regular line pattern; a matching sibling (the same decoration and size) shares what was seen
                # on it. None anywhere: the reading said so, but the photo does not; not built.
                band = bands.get(p.get("specId"))
            if deco in ("grooved", "fluted") and band is None:
                mesh = box_mesh(lo, hi)
                decorations.append({"component": p["id"], "decoration": deco, "method": "not built",
                                    "reason": "The photo shows no regular lines on this part."})
            else:
                part_lo, part_hi = lo.copy(), hi.copy()
                if band is not None:  # the band's height range (photo rows run top → down)
                    part_lo[1], part_hi[1] = hi[1] - band["to"] * (hi[1] - lo[1]), hi[1] - band["from"] * (hi[1] - lo[1])
                solid, made = solids.decorated(lo, hi, deco, band, region=(part_lo, part_hi) if band is not None else None)
                mesh = solids.arrays(solid, lo, hi)
                decorations.append({"component": p["id"], **made, **({"band": [round(band["from"], 2), round(band["to"], 2)]} if band else {})})
        elif spec is not None and any(w in _words(spec) for w in HANDLE_WORDS) and re.search(r"\b(round|rounded|circular|semicircular|disc|knob|cylindrical)\b", f"{_words(spec)} {spec.get('shape') or ''}".lower()):
            # A round knob or pull: a disc / cylinder standing on its door, its face the photo's size.
            r = 0.5 * min(hi[0] - lo[0], hi[1] - lo[1])
            centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
            disc = solids.mf.Manifold.cylinder(hi[2] - lo[2], r, r, solids.SEGMENTS).translate(centre)
            mesh = solids.arrays(disc, lo, hi)
            decorations.append({"component": p["id"], "decoration": "round", "method": "geometry"})  # shaped: never flattened
        else:
            mesh = box_mesh(lo, hi)
            if deco not in ("none", "upholstered", "carved"):
                decorations.append({"component": p["id"], "decoration": deco, "method": "not built", "reason": "Not a decoration the geometry engine makes."})
        estimated = p.get("source") == "inferred"
        built.append({
            "name": p["name"], "id": p["id"], "lo": lo, "hi": hi, "mesh": mesh, "texture": texture, "colour": [float(c) for c in colour],
            "alpha": 0.35 if cls == "glass" else 1.0, "metallic": metallic, "roughness": rough, "spec": spec, "cls": cls, "estimated": estimated,
            "extras": {"partId": p["id"], "role": p.get("role"), "source": p.get("source"), "generation": "procedural",
                       **({"specId": p["specId"]} if p.get("specId") else {}), **({"estimated": True} if estimated else {})},
        })
    glb = write(built)
    family = furniture_family(category, kind)
    pending = pending_components(analysis, details, rgb, obj)
    components = [{
        "id": b["id"], "name": b["name"], "mesh": i, "node": i, "nameAr": (b["spec"] or {}).get("nameAr") or None, "editable": True,
        "box": [[round(float(v), 4) for v in b["lo"]], [round(float(v), 4) for v in b["hi"]]],
        **({"analysisId": b["spec"]["id"]} if b["spec"] and not b["estimated"] else {}),
        **({"estimated": True, "naming": "estimated"} if b["estimated"] else {}),
        "material": {"class": b["cls"], "color": "#%02X%02X%02X" % tuple(int(255 * c) for c in b["colour"]), "fromPhoto": bool(b["texture"]),
                     "confirmed": False},
        "generation": "procedural",
    } for i, b in enumerate(built)]
    doc: dict[str, Any] = {
        "version": 1, "separated": True, "method": mode, "modelType": family, "components": components, "analysis": analysis,
        # The reading itself (sizes, box in the photo): a rebuild at confirmed sizes needs no new vision call.
        "specDetails": {k: details.get(k) for k in ("bbox", "width_m", "height_m", "depth_m", "dimension_source", "proportions", "object_type") if k in details},
        "sizeM": report.get("sizeM"), "scaleKnown": report.get("scaleKnown", False),
        "sizeSource": "vendor" if details.get("dimension_source") == "vendor" else "photo" if report.get("scaleKnown") else "estimated",
        "hybrid": {"mode": mode, "decorations": decorations, "excluded": report.get("excluded") or [],
                   "note": "Sizes are estimated from the photo unless the vendor confirmed them; inferred parts are marked estimated."},
    }
    for c in pending:
        movable = {x["id"] for x in components if ((spec_by_id.get(x.get("analysisId") or "") or {}).get("motion") or {}).get("type") not in (None, "none")}
        c["attachedTo"] = host_of(c, components, movable)
    if pending:
        doc["pendingComponents"] = [{k: v for k, v in c.items() if k != "image"} for c in pending]
        doc["hybrid"]["complex"] = [c["id"] for c in pending]
    if limits:
        doc["limits"] = " ".join(limits)
    glb, doc = animation.configure(glb, doc)
    doc = attach(doc, spec_by_id)
    doc["_pendingImages"] = {c["group"]: c["image"] for c in pending}  # popped by the caller: never stored or sent
    doc["validation"] = validate(doc, family)
    logger.info("Hybrid %s: %d parts, %d textured from the photo, %d animations, %d errors", family, len(built),
                sum(bool(b["texture"]) for b in built), len((doc.get("animation") or {}).get("animations") or []), len(doc["validation"]["errors"]))
    return glb, doc


def line_bands(spec_by_id: dict[str, dict[str, Any]], rgb: np.ndarray, obj: list[float]) -> dict[str, dict[str, Any]]:
    """Where the photo shows the flutes / grooves the reading names, per component; a component whose own crop shows
    none takes the band of a sibling with the same decoration and size (three doors fluted alike)."""
    found: dict[str, dict[str, Any]] = {}
    lined = [c for c in spec_by_id.values() if c.get("decoration") in ("fluted", "grooved") and c.get("box")]
    for c in lined:
        rect = _rect(obj, c["box"], (rgb.shape[1], rgb.shape[0]))
        band = photo_tools.pattern_band(photo_tools.rectified(rgb, rect)[0]) if rect else None
        if band:
            found[c["id"]] = band
    for c in lined:
        if c["id"] in found:
            continue
        w, h = c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]
        twin = next((found[o["id"]] for o in lined if o["id"] in found and o.get("decoration") == c.get("decoration")
                     and abs((o["box"][2] - o["box"][0]) - w) <= 0.15 * w and abs((o["box"][3] - o["box"][1]) - h) <= 0.15 * h), None)
        if twin:
            found[c["id"]] = {**twin, "fromSibling": True}
    return found


def pending_components(analysis: dict[str, Any], details: dict[str, Any], rgb: np.ndarray, obj: list[float]) -> list[dict[str, Any]]:
    """The complex parts, each with its box in the model, the structural part it is attached to, and its cutout from
    the original photo (background transparent, pixels untouched) for the mesh provider. Repeated ones share a group."""
    dims, _, _ = structural.size(details)
    if dims is None:
        return []
    W, H, D = dims
    mask = photo_tools.silhouette(rgb, obj)
    out = []
    for c in analysis.get("components") or []:
        if not is_complex(c) or not c.get("box") or c.get("visibility") == "inferred":
            continue
        box3 = structural._box3(c, W, H, D)
        rect = _rect(obj, c["box"], (rgb.shape[1], rgb.shape[0]))
        if box3 is None or rect is None:
            continue
        lo, hi = np.array(box3[0]), np.array(box3[1])  # structural._box3 is already centred on x and z, floor at y 0
        out.append({"id": re.sub(r"[^a-z0-9]+", "_", f"{c.get('name') or c['id']}".lower()).strip("_") + "_generated", "specId": c["id"],
                    "name": c.get("name") or c["id"], "nameAr": c.get("nameAr") or None, "decoration": c.get("decoration"),
                    "box": [[round(float(v), 4) for v in lo], [round(float(v), 4) for v in hi]],
                    "relatedTo": [r.get("target") for r in c.get("relations") or [] if r.get("type") in ("attached_to", "on")],
                    "group": group_key(c, hi - lo), "status": "pending", "image": photo_tools.cutout(rgb, rect, mask)})
    return out


def _gap(a: list[list[float]], b: list[list[float]]) -> float:
    """The distance between two boxes (0 when they touch or overlap)."""
    d = np.maximum(0, np.maximum(np.subtract(a[0], b[1]), np.subtract(b[0], a[1])))
    return float(np.linalg.norm(d))


def host_of(part: dict[str, Any], components: list[dict[str, Any]], movable: set[str] = frozenset()) -> str | None:
    """The part a generated component is attached to: the one the reading relates it to, when it has one; else the
    FIXED part (never a door, drawer or handle: it would carry the ornament away) it touches or is nearest to."""
    related = [c for c in components if c.get("analysisId") in part.get("relatedTo", [])]
    if related:
        return min(related, key=lambda c: _gap(c["box"], part["box"]))["id"]
    fixed = [c for c in components if c["id"] not in movable and not any(w in _words(c) for w in HANDLE_WORDS)]
    if not fixed:
        return None
    # Touching first; among those touching, the one sharing the most of the part's outline (the top for a crest).
    def overlap(c):
        return float(np.prod(np.maximum(0, np.minimum(c["box"][1], part["box"][1]) - np.maximum(c["box"][0], part["box"][0]))[:2]))
    return min(fixed, key=lambda c: (round(_gap(c["box"], part["box"]), 3), -overlap(c)))["id"]


def _within_face(host: list[list[float]], child: list[list[float]]) -> bool:
    """`child` lies over `host`'s front (x and y overlap ≥ 80%) and in front of its back face."""
    (hx0, hy0, hz0), (hx1, hy1, hz1) = host
    (cx0, cy0, cz0), (cx1, cy1, cz1) = child
    ox, oy = max(0, min(hx1, cx1) - max(hx0, cx0)), max(0, min(hy1, cy1) - max(hy0, cy0))
    return ox >= 0.8 * max(cx1 - cx0, 1e-9) and oy >= 0.8 * max(cy1 - cy0, 1e-9) and cz0 >= hz0


def attach(doc: dict[str, Any], spec_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Each moving part's own mounted parts follow it: a spec relation the geometry confirms, or a small part lying on
    its face. A part is given to one mover only (the smallest face it lies on)."""
    comps = {c["id"]: c for c in doc["components"]}
    by_spec = {c.get("analysisId"): c["id"] for c in doc["components"] if c.get("analysisId")}
    anims = [a for a in (doc.get("animation") or {}).get("animations") or [] if a.get("component") in comps]
    movers = {a["component"] for a in anims}
    owner: dict[str, str] = {}
    for cid, c in comps.items():
        if cid in movers or c.get("estimated"):
            continue
        size = np.subtract(c["box"][1], c["box"][0])
        hosts = [a["component"] for a in anims if _within_face(comps[a["component"]]["box"], c["box"])
                 and np.prod(size[:2]) < 0.5 * np.prod(np.subtract(comps[a["component"]]["box"][1], comps[a["component"]]["box"][0])[:2])]
        spec = spec_by_id.get(c.get("analysisId") or "") or {}
        said = [by_spec.get(r.get("target")) for r in spec.get("relations") or [] if r.get("type") in ("attached_to", "on")]
        agreed = [h for h in hosts if h in said]
        pick = agreed or hosts
        if pick:
            owner[cid] = min(pick, key=lambda h: np.prod(np.subtract(comps[h]["box"][1], comps[h]["box"][0])[:2]))
    for a in anims:
        a["follow"] = sorted({*[f for f in a.get("follow") or [] if f not in owner or owner[f] == a["component"]],
                              *[cid for cid, h in owner.items() if h == a["component"]]})
    doc["attachments"] = owner
    return doc


def validate(doc: dict[str, Any], family: str) -> dict[str, Any]:
    """Structured checks of the built parts. Boxes are closed, flat and straight by construction; what can still be
    wrong is thickness, floating parts, overlaps, a handle on two movers or on none, and pivots off their part."""
    errors, warnings = [], []
    comps = doc["components"]
    board = carpenter.BOARD_CM / 100
    boxes = {c["id"]: (np.array(c["box"][0]), np.array(c["box"][1])) for c in comps}
    movers = {a["component"] for a in (doc.get("animation") or {}).get("animations") or []}
    for c in comps:
        lo, hi = boxes[c["id"]]
        size = hi - lo
        if (size <= 0).any():
            errors.append(f"{c['name']}: has no volume.")
        cls = (c.get("material") or {}).get("class")
        thin = float(size.min())
        if cls not in ("glass", "mirror", "metal") and not any(w in _words(c) for w in HANDLE_WORDS) and thin < 0.5 * board:
            warnings.append(f"{c['name']}: {thin * 100:.1f} cm thick, thinner than a board.")
        near = any(o != c["id"] and np.all(lo <= boxes[o][1] + TOUCH) and np.all(boxes[o][0] <= hi + TOUCH) for o in boxes)
        if not near and len(comps) > 1 and c["id"] not in movers:  # a door or drawer hangs on hinges / runners
            errors.append(f"{c['name']}: floats, touching no other part.")
    followers: dict[str, list[str]] = {}
    for a in (doc.get("animation") or {}).get("animations") or []:
        for f in a.get("follow") or []:
            followers.setdefault(f, []).append(a["component"])
        if a.get("type") == "rotation" and a.get("pivot") and a["component"] in boxes:
            lo, hi = boxes[a["component"]]
            if not np.all((np.array(a["pivot"]) >= lo - 0.01) & (np.array(a["pivot"]) <= hi + 0.01)):
                errors.append(f"{a.get('label')}: its hinge pivot is off the part.")
    for cid, hosts in followers.items():
        if len(hosts) > 1:
            errors.append(f"{cid}: moves with {len(hosts)} parts ({', '.join(hosts)}).")
    for c in comps:
        if any(w in _words(c) for w in HANDLE_WORDS) and c["id"] not in followers and any(_within_face(next(x["box"] for x in comps if x["id"] == m), c["box"]) for m in movers):
            errors.append(f"{c['name']}: lies on a moving part but does not move with it.")
    estimated = [c["name"] for c in comps if c.get("estimated")]
    if estimated:
        warnings.append(f"Estimated, not seen in the photo: {', '.join(estimated)}.")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "confidence": "measured" if doc.get("scaleKnown") else "estimated",
            "checks": ["volume", "thickness", "floating", "handle ownership", "pivots"], "furniture": family}
