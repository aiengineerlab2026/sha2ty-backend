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

from app.services import animation, assembly, carpenter, materials, photo_tools, solids, structural
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
    return pos, nrm, solids.face_uv(pos, nrm), np.array(idx)


def write(parts: list[dict[str, Any]], assemblies: list[dict[str, Any]] | None = None) -> bytes:
    """One node and mesh per physical part (partId in its extras), under its logical assembly's node: an assembly is a
    parent with no mesh of its own (a mirrored door is its frame and its mirror, never a third solid)."""
    views: list[bytes] = []
    accessors, meshes, nodes, materials, images, textures = [], [], [], [], [], []

    def view(blob: bytes) -> int:
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
        # glTF base-colour factors are LINEAR; the colours here are sRGB (as photographed): converted, or a viewer shows
        # every plain-colour part washed out. Textures stay sRGB images (the viewer decodes them as such).
        linear = [srgb_to_linear(c) for c in part["colour"]]
        pbr: dict[str, Any] = {"baseColorFactor": [*linear, part.get("alpha", 1.0)], "metallicFactor": part["metallic"], "roughnessFactor": part["roughness"]}
        if part.get("texture"):
            images.append({"bufferView": view(part["texture"]), "mimeType": "image/jpeg"})
            textures.append({"source": len(images) - 1, "sampler": 0})
            pbr = {**pbr, "baseColorFactor": [1, 1, 1, 1], "baseColorTexture": {"index": len(textures) - 1}}
        material = {"name": part["name"], "pbrMetallicRoughness": pbr, **({"alphaMode": "BLEND"} if part.get("alpha", 1.0) < 1 else {})}
        materials.append(material)
        meshes.append({"name": part["name"], "primitives": [{"attributes": {"POSITION": a0, "NORMAL": a0 + 1, "TEXCOORD_0": a0 + 2}, "indices": a0 + 3, "material": len(materials) - 1}]})
        nodes.append({"name": part["name"], "mesh": len(meshes) - 1, "extras": part["extras"]})
    groups: dict[str, list[int]] = {}
    for i, part in enumerate(parts):
        groups.setdefault(part.get("assembly") or "", []).append(i)
    root_children = list(groups.pop("", []))
    known = {a["id"]: a for a in assemblies or []}
    for asm_id, members in groups.items():
        a = known.get(asm_id, {"id": asm_id, "name": asm_id})
        nodes.append({"name": a.get("name") or asm_id, "children": members, "extras": {"assemblyId": asm_id, "logical": True, "kind": a.get("kind")}})
        root_children.append(len(nodes) - 1)
    nodes.append({"name": "Furniture", "children": root_children})
    doc: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "sha2ty hybrid assembly"}, "scene": 0,
        "scenes": [{"nodes": [len(nodes) - 1]}],
        "nodes": nodes, "meshes": meshes, "materials": materials, "accessors": accessors, "buffers": [{}],
        "bufferViews": [{} for _ in views],
        **({"images": images, "textures": textures, "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]} if images else {}),
    }
    return write_glb(doc, views)


# ---- the build ----


def _delta_e(a, b) -> float:
    import cv2

    lab = cv2.cvtColor(np.uint8([[np.clip(np.asarray(a) * 255, 0, 255), np.clip(np.asarray(b) * 255, 0, 255)]]), cv2.COLOR_RGB2LAB).astype(float)[0]
    return float(np.linalg.norm(lab[0] - lab[1]))


def srgb_to_linear(c: float) -> float:
    c = min(max(float(c), 0.0), 1.0)
    return round(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4, 5)


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
    """(the GLB, its part document) or (None, {"reason": why not}). `size`: the vendor's width, height and depth in
    metres, which win over the photo's estimate. Physical parts come from the constraint assembly (services/
    assembly.py), their shapes from the solids engine, their materials from clean samples of the photo."""
    details = with_vendor_size((spec_doc or {}).get("details") or {}, *size)
    analysis = (spec_doc or {}).get("analysis") or {}
    why = unsuitable(analysis)
    if why:
        return None, {"reason": why}
    dims, known, size_notes = structural.size(details)
    if dims is None:
        return None, {"reason": " ".join(size_notes) or "The overall size could not be read."}
    W, H, D = dims
    family = furniture_family(category, kind)
    img = _photo(photo)
    rgb = np.asarray(img)
    analysis = measure_hardware(analysis, details, rgb)
    plan = assembly.assemble(analysis, details, (W, H, D), family)
    if len(plan["parts"]) < 2:
        return None, {"reason": "Too few located parts to assemble the furniture from."}
    # The carcass parts hang under the reading's own carcass id when it has one (never a second invented carcass).
    carcass = next((a["id"] for a in plan["assemblies"] if a.get("kind") == "carcass" and a["id"] != "carcass"), None)
    if carcass:
        plan["assemblies"] = [a for a in plan["assemblies"] if a["id"] != "carcass"]
        for p in plan["parts"]:
            if p.get("assembly") == "carcass":
                p["assembly"] = carcass
    spec_by_id = {c["id"]: c for c in analysis.get("components") or []}
    silhouette = photo_tools.silhouette(rgb, details.get("bbox"))
    bands = line_bands(spec_by_id, rgb, details.get("bbox") or [0, 0, 1, 1])
    # A part the photo does not show takes the finish of the visible part of the same structure.
    # A carcass panel the photo does not show takes the finish of a visible side or top panel; never of a trim (a plinth
    # or crown often has its own finish, and its strip in the photo takes in the floor or ceiling); else the reading's
    # colour for the carcass itself (an estimate, recorded as such).
    visible_carcass = next((p["spec"] for p in plan["parts"] if p.get("assembly") in (carcass, "carcass") and p.get("spec") and not p["estimated"]
                            and p["kind"] in ("side_panel", "top_panel") and (p["spec"].get("quad") or p["spec"].get("sideQuad"))), None) \
        or spec_by_id.get(carcass or "") or None
    built, decorations, fallbacks, recesses = [], [], [], []
    for p in plan["parts"]:
        lo, hi, spec = p["lo"], p["hi"], p.get("spec")
        source_spec = spec_by_id.get(p.get("material_from") or "") or spec
        if p["estimated"] or source_spec is None or not (source_spec.get("quad") or source_spec.get("sideQuad") or source_spec.get("box")):
            source_spec = source_spec if (source_spec or {}).get("color") and not p["estimated"] else (visible_carcass or source_spec or spec_by_id.get(carcass or "", {}))
        cls = str((source_spec or {}).get("materialClass") or "other")
        _, rough = _finish(source_spec, cls)
        if source_spec is not None and (source_spec.get("logical") or assembly.kind_of(source_spec) == "carcass"):
            # The whole object's entry: its box is everything (mirrors, the room in them); its colour is the reading's.
            source_spec = {k: v for k, v in source_spec.items() if k not in ("box", "quad", "sideQuad")}
        mat = materials.extract(rgb, source_spec, analysis, silhouette, rough, details.get("bbox"))
        if p["kind"] in ("mirror", "glass"):
            mat = materials.extract(rgb, {"kind": p["kind"]}, analysis, None, rough)
        said = _hex((source_spec or {}).get("color"))
        if said and mat["source"] == "colour sample" and (source_spec or {}).get("sideQuad") and not (source_spec or {}).get("quad") \
                and _delta_e(mat["color"], said) > 15:
            # A side seen edge-on, in its own shadow and the room's coloured light: its pixels are not its finish. The
            # reading's colour for it is used, as an estimate.
            mat = {**mat, "color": said, "source": "reading colour", "confidence": 0.5, "reason": "its visible face is in shadow in the photo"}
        if mat["source"] == "reading colour":
            fallbacks.append({"component": p["id"], "name": p["name"], "reason": mat.get("reason") or "no clean surface in the photo", "used": "reading colour"})
        # geometry: the part's semantic shape
        if p["geometry"] == "hardware":
            hw = p["hardware"]
            if hw["type"] == "bar":
                solid = solids.bar_handle(lo, hi, hw["orientation"], hw["diameter"], hw["profile"], hw["mounts"])
            elif hw["type"] == "knob":
                solid = solids.knob(lo, hi)
            else:
                solid = None  # a recess is cut into its door below, not a part of its own
            measured = (spec or {}).get("measured") or {}
            if measured.get("colour") and mat["source"] != "colour sample":
                mat = {**mat, "color": measured["colour"], "source": "colour sample", "confidence": measured.get("confidence", 0.6)}
                fallbacks = [f for f in fallbacks if f["component"] != p["id"]]
            if cls == "metal" or (spec or {}).get("materialClass") == "metal":
                finish = str((spec or {}).get("finish") or "").lower()
                mat = {**mat, "metallic": 0.85, "roughness": 0.25 if "polish" in finish or "chrome" in finish else 0.4}
            if solid is None:
                recesses.append(p)
                continue
            decorations.append({"component": p["id"], "decoration": f"{hw['type']} handle", "method": "geometry", "orientation": hw["orientation"],
                                "lengthM": hw["length"], "estimated": hw["estimated"]})
        elif p["geometry"] == "frame" and p.get("opening"):
            solid = solids.frame(lo, hi, p["opening"])
            decorations.append({"component": p["id"], "decoration": "frame with opening", "method": "geometry"})
        elif p["geometry"] == "pane":
            solid = solids.box(lo, hi)
        elif p["kind"] in ("knob", "handle") and re.search(r"\b(round|rounded|circular|semicircular|disc|knob|cylindrical)\b", f"{_words(spec or {})} {(spec or {}).get('shape') or ''}".lower()):
            r = 0.5 * min(hi[0] - lo[0], hi[1] - lo[1])
            solid = solids.mf.Manifold.cylinder(hi[2] - lo[2], r, r, solids.SEGMENTS).translate([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
            decorations.append({"component": p["id"], "decoration": "round", "method": "geometry"})
        else:
            deco = (spec or {}).get("decoration") or "none"
            band = bands.get((spec or {}).get("id")) if deco in ("fluted", "grooved") else None
            if deco in ("fluted", "grooved", "recessed_panel", "raised_panel", "arched") and not p.get("mouldings") and (deco not in ("fluted", "grooved") or band):
                region = None
                if band:
                    region = (np.array([lo[0], hi[1] - band["to"] * (hi[1] - lo[1]), lo[2]]), np.array([hi[0], hi[1] - band["from"] * (hi[1] - lo[1]), hi[2]]))
                solid, made = solids.decorated(lo, hi, deco, band, region=region)
                decorations.append({"component": p["id"], **made})
            else:
                solid = solids.box(lo, hi)
                if deco in ("fluted", "grooved") and not band:
                    decorations.append({"component": p["id"], "decoration": deco, "method": "not built", "reason": "The photo shows no regular lines on this part."})
        if p.get("mouldings"):
            solid, made = solids.with_mouldings(solid, lo, p["hi"], [m if isinstance(m, dict) else m.model_dump() for m in p["mouldings"]])
            decorations += [{"component": p["id"], "decoration": f"{m['profile']} moulding", **m} for m in made]
        built.append({
            "name": p["name"], "id": p["id"], "kind": p["kind"], "assembly": p.get("assembly"), "lo": lo, "hi": hi,
            "mesh": solids.mesh_arrays(solid), "solid": solid, "texture": mat.get("texture"), "colour": [float(c) for c in mat["color"]],
            "alpha": mat.get("alpha", 1.0), "metallic": mat["metallic"], "roughness": mat["roughness"], "spec": spec, "estimated": p["estimated"],
            "material": {k: mat.get(k) for k in ("class", "source", "confidence", "grain") if mat.get(k) is not None}, "mover_of": p.get("mover_of"), "host": p.get("host"),
            **({"hardware": {k: v for k, v in p["hardware"].items() if k not in ("x0", "x1", "y0", "y1")}} if p.get("hardware") else {}),
            "extras": {"partId": p["id"], "kind": p["kind"], "assembly": p.get("assembly"), "source": "inferred" if p["estimated"] else "observed",
                       "generation": "procedural", **({"specId": spec["id"]} if spec else {}), **({"estimated": True} if p["estimated"] else {})},
        })
    for r in recesses:  # finger recesses are cut into their door (the door stays one editable part)
        host = next((b for b in built if b["id"] == r.get("host")), None)
        if host is not None:
            host["solid"] = host["solid"] - solids.recess(r["lo"], r["hi"])
            host["mesh"] = solids.mesh_arrays(host["solid"])
            decorations.append({"component": host["id"], "decoration": "recessed handle", "method": "geometry"})
    glb = write(built, plan["assemblies"])
    components = [{
        "id": b["id"], "name": b["name"], "kind": b["kind"], "assembly": b["assembly"], "mesh": i, "node": i,
        "nameAr": (b["spec"] or {}).get("nameAr") or None, "editable": True,
        "box": [[round(float(v), 4) for v in b["lo"]], [round(float(v), 4) for v in b["hi"]]],
        **({"analysisId": b["spec"]["id"]} if b["spec"] and not b["estimated"] else {}),
        **({"estimated": True, "naming": "estimated"} if b["estimated"] else {}),
        "material": {**b["material"], "color": "#%02X%02X%02X" % tuple(int(255 * min(max(c, 0), 1)) for c in b["colour"]),
                     "fromPhoto": b["material"].get("source") in ("texture sample", "colour sample"), "confirmed": False},
        "generation": "procedural",
        **({"hardware": b["hardware"], "host": b.get("host")} if b.get("hardware") else {}),
    } for i, b in enumerate(built)]
    size_m = {"width": round(W, 3), "height": round(H, 3), "depth": round(D, 3)}
    doc: dict[str, Any] = {
        "version": 2, "separated": True, "method": mode, "modelType": family, "components": components, "analysis": analysis,
        "assemblies": [{"id": a["id"], "name": a.get("name"), "kind": a.get("kind"), "children": [b["id"] for b in built if b["assembly"] == a["id"]]}
                       for a in plan["assemblies"]],
        "specDetails": {k: details.get(k) for k in ("bbox", "width_m", "height_m", "depth_m", "dimension_source", "proportions", "object_type") if k in details},
        "sizeM": size_m, "scaleKnown": known,
        "sizeSource": "vendor" if details.get("dimension_source") == "vendor" else "photo" if known else "estimated",
        "hybrid": {"mode": mode, "engine": "assembly", "decorations": decorations, "materialFallbacks": fallbacks, "notes": plan["notes"],
                   "frontPlane": plan.get("frontPlane"),
                   "note": "Sizes are estimated from the photo unless the vendor confirmed them; inferred parts are marked estimated."},
    }
    pending = pending_components(analysis, details, rgb, (W, H, D))
    for c in pending:
        movable = {x["id"] for x in components if any(b["mover_of"] and b["id"] == x["id"] for b in built)}
        c["attachedTo"] = host_of(c, components, movable)
    if pending:
        doc["pendingComponents"] = [{k: v for k, v in c.items() if k != "image"} for c in pending]
        doc["hybrid"]["complex"] = [c["id"] for c in pending]
    doc["animation"] = motions(built, plan["assemblies"], (W, H, D))
    doc["attachments"] = {f: a["component"] for a in doc["animation"]["animations"] for f in a["follow"]}
    doc["validation"] = validate(doc, family)
    doc["_pendingImages"] = {c["group"]: c["image"] for c in pending}  # popped by the caller: never stored or sent
    logger.info("Hybrid %s (assembly): %d parts, %d assemblies, %d textured, %d fallbacks, %d motions", family, len(built), len(plan["assemblies"]),
                sum(bool(b["texture"]) for b in built), len(fallbacks), len(doc["animation"]["animations"]))
    return glb, doc


HARDWARE_KINDS = ("handle", "knob")


def measure_hardware(analysis: dict[str, Any], details: dict[str, Any], rgb: np.ndarray) -> dict[str, Any]:
    """Each handle the reading lists, found and measured on its owner's straightened face in the photo (photo_tools.
    find_bar): its exact place, length, orientation, thickness when the photo resolves it, and its own colour. Only
    hardware the reading lists is measured (a reflection in a mirror is never taken for a handle); panes and excluded
    regions are masked out of the search. No extra AI call."""
    import copy

    import cv2

    analysis = copy.deepcopy(analysis)
    comps = analysis.get("components") or []
    by_id = {c["id"]: c for c in comps}
    obj = details.get("bbox")
    h, w = rgb.shape[:2]
    for c in comps:
        if assembly.kind_of(c) not in HARDWARE_KINDS and not any(x in _words(c) for x in HANDLE_WORDS):
            continue
        owner_id = c.get("assembly") or next((r.get("target") for r in c.get("relations") or [] if r.get("type") in ("attached_to", "on")), None)
        owner = by_id.get(owner_id or "")
        quad = materials.quad_of(owner, obj) if owner else None
        if not quad:
            continue
        mask = np.full((h, w), 255, np.uint8)
        for o in comps:  # panes of the owner and everything excluded: never searched for a handle
            if o is not c and (o.get("assembly") == owner_id and assembly.kind_of(o) in ("mirror", "glass")):
                q = materials.quad_of(o, obj)
                if q:
                    cv2.fillPoly(mask, [np.array([[p[0] * w, p[1] * h] for p in q], np.int32)], 0)
        for e in analysis.get("exclusions") or []:
            cv2.fillPoly(mask, [np.array([[p[0] * w, p[1] * h] for p in e["quad"]], np.int32)], 0)
        face, valid = materials.rectify(rgb, mask, quad)
        declared = str(c.get("hardwareType") or "").lower()
        expect = "knob" if "knob" in declared or assembly.kind_of(c) == "knob" else ("horizontal" if "horizontal" in declared else "vertical" if "vertical" in declared or "bar" in declared else None)
        found = photo_tools.find_bar(face, valid, expect)
        if not found:
            continue
        fh, fw = face.shape[:2]
        x0, y0, x1, y1 = found["rect"]
        src = np.array([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]], np.float32)
        dst = np.array([[p[0] * w, p[1] * h] for p in quad], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)  # the face back into the image
        corners = np.array([[[x0 * fw, y0 * fh], [x1 * fw, y0 * fh], [x1 * fw, y1 * fh], [x0 * fw, y1 * fh]]], np.float32)
        img_pts = cv2.perspectiveTransform(corners, M)[0]
        c["quad"] = [[float(p[0] / w), float(p[1] / h)] for p in img_pts]
        c["assembly"] = c.get("assembly") or owner_id
        short_px = min((x1 - x0) * fw, (y1 - y0) * fh)
        c["measured"] = {"orientation": found["orientation"], "aspect": found["aspect"], "colour": found["colour"], "pixels": found["pixels"],
                         "shortPx": round(float(short_px), 1), "confidence": found["confidence"], "owner": owner_id}
    return analysis


MOVING_KINDS = ("door", "sliding_door", "drawer_front", "door_frame")


def motions(built: list[dict[str, Any]], assemblies: list[dict[str, Any]], size: tuple[float, float, float]) -> dict[str, Any]:
    """One movement per movable assembly (a door, a drawer, a mirrored door: its frame moves and its pane, decorations
    and hardware go with it), written at build time.

    What the photo does not show clearly is filled with a sensible, editable default and recorded as a NOTE, never as an
    issue (an issue disables the movement in the editor): an unknown hinge is taken from the handle (the hinge is on the
    other side), else from the door's place in its row (a pair of doors opens from the seam between them). A door the
    reading saw but gave no motion still opens when it is clearly a door, with that note."""
    W, H, D = size
    whole = ([-W / 2, 0.0, -D / 2], [W / 2, H, D / 2])
    asm = {a["id"]: a for a in assemblies}
    movers = [b for b in built if b.get("mover_of")]
    hinged_row = sorted([b for b in movers if b["kind"] in ("door", "door_frame")], key=lambda b: b["lo"][0])
    entries, uncertain = [], []
    for b in movers:
        owner = b["mover_of"]
        spec = asm.get(owner, {}).get("spec") or b.get("spec") or {}
        motion = dict(spec.get("motion") or {})
        notes: list[str] = []
        kind = (asm.get(owner) or {}).get("kind") or b["kind"]
        if motion.get("type") in (None, "none"):
            # A door the reading confidently says is fixed stays fixed; an unclear one gets a usual, editable movement.
            if kind not in ("door", "sliding_door", "drawer_front") or float(motion.get("confidence") or 0) >= 0.6:
                continue
            motion = {"type": "slide" if kind in ("sliding_door", "drawer_front") else "rotate", "confidence": 0.2}
            notes.append("The photo does not show how it opens: a usual movement for its kind is set; change it if needed.")
        follow = [o["id"] for o in built if o is not b and (o.get("assembly") == owner and owner in asm or o.get("host") in (b["id"], owner))]
        if motion["type"] == "rotate" and motion.get("hinge") in (None, "none"):
            handles = [o for o in built if o["id"] in follow and o["kind"] in ("handle", "knob")]
            cx = (b["lo"][0] + b["hi"][0]) / 2
            if handles:
                hx = (handles[0]["lo"][0] + handles[0]["hi"][0]) / 2
                motion["hinge"] = "left" if hx > cx else "right"
                notes.append("Hinge side taken from the handle (the hinge is opposite it).")
            else:
                i = hinged_row.index(b) if b in hinged_row else -1
                left_n = hinged_row[i - 1] if i > 0 else None
                right_n = hinged_row[i + 1] if 0 <= i < len(hinged_row) - 1 else None
                has_handle = lambda d: any(o.get("host") in (d["id"], d.get("mover_of")) or (o.get("assembly") == d.get("mover_of") and o["kind"] in ("handle", "knob")) for o in built)  # noqa: E731
                # Two handleless doors side by side open from the seam between them: each hinges on its outer edge.
                if right_n is not None and not has_handle(right_n) and (left_n is None or has_handle(left_n) or i % 2 == (1 if hinged_row and has_handle(hinged_row[0]) else 0)):
                    motion["hinge"] = "left"
                elif left_n is not None and not has_handle(left_n):
                    motion["hinge"] = "right"
                else:
                    motion["hinge"] = "left" if cx < 0 else "right"
                notes.append("Hinge side not visible: set from the door's place in its row.")
        if motion["type"] == "slide" and motion.get("direction") in (None, "none", "front"):
            motion["direction"] = "front" if kind == "drawer_front" or b["kind"] == "drawer_front" else "right" if (b["lo"][0] + b["hi"][0]) / 2 < 0 else "left"
        if motion["type"] == "slide" and motion["direction"] in ("left", "right"):
            motion["travel"] = max(0.05, ((b["hi"][0] - b["lo"][0]) - 0.04) / W)
        entry = animation._entry({"id": b["id"], "box": [b["lo"].tolist(), b["hi"].tolist()]}, motion, whole, spec.get("name") or b["name"], spec.get("nameAr") or None, "assembly")
        entry["follow"] = follow
        notes += [i for i in entry["issues"] if not i.startswith("The hinge side was not detected")]
        entry["issues"] = []  # nothing here is a defect: the movement is set, editable, and checked for collisions
        entry["notes"] = notes
        entry["enabled"] = True
        if float(motion.get("confidence") or 0) < 0.4 or notes:
            entry["uncertain"] = True
            uncertain.append(entry["label"])
        entries.append(entry)
    return {"version": 1, "movable": bool(entries), "animations": entries, "flagged": [], "source": "assembly", "authoritative": True,
            **({"uncertain": uncertain} if uncertain else {})}


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


def pending_components(analysis: dict[str, Any], details: dict[str, Any], rgb: np.ndarray, size: tuple[float, float, float]) -> list[dict[str, Any]]:
    """The complex parts, each with its box in the model (placed like any other part: its perspective-corrected front
    rectangle, its role's depth), and its cutout from the original photo for the mesh provider."""
    W, H, D = size
    quad = assembly.front_quad(analysis, details)
    Hm = assembly.homography(quad) if quad else None
    obj = details.get("bbox") or [0, 0, 1, 1]
    mask = photo_tools.silhouette(rgb, obj)
    complex_comps = [c for c in analysis.get("components") or [] if is_complex(c) and (c.get("box") or c.get("quad")) and c.get("visibility") != "inferred"]
    placed = assembly._loose(complex_comps, lambda c: assembly.front_rect(c, Hm, obj), W, H, D)
    out = []
    for p in placed:
        c = p["spec"]
        if c.get("quad"):
            xs, ys = [q[0] for q in c["quad"]], [q[1] for q in c["quad"]]
            rect = (int(min(xs) * rgb.shape[1]), int(min(ys) * rgb.shape[0]), int(max(xs) * rgb.shape[1]), int(max(ys) * rgb.shape[0]))
        else:
            rect = _rect(obj, c["box"], (rgb.shape[1], rgb.shape[0]))
        if rect is None or rect[2] - rect[0] < 4 or rect[3] - rect[1] < 4:
            continue
        lo, hi = p["lo"], p["hi"]
        out.append({"id": re.sub(r"[^a-z0-9]+", "_", f"{c.get('name') or c['id']}".lower()).strip("_") + "_generated", "specId": c["id"],
                    "name": c.get("name") or c["id"], "nameAr": c.get("nameAr") or None, "decoration": c.get("decoration"),
                    "box": [[round(float(v), 4) for v in lo], [round(float(v), 4) for v in hi]],
                    "relatedTo": [r.get("target") for r in c.get("relations") or [] if r.get("type") in ("attached_to", "on")] + ([c["assembly"]] if c.get("assembly") else []),
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
        if cls not in ("glass", "mirror", "metal") and c.get("kind") not in ("back_panel", "mirror", "glass") and not any(w in _words(c) for w in HANDLE_WORDS) and thin < 0.5 * board:
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
