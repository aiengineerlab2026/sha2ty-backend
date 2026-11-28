"""One complex component (a carved crest, a cushion, a sculpted leg) generated on its own by the mesh provider and set
into the rebuilt furniture model (services/hybrid.py).

  fit(glb, box)        the provider's mesh scaled uniformly (its proportions kept) and centred into the part's box,
                       its front facing +z as Image to 3D writes it; one mesh, its own material and texture
  check(mesh, box)     a usable result: not empty, enough triangles, its proportions near the box the photo shows
  append(base, part)   the part added to the model GLB as its own named node (partId in its extras), every existing
                       node, material and animation target left as it was
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from app.services.glb_size import read_glb, write_glb, _views

MIN_TRIANGLES = 50
PROPORTION_TOLERANCE = 0.5  # the generated part's width : height may differ this much from the photo's box


def _mesh(glb: bytes):
    import trimesh

    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    meshes = [g for g in scene.dump() if len(g.faces)] if hasattr(scene, "dump") else []
    if not meshes:
        raise ValueError("the generated file has no mesh")
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def check(glb: bytes, box: list[list[float]]) -> list[str]:
    """Why the generated part cannot be used ([] when it can)."""
    try:
        mesh = _mesh(glb)
    except Exception as exc:  # noqa: BLE001 - an unreadable result is a failed generation
        return [f"The generated file could not be read ({str(exc)[:80]})."]
    issues = []
    if len(mesh.faces) < MIN_TRIANGLES:
        issues.append(f"The generated mesh has only {len(mesh.faces)} triangles.")
    ext = mesh.extents
    want = np.subtract(box[1], box[0])
    if ext[1] > 0 and want[1] > 0:
        got, expected = ext[0] / ext[1], want[0] / want[1]
        if abs(got - expected) / expected > PROPORTION_TOLERANCE:
            issues.append(f"Its width : height is {got:.2f}, the photo shows {expected:.2f}.")
    return issues


def fit(glb: bytes, box: list[list[float]]) -> bytes:
    """The generated part placed in its box: uniform scale (no stretching), centred, standing on the box's floor."""
    mesh = _mesh(glb)
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    size = np.maximum(hi - lo, 1e-4)
    ext = np.maximum(mesh.extents, 1e-9)
    scale = float(np.min(size[:2] / ext[:2]))  # width and height from the photo; depth follows the generated shape
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(scale)
    mesh.apply_translation([(lo[0] + hi[0]) / 2, lo[1] + mesh.extents[1] / 2, (lo[2] + hi[2]) / 2])
    return mesh.export(file_type="glb")


def append(base: bytes, part: bytes, name: str, extras: dict[str, Any]) -> tuple[bytes, int]:
    """`part`'s first mesh appended to `base` as a new node under the model's root. (new GLB, the new node index)."""
    bdoc, bbin = read_glb(base)
    pdoc, pbin = read_glb(part)
    views = _views(bdoc, bbin)
    pviews = _views(pdoc, pbin)
    off = {k: len(bdoc.get(k) or []) for k in ("bufferViews", "accessors", "images", "textures", "materials", "samplers", "meshes")}
    for v in pdoc.get("bufferViews") or []:
        bdoc.setdefault("bufferViews", []).append({k: val for k, val in v.items() if k not in ("buffer", "byteOffset", "byteLength")})
    views += pviews
    for a in pdoc.get("accessors") or []:
        bdoc.setdefault("accessors", []).append({**a, **({"bufferView": a["bufferView"] + off["bufferViews"]} if "bufferView" in a else {})})
    for img in pdoc.get("images") or []:
        bdoc.setdefault("images", []).append({**img, **({"bufferView": img["bufferView"] + off["bufferViews"]} if "bufferView" in img else {})})
    for s in pdoc.get("samplers") or []:
        bdoc.setdefault("samplers", []).append(s)
    for t in pdoc.get("textures") or []:
        bdoc.setdefault("textures", []).append({**t, "source": t["source"] + off["images"], **({"sampler": t["sampler"] + off["samplers"]} if "sampler" in t else {})})

    def shift_tex(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: ({**v, "index": v["index"] + off["textures"]} if k.endswith("Texture") and isinstance(v, dict) and "index" in v else shift_tex(v)) for k, v in obj.items()}
        return [shift_tex(v) for v in obj] if isinstance(obj, list) else obj

    for m in pdoc.get("materials") or []:
        bdoc.setdefault("materials", []).append({**shift_tex(m), "name": f"{name}"})
    mesh = (pdoc.get("meshes") or [{}])[0]
    prims = []
    for p in mesh.get("primitives") or []:
        prims.append({**p, "attributes": {k: v + off["accessors"] for k, v in p["attributes"].items()},
                      **({"indices": p["indices"] + off["accessors"]} if "indices" in p else {}),
                      **({"material": p["material"] + off["materials"]} if "material" in p else {})})
    bdoc.setdefault("meshes", []).append({"name": name, "primitives": prims})
    # trimesh writes the mesh with its node transform: carry the node's matrix / TRS onto the new node.
    pnode = next((n for n in pdoc.get("nodes") or [] if n.get("mesh") == 0), {})
    node = {"name": name, "mesh": len(bdoc["meshes"]) - 1, "extras": extras,
            **{k: pnode[k] for k in ("matrix", "translation", "rotation", "scale") if k in pnode}}
    bdoc["nodes"].append(node)
    index = len(bdoc["nodes"]) - 1
    root = bdoc["scenes"][bdoc.get("scene", 0)]["nodes"][0]
    bdoc["nodes"][root].setdefault("children", []).append(index)
    used = set(bdoc.get("extensionsUsed") or []) | set(pdoc.get("extensionsUsed") or [])
    if used:
        bdoc["extensionsUsed"] = sorted(used)
    bdoc["buffers"] = [{}]
    return write_glb(bdoc, views), index


def remove(glb: bytes, part_id: str) -> bytes:
    """The model without the node carrying `part_id` (unlinked from the scene; its data stays unreferenced). Every
    other node keeps its index, so part ids, relations and animations stay valid."""
    doc, binary = read_glb(glb)
    gone = {i for i, n in enumerate(doc.get("nodes") or []) if (n.get("extras") or {}).get("partId") == part_id}
    if not gone:
        return glb
    for n in doc["nodes"]:
        if n.get("children"):
            n["children"] = [c for c in n["children"] if c not in gone]
    for scene in doc.get("scenes") or []:
        scene["nodes"] = [c for c in scene.get("nodes") or [] if c not in gone]
    return write_glb(doc, _views(doc, binary))
