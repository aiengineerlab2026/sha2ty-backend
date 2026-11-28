"""A small software renderer for checking generated furniture (no GPU, no extra dependency): triangles rasterised with
a z-buffer, Lambert shading, base-colour textures sampled through their UVs, from any camera around the model.

  render(glb, yaw, pitch)   → (RGB image, part-id map, mirror mask)
      yaw   degrees around the vertical (0 = front, 90 = the model's right side, 180 = rear)
      pitch degrees above the horizon

The part-id map says which component each pixel shows (glTF node extras.partId), so a render can be compared with the
photo component by component. Mirrors (metallic ≥ 0.9, roughness ≤ 0.1) are drawn as a neutral sky gradient and
reported in their own mask: what they reflect depends on the room, never compared pixel by pixel.

ponytail: one triangle at a time in numpy (a few thousand triangles at 480 px take about a second). A GPU or
pyrender path is the upgrade if renders are needed in bulk.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

LIGHT = np.array([-0.35, 0.6, 0.75])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def _parts(glb: bytes) -> list[dict[str, Any]]:
    """Each mesh primitive in world coordinates with its part id, colour / texture and whether it is a mirror."""
    import trimesh

    from app.services.glb_size import read_glb

    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene", process=False)
    # trimesh drops glTF node extras: the part ids come from the file's own nodes, by name.
    part_of = {nd.get("name"): (nd.get("extras") or {}).get("partId") for nd in read_glb(glb)[0].get("nodes") or [] if nd.get("name")}
    out = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geom = scene.geometry[name]
        if not hasattr(geom, "faces") or not len(geom.faces):
            continue
        extras = {"partId": part_of.get(node) or part_of.get(str(node).rsplit("_", 1)[0])}
        verts = trimesh.transform_points(geom.vertices, transform)
        mat = getattr(geom.visual, "material", None)
        colour, texture, uv, metallic, rough = np.array([0.8, 0.8, 0.8]), None, None, 0.0, 0.7
        if mat is not None:
            factor = getattr(mat, "baseColorFactor", None)
            if factor is not None:
                lin = np.asarray(factor[:3], float) / (255.0 if np.max(factor) > 1 else 1.0)
                colour = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)  # linear → sRGB
            img = getattr(mat, "baseColorTexture", None)
            if img is not None and getattr(geom.visual, "uv", None) is not None:
                texture, uv = np.asarray(img.convert("RGB"), float) / 255.0, np.asarray(geom.visual.uv, float)
            metallic = float(getattr(mat, "metallicFactor", 0.0) or 0.0)
            rough = float(getattr(mat, "roughnessFactor", 0.7) if getattr(mat, "roughnessFactor", None) is not None else 0.7)
        out.append({"id": extras.get("partId") or name, "verts": verts, "faces": np.asarray(geom.faces), "colour": colour, "texture": texture,
                    "uv": uv, "mirror": metallic >= 0.9 and rough <= 0.1})
    return out


def _camera(points: np.ndarray, yaw: float, pitch: float):
    """World → view rotation looking at the model's centre from (yaw, pitch); y up."""
    y, p = np.radians(yaw), np.radians(pitch)
    forward = -np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])  # from the eye towards the model
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    centre = (points.min(0) + points.max(0)) / 2
    return np.stack([right, up, -forward]), centre


def render(glb: bytes, yaw: float = 0.0, pitch: float = 8.0, size: int = 480, fov: float = 30.0, background=(1.0, 1.0, 1.0),
           ortho: tuple[float, float, float, float] | None = None, shaded: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """(RGB uint8, part index map (-1 = background), mirror mask, part ids) of the model seen from (yaw, pitch).
    `ortho` (x0, y0, x1, y1 in metres): a front orthographic view of exactly that rectangle onto the size × size image
    (for comparing with the photo's perspective-corrected front). `shaded` False: base colours only."""
    parts = _parts(glb)
    allv = np.concatenate([p["verts"] for p in parts])
    rot, centre = _camera(allv, yaw, pitch)
    radius = float(np.linalg.norm(allv.max(0) - allv.min(0))) / 2
    dist = radius / np.tan(np.radians(fov) / 2) * 1.05
    f = size / 2 / np.tan(np.radians(fov) / 2)
    img = np.ones((size, size, 3)) * np.asarray(background)
    zbuf = np.full((size, size), np.inf)
    ids = np.full((size, size), -1, np.int32)
    mirror = np.zeros((size, size), bool)
    view_dir = rot[2]
    for pi, part in enumerate(parts):
        if ortho is not None:
            x0, y0, x1, y1 = ortho
            cam = part["verts"].copy()
            sx = (cam[:, 0] - x0) / (x1 - x0) * size
            sy = (y1 - cam[:, 1]) / (y1 - y0) * size
            cam[:, 2] = 100.0 - cam[:, 2]  # nearer the front = smaller depth
        else:
            cam = (part["verts"] - centre) @ rot.T
            cam[:, 2] = dist - cam[:, 2]  # depth in front of the eye
            sx = size / 2 + f * cam[:, 0] / np.maximum(cam[:, 2], 1e-6)
            sy = size / 2 - f * cam[:, 1] / np.maximum(cam[:, 2], 1e-6)
        for tri in part["faces"]:
            a, b, c = tri
            xs, ys, zs = sx[tri], sy[tri], cam[tri, 2]
            x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), size - 1)
            y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), size - 1)
            if x0 > x1 or y0 > y1:
                continue
            den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(den) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
            w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            # perspective-correct depth and attributes
            iz = w0 / zs[0] + w1 / zs[1] + w2 / zs[2]
            depth = 1 / np.maximum(iz, 1e-9)
            region = zbuf[y0:y1 + 1, x0:x1 + 1]
            draw = inside & (depth < region)
            if not draw.any():
                continue
            v = part["verts"][tri]
            normal = np.cross(v[1] - v[0], v[2] - v[0])
            normal /= max(np.linalg.norm(normal), 1e-12)
            if np.dot(normal, view_dir) < 0:  # back face: seen from inside (an open frame's edge), shade it from the other side
                normal = -normal
            shade = 0.45 + 0.55 * max(0.0, float(np.dot(normal, LIGHT))) if shaded else 1.0
            if part["mirror"]:
                t = (gy - y0) / max(y1 - y0 + 1, 1)
                colour = np.stack([0.78 + 0.1 * t, 0.83 + 0.08 * t, 0.88 + 0.05 * t], -1)  # a neutral sky gradient
                mirror[y0:y1 + 1, x0:x1 + 1][draw] = True
            elif part["texture"] is not None:
                uv = part["uv"][tri]
                pu = (w0 * uv[0, 0] / zs[0] + w1 * uv[1, 0] / zs[1] + w2 * uv[2, 0] / zs[2]) * depth
                pv = (w0 * uv[0, 1] / zs[0] + w1 * uv[1, 1] / zs[1] + w2 * uv[2, 1] / zs[2]) * depth
                tex = part["texture"]
                th, tw = tex.shape[:2]
                colour = tex[(np.mod(1 - pv, 1.0) * (th - 1)).astype(int), (np.mod(pu, 1.0) * (tw - 1)).astype(int)] * shade
            else:
                colour = np.broadcast_to(part["colour"] * shade, (*draw.shape, 3))
            img[y0:y1 + 1, x0:x1 + 1][draw] = np.asarray(colour)[draw]
            region[draw] = depth[draw]
            ids[y0:y1 + 1, x0:x1 + 1][draw] = pi
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), ids, mirror, [p["id"] for p in parts]


def sheet(glb: bytes, views=((0, 8), (-35, 10), (35, 10), (90, 5), (180, 5)), size: int = 360):
    """Several renders side by side (PIL image): front, the two three-quarters, the side, the rear by default."""
    from PIL import Image

    tiles = [Image.fromarray(render(glb, y, p, size)[0]) for y, p in views]
    out = Image.new("RGB", (size * len(tiles) + 8 * (len(tiles) - 1), size), "white")
    for i, t in enumerate(tiles):
        out.paste(t, (i * (size + 8), 0))
    return out
