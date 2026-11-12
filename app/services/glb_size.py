"""Make generated GLBs lighter, aiming for MAX_GLB_BYTES (8 MB): no provider can promise a file size.

8 MB is a target, not a limit: a model still over it after optimising is saved as it is (never refused).

Meshy is asked for fewer polygons and 2k textures (its real parameters) and both providers are told the limit in
their prompt, but the file that comes back is what counts. Over the limit, `fit_glb` tries, smallest change first:
  1. textures re-encoded at most 2048 px (JPEG when the image has no transparency, else PNG);
  2. vertex data stored compactly (KHR_mesh_quantization: normals as int8, UVs in [0, 1] as uint16, indices as
     uint16 where they fit), which three.js and Filament read natively;
  3. textures at 1024, then 512 px.
Meshes, nodes, names and materials are never merged or removed, so every part stays its own editable mesh.
"""

from __future__ import annotations

import io
import json
import logging
import struct
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MAX_GLB_BYTES = 8 * 1024 * 1024
QUANTIZATION = "KHR_mesh_quantization"
_TYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    if data[:4] != b"glTF" or len(data) < 20:
        raise ValueError("not a GLB")
    json_len = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20 : 20 + json_len])
    rest = 20 + json_len
    binary = b""
    if rest + 8 <= len(data):
        bin_len, kind = struct.unpack_from("<I4s", data, rest)
        if kind == b"BIN\x00":
            binary = data[rest + 8 : rest + 8 + bin_len]
    return doc, binary


def write_glb(doc: dict[str, Any], views: list[bytes]) -> bytes:
    """`views`: each bufferView's bytes, in order; the buffer is rebuilt from them (4-byte aligned)."""
    out = bytearray()
    for view, blob in zip(doc.get("bufferViews") or [], views):
        out += b"\x00" * (-len(out) % 4)
        view["byteOffset"], view["byteLength"], view["buffer"] = len(out), len(blob), 0
        out += blob
    out += b"\x00" * (-len(out) % 4)
    if doc.get("buffers"):
        doc["buffers"] = [{"byteLength": len(out)}]
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text + (struct.pack("<I4s", len(out), b"BIN\x00") + bytes(out) if out else b"")
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def _views(doc: dict[str, Any], binary: bytes) -> list[bytes]:
    return [binary[v.get("byteOffset", 0) : v.get("byteOffset", 0) + v["byteLength"]] for v in doc.get("bufferViews") or []]


def _textures(doc: dict[str, Any], views: list[bytes], max_side: int) -> None:
    from PIL import Image

    for image in doc.get("images") or []:
        index = image.get("bufferView")
        if not isinstance(index, int):
            continue
        try:
            picture = Image.open(io.BytesIO(views[index]))
            picture.load()
        except Exception:  # noqa: BLE001 - an image Pillow cannot read is left as it is
            continue
        if max(picture.size) > max_side:
            picture.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        clear = picture.mode in ("RGBA", "LA", "PA") and picture.getchannel("A").getextrema()[0] < 255
        buffer = io.BytesIO()
        if clear:
            picture.save(buffer, "PNG", optimize=True)
        else:
            picture.convert("RGB").save(buffer, "JPEG", quality=85, optimize=True)
        if buffer.tell() < len(views[index]):
            views[index] = buffer.getvalue()
            image["mimeType"] = "image/png" if clear else "image/jpeg"


def _quantize(doc: dict[str, Any], views: list[bytes]) -> None:
    """Normals → int8, UVs → uint16 (only within [0, 1]), indices → uint16, for accessors that own their bufferView."""
    accessors = doc.get("accessors") or []
    owners: dict[int, int] = {}
    for acc in accessors:
        if isinstance(acc.get("bufferView"), int):
            owners[acc["bufferView"]] = owners.get(acc["bufferView"], 0) + 1

    def values(acc):
        view = doc["bufferViews"][acc["bufferView"]]
        n = _SIZES.get(acc["type"], 1)
        dtype = np.dtype(_TYPES[acc["componentType"]])
        if view.get("byteStride", n * dtype.itemsize) != n * dtype.itemsize or acc.get("byteOffset", 0) or acc.get("sparse"):
            return None
        return np.frombuffer(views[acc["bufferView"]], dtype=dtype, count=acc["count"] * n).reshape(acc["count"], n)

    def replace(acc, data: np.ndarray, component: int, stride: int | None, normalized: bool) -> None:
        view = doc["bufferViews"][acc["bufferView"]]
        views[acc["bufferView"]] = data.tobytes()
        acc["componentType"] = component
        acc["normalized"] = normalized or None
        if acc["normalized"] is None:
            acc.pop("normalized")
        if stride:
            view["byteStride"] = stride
        else:
            view.pop("byteStride", None)

    done: set[int] = set()
    used = False
    for mesh in doc.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            attrs = prim.get("attributes") or {}
            jobs = [("NORMAL", attrs.get("NORMAL")), ("TEXCOORD_0", attrs.get("TEXCOORD_0")), ("indices", prim.get("indices"))]
            for kind, index in jobs:
                if not isinstance(index, int) or index in done:
                    continue
                acc = accessors[index]
                if owners.get(acc.get("bufferView"), 0) != 1 or acc.get("componentType") not in (5126, 5125):
                    continue
                data = values(acc)
                if data is None:
                    continue
                done.add(index)
                if kind == "NORMAL" and acc["componentType"] == 5126:
                    q = np.clip(np.round(data * 127), -127, 127).astype(np.int8)
                    replace(acc, np.hstack([q, np.zeros((len(q), 1), np.int8)]), 5120, 4, True)  # padded to 4 bytes
                    used = True
                elif kind == "TEXCOORD_0" and acc["componentType"] == 5126 and data.size and data.min() >= 0 and data.max() <= 1:
                    replace(acc, np.round(data * 65535).astype(np.uint16), 5123, None, True)
                    used = True
                elif kind == "indices" and acc["componentType"] == 5125 and (not data.size or data.max() < 65535):
                    replace(acc, data.astype(np.uint16), 5123, None, False)
    if used:
        for key in ("extensionsUsed", "extensionsRequired"):
            if QUANTIZATION not in doc.setdefault(key, []):
                doc[key].append(QUANTIZATION)


def fit_glb(data: bytes, limit: int = MAX_GLB_BYTES) -> tuple[bytes, dict[str, Any]]:
    """The GLB at most `limit` bytes when it can be made so, and a report: size before / after and the steps used.
    The structure (meshes, nodes, materials) is checked unchanged; a file that cannot be read is returned as it is."""
    report: dict[str, Any] = {"before": len(data), "after": len(data), "limit": limit, "steps": []}
    if len(data) <= limit:
        return data, report
    try:
        doc, binary = read_glb(data)
    except (ValueError, json.JSONDecodeError):
        return data, report
    shape = lambda d: (len(d.get("meshes") or []), len(d.get("nodes") or []), len(d.get("materials") or []))  # noqa: E731
    before = shape(doc)
    views = _views(doc, binary)
    best = data
    for step, apply in (
        ("textures 2048 px", lambda: _textures(doc, views, 2048)),
        ("compact vertex data", lambda: _quantize(doc, views)),
        ("textures 1024 px", lambda: _textures(doc, views, 1024)),
        ("textures 512 px", lambda: _textures(doc, views, 512)),
    ):
        apply()
        report["steps"].append(step)
        out = write_glb(json.loads(json.dumps(doc)), list(views))
        if len(out) < len(best):
            best = out
        if len(best) <= limit:
            break
    if shape(read_glb(best)[0]) != before:  # never: the steps only re-encode data
        logger.error("GLB optimisation changed the model's structure; keeping the original")
        return data, report
    report["after"] = len(best)
    logger.info("GLB %.1f MB -> %.1f MB (%s)", len(data) / 1048576, len(best) / 1048576, ", ".join(report["steps"]))
    return best, report
