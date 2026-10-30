"""Convert CAD files that Three.js cannot load into cached GLB meshes."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import trimesh

GLB_SUFFIXES = {".glb", ".gltf"}
TRIMESH_SUFFIXES = {".stl", ".obj", ".ply", ".off", ".dae"}


def convert_to_glb(data: bytes, filename: str, cache_dir: Path) -> tuple[bytes, str]:
    suffix = Path(filename).suffix.lower() or ".bin"
    digest = hashlib.sha256(data).hexdigest()[:16]
    cached = cache_dir / f"{digest}.glb"
    if cached.exists():
        return cached.read_bytes(), "cache"
    if suffix in GLB_SUFFIXES:
        cached.write_bytes(data)
        return data, "passthrough"
    if suffix in TRIMESH_SUFFIXES:
        mesh = trimesh.load(io.BytesIO(data), file_type=suffix.lstrip("."), force="mesh")
        scene = mesh if isinstance(mesh, trimesh.Scene) else trimesh.Scene(mesh)
        buffer = io.BytesIO()
        scene.export(file_obj=buffer, file_type="glb")
        out = buffer.getvalue()
        cached.write_bytes(out)
        return out, "trimesh"
    if suffix in {".step", ".stp", ".iges", ".igs"}:
        raise ValueError(
            "STEP/IGES conversion needs CadQuery/OpenCascade. This server stores the source file but "
            "does not claim a parametric solid conversion. Upload a GLB for the editor."
        )
    raise ValueError(f"Unsupported CAD format: {suffix or 'unknown'}. Use GLB/GLTF.")
