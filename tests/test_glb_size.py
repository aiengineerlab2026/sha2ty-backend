import numpy as np
import pytest
import trimesh
from PIL import Image

from app.services.glb_parts import inspect_glb
from app.services.glb_size import GlbTooLarge, MAX_GLB_BYTES, enforce, fit_glb, read_glb


def _chair_glb() -> bytes:
    """Three separate textured parts sharing one large noisy texture (like a generated model's 4k PNG)."""
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (3000, 3000, 3), dtype=np.uint8))
    scene = trimesh.Scene()
    for name, (w, h, d, y) in {"seat": (0.5, 0.08, 0.5, 0.45), "backrest": (0.5, 0.5, 0.05, 0.75), "legs": (0.5, 0.45, 0.5, 0.22)}.items():
        part = trimesh.creation.box(extents=(w, h, d)).subdivide().subdivide()
        part.apply_translation((0, y, 0))
        part.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(1).random((len(part.vertices), 2)), image=noise)
        scene.add_geometry(part, geom_name=name, node_name=name)
    return scene.export(file_type="glb")


def test_an_oversized_glb_is_compacted_with_every_part_kept():
    original = _chair_glb()
    limit = len(original) // 4
    out, report = fit_glb(original, limit)
    assert len(out) <= limit and report["after"] == len(out) and report["steps"]
    assert [c["name"] for c in inspect_glb(out)["components"]] == [c["name"] for c in inspect_glb(original)["components"]]
    loaded = trimesh.load(trimesh.util.wrap_as_stream(out), file_type="glb", force="scene")
    assert set(loaded.geometry) == {"seat", "backrest", "legs"}
    for part in loaded.geometry.values():  # still textured, UVs kept
        assert part.visual.uv is not None and len(part.visual.uv) == len(part.vertices)


def test_vertex_data_is_stored_compactly_and_stays_close():
    original = _chair_glb()
    doc, _ = read_glb(fit_glb(original, 1)[0])  # an impossible limit runs every step
    assert "KHR_mesh_quantization" in doc["extensionsUsed"]
    loaded = trimesh.load(trimesh.util.wrap_as_stream(fit_glb(original, 1)[0]), file_type="glb", force="scene")
    before = trimesh.load(trimesh.util.wrap_as_stream(original), file_type="glb", force="scene")
    for name, part in before.geometry.items():
        assert np.allclose(loaded.geometry[name].vertices, part.vertices)  # positions untouched
        # trimesh ignores glTF's `normalized` flag (three.js and Filament apply it): undo it as they do.
        assert np.abs(loaded.geometry[name].visual.uv / 65535 - part.visual.uv).max() < 1e-4


def test_small_files_are_left_alone_and_too_large_ones_refused():
    small = _chair_glb()
    assert fit_glb(small, len(small))[0] == small
    assert len(small) < MAX_GLB_BYTES or enforce(small)
    with pytest.raises(GlbTooLarge, match="still"):
        import app.services.glb_size as glb_size

        glb_size.MAX_GLB_BYTES, keep = 100, glb_size.MAX_GLB_BYTES
        try:
            enforce(small)
        finally:
            glb_size.MAX_GLB_BYTES = keep
