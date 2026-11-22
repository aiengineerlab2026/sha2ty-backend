"""Photographed lighting tiled across panels (rectangular patches) becomes the panel's own even colour; real wood
grain and UV atlases are left alone. No geometry, UV or name changes."""

import io

import numpy as np
import trimesh
from PIL import Image

from app.services import fidelity, surface_finish
from app.services.glb_size import read_glb


def panel(image: Image.Image, tiles: float, cols: int = 30, rows: int = 6) -> bytes:
    """A 2.9 m × 0.57 m back panel (facing +z, a fine grid of quads) whose colour texture is repeated `tiles` times
    along it."""
    xs, ys = np.linspace(0, 1, cols + 1), np.linspace(0, 1, rows + 1)
    gx, gy = np.meshgrid(xs, ys)
    verts = np.stack([gx.ravel() * 2.9 - 1.45, gy.ravel() * 0.57, np.zeros(gx.size)], -1)
    uv = np.stack([gx.ravel() * tiles, gy.ravel()], -1)
    faces = []
    for r in range(rows):
        for c in range(cols):
            i = r * (cols + 1) + c
            faces += [[i, i + 1, i + cols + 2], [i, i + cols + 2, i + cols + 1]]
    mesh = trimesh.Trimesh(verts, faces, process=False, visual=trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(name="Dark_backing", baseColorTexture=image, roughnessFactor=0.4)))
    return trimesh.Scene({"Back panel": mesh}).export(file_type="glb")


def lit_crop(base: tuple[int, int, int] = (36, 34, 30), light: float = 190.0) -> Image.Image:
    """A finish photographed with a highlight (or, negative, a shadow) in one corner: what Astra cropped from the photo."""
    y, x = np.mgrid[0:128, 0:128]
    glow = np.exp(-(((x - 100) ** 2 + (y - 60) ** 2) / 900.0))
    rgb = np.stack([b + light * glow for b in base], -1)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def grain() -> Image.Image:
    """Fine wood grain: stripes, evenly lit."""
    x = np.arange(256)
    row = 120 + 25 * np.sin(x / 2.5) + 10 * np.sin(x / 0.9)
    rgb = np.stack([row, row * 0.75, row * 0.5], -1)[None].repeat(256, 0)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def base(glb: bytes):
    doc, _ = read_glb(glb)
    return doc["materials"][0]["pbrMetallicRoughness"]


def test_photographed_lighting_tiled_on_a_panel_becomes_its_even_colour():
    raw = panel(lit_crop(), 4.5)
    fixed, changed = surface_finish.flatten(raw)
    pbr = base(fixed)
    assert len(changed) == 1 and changed[0]["shading"] > surface_finish.SHADED and changed[0]["tiles"] == 4.5
    assert "baseColorTexture" not in pbr and pbr["roughnessFactor"] == 0.4  # the finish keeps its own properties
    colour = [int(changed[0]["colour"][i : i + 2], 16) for i in (1, 3, 5)]
    assert all(abs(c - b) <= 8 for c, b in zip(colour, (36, 34, 30))), changed  # the finish's own dark colour, not the highlight
    # Nothing else moved: same vertices, UVs and name.
    a = trimesh.load(io.BytesIO(raw), file_type="glb", force="scene")
    b = trimesh.load(io.BytesIO(fixed), file_type="glb", force="scene")
    assert list(a.geometry) == list(b.geometry)
    assert np.allclose(next(iter(a.geometry.values())).vertices, next(iter(b.geometry.values())).vertices)
    # And the check after processing calls it a correction, not a lost map.
    stages = {s["stage"]: s for s in fidelity.compare_glbs(raw, fixed)}
    assert stages["materials"]["ok"] and any("even colour" in n for n in stages["materials"]["notes"])


def test_an_evenly_lit_render_of_the_fixed_panel_has_no_patches():
    fixed, _ = surface_finish.flatten(panel(lit_crop(), 4.5))
    rgb, mask, _ = fidelity.front_render(trimesh.Scene(trimesh.load(io.BytesIO(fixed), file_type="glb", force="scene")).export(file_type="glb"), 64)
    before, bmask, _ = fidelity.front_render(panel(lit_crop(), 4.5), 64)
    assert rgb[mask].std(axis=0).max() < 1.0 < before[bmask].std(axis=0).max()  # one colour after; patches before


def test_real_wood_grain_and_a_uv_atlas_are_left_alone():
    assert surface_finish.flatten(panel(grain(), 4.5))[1] == []  # repeated, but its variation is grain, not light
    assert surface_finish.flatten(panel(lit_crop(), 1.0))[1] == []  # shaded, but an atlas (not repeated): Meshy's case


def test_black_and_white_finishes_keep_their_own_colour():
    for base, light, want in (((18, 18, 18), 200.0, "#1"), ((236, 236, 236), -150.0, "#e")):
        _, changed = surface_finish.flatten(panel(lit_crop(base, light), 3))
        assert changed and changed[0]["colour"].startswith(want), changed  # black stays black, white stays white
