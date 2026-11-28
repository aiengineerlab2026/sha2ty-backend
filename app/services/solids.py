"""Solid geometry for rebuilt furniture (manifold3d: exact booleans on closed, watertight solids).

Every board is a solid box; its decoration is real geometry cut into or added onto its FRONT face (+z, the face the
photo shows):

  grooves(n, axis)        n straight channels, vertical or horizontal
  flutes(n)               n half-round ribs (reeded / fluted fronts)
  recessed / raised       a field inset from the edges, sunk into or standing proud of the face (chamfered rim)
  arch                    an arched field (a round top) recessed into the face
  chamfer                 bevelled front edges
  extrude / revolve       a curved profile (shapely polygon) or a turned leg

Sizes come from the part's box; counts and orientation from the photo (photo_tools.pattern), else sensible defaults
(said in the part's record). Output: flat-shaded triangles (each face its own normal), UVs projected from the front so
the photo texture lands where the photo shows it.

ponytail: CadQuery (OpenCascade) would add NURBS and fillets but is several hundred MB with VTK; manifold3d is a few
MB and does every boolean here exactly. Fillets are chamfers.
"""

from __future__ import annotations

from typing import Any

import manifold3d as mf
import numpy as np

SEGMENTS = 24  # around a round (flute, arch, turned leg)


def box(lo, hi) -> mf.Manifold:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return mf.Manifold.cube((hi - lo).tolist()).translate(lo.tolist())


def _face(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return lo, hi, hi - lo


def grooves(lo, hi, count: int, axis: str = "vertical", width: float | None = None, depth: float | None = None, margin: float = 0.06) -> mf.Manifold:
    """The channels to subtract: `count` grooves across the face, evenly spaced inside a `margin` (share of the face)."""
    lo, hi, s = _face(lo, hi)
    depth = depth or min(0.004, 0.35 * s[2])
    along, across = (1, 0) if axis == "vertical" else (0, 1)
    width = width or min(0.006, 0.4 * s[across] / max(count, 1))
    cut = mf.Manifold()
    span = s[across] * (1 - 2 * margin)
    for i in range(count):
        c = lo[across] + s[across] * margin + span * (i + 0.5) / count
        g_lo, g_hi = lo.copy(), hi.copy()
        g_lo[across], g_hi[across] = c - width / 2, c + width / 2
        g_lo[along], g_hi[along] = lo[along] + s[along] * margin, hi[along] - s[along] * margin
        g_lo[2], g_hi[2] = hi[2] - depth, hi[2] + 1e-3
        cut += box(g_lo, g_hi)
    return cut


def flutes(lo, hi, count: int) -> mf.Manifold:
    """Half-round vertical ribs standing on the front face, side by side, to add."""
    lo, hi, s = _face(lo, hi)
    r = 0.5 * s[0] / count
    ribs = mf.Manifold()
    for i in range(count):
        rib = mf.Manifold.cylinder(s[1], r * 0.98, r * 0.98, SEGMENTS).rotate([-90, 0, 0])  # along +y
        ribs += rib.translate([lo[0] + (i + 0.5) * s[0] / count, lo[1], hi[2]])
    return ribs ^ box([lo[0], lo[1], hi[2] - 1e-4], [hi[0], hi[1], hi[2] + r + 1e-3])  # the front half only


def field(lo, hi, inset: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """The panel's inner field: inset from every edge by a share of the shorter side."""
    lo, hi, s = _face(lo, hi)
    m = inset * min(s[0], s[1])
    return np.array([lo[0] + m, lo[1] + m]), np.array([hi[0] - m, hi[1] - m])


def recessed(lo, hi, inset: float = 0.12, depth: float | None = None) -> mf.Manifold:
    """The pocket to subtract: the inner field sunk into the face, its rim chamfered."""
    lo, hi, s = _face(lo, hi)
    (fx0, fy0), (fx1, fy1) = field(lo, hi, inset)
    depth = depth or min(0.005, 0.4 * s[2])
    bevel = min(depth, 0.003)
    top = box([fx0 - bevel, fy0 - bevel, hi[2] - 1e-4], [fx1 + bevel, fy1 + bevel, hi[2] + 1e-3])
    bottom = box([fx0, fy0, hi[2] - depth], [fx1, fy1, hi[2] - depth + 1e-4])
    return mf.Manifold.batch_hull([top, bottom])


def raised(lo, hi, inset: float = 0.12, height: float = 0.004) -> mf.Manifold:
    """The field to add: standing `height` proud of the face, its edges chamfered."""
    lo, hi, _ = _face(lo, hi)
    (fx0, fy0), (fx1, fy1) = field(lo, hi, inset)
    b = min(height, 0.004)
    base = box([fx0, fy0, hi[2] - 1e-4], [fx1, fy1, hi[2] + 1e-4])
    top = box([fx0 + b, fy0 + b, hi[2] + height - 1e-4], [fx1 - b, fy1 - b, hi[2] + height])
    return mf.Manifold.batch_hull([base, top])


def arch_outline(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """A rectangle with a semicircular top (the round as wide as the rectangle), counter-clockwise."""
    r = (x1 - x0) / 2
    cx, top = (x0 + x1) / 2, max(y0 + 1e-3, y1 - r)
    arc = [(cx + r * np.cos(a), top + r * np.sin(a)) for a in np.linspace(0, np.pi, SEGMENTS)]
    return np.array([(x0, y0), (x1, y0), *arc[:0], *arc, (x0, y0)][:-1])


def arch(lo, hi, inset: float = 0.12, depth: float | None = None) -> mf.Manifold:
    """An arched field (round top) to subtract from the face."""
    lo, hi, s = _face(lo, hi)
    (fx0, fy0), (fx1, fy1) = field(lo, hi, inset)
    depth = depth or min(0.005, 0.4 * s[2])
    outline = arch_outline(fx0, fy0, fx1, fy1)
    return extrude(outline, depth + 1e-3).translate([0, 0, hi[2] - depth])


def chamfer(lo, hi, size: float = 0.002) -> mf.Manifold:
    """A board with its front edges bevelled (the hull of the full back and the inset front)."""
    lo, hi, s = _face(lo, hi)
    c = min(size, 0.3 * s[2], 0.2 * min(s[0], s[1]))
    back = box(lo, [hi[0], hi[1], hi[2] - c])
    front = box([lo[0] + c, lo[1] + c, hi[2] - c], [hi[0] - c, hi[1] - c, hi[2]])
    return mf.Manifold.batch_hull([back, front])


def extrude(outline, depth: float) -> mf.Manifold:
    """A 2-D outline (x, y points, counter-clockwise) extruded along +z by `depth`."""
    pts = np.asarray(outline, float)
    if _signed_area(pts) < 0:
        pts = pts[::-1]
    return mf.Manifold.extrude(mf.CrossSection([pts.tolist()]), depth)


def revolve(profile, segments: int = SEGMENTS) -> mf.Manifold:
    """A turned part: `profile` (radius, height) points revolved about the y axis (a turned leg, a knob)."""
    pts = np.asarray(profile, float)
    if _signed_area(pts) < 0:
        pts = pts[::-1]
    return mf.Manifold.revolve(mf.CrossSection([pts.tolist()]), segments).rotate([-90, 0, 0])


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


DECORATIONS = ("grooved", "fluted", "recessed_panel", "raised_panel", "arched")


def decorated(lo, hi, decoration: str, pattern: dict[str, Any] | None = None, region=None) -> tuple[mf.Manifold, dict[str, Any]]:
    """The board with its decoration as real geometry, and what was made (count, axis, whether from the photo).
    `region`: (lo, hi) of the face that carries it (flutes on the lower band of a door), default the whole face."""
    board_lo, board_hi = np.asarray(lo, float), np.asarray(hi, float)
    lo, hi, s = _face(*(region if region is not None else (lo, hi)))
    pattern = pattern or {}
    solid = chamfer(board_lo, board_hi) if decoration in DECORATIONS else box(board_lo, board_hi)
    made: dict[str, Any] = {"decoration": decoration, "method": "geometry"}
    if decoration == "grooved":
        axis = pattern.get("axis") or ("vertical" if s[1] >= s[0] else "horizontal")
        across = 0 if axis == "vertical" else 1
        count = int(pattern.get("count") or max(2, round(s[across] / 0.05)))
        solid -= grooves(lo, hi, count, axis)
        made.update(count=count, axis=axis, fromPhoto=bool(pattern.get("count")))
    elif decoration == "fluted":
        count = int(pattern.get("count") or max(3, round(s[0] / 0.03)))
        solid += flutes(lo, hi, count)
        made.update(count=count, axis="vertical", fromPhoto=bool(pattern.get("count")))
    elif decoration == "recessed_panel":
        solid -= recessed(lo, hi)
    elif decoration == "raised_panel":
        solid += raised(lo, hi)
    elif decoration == "arched":
        solid -= arch(lo, hi)
    else:
        made["method"] = "none"
    return solid, made


def arrays(solid: mf.Manifold, lo, hi) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flat-shaded triangles of the solid: positions, normals, front-projected UVs (0..1 over the part's box), indices."""
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, float)[:, :3]
    tris = np.asarray(mesh.tri_verts, np.int64)
    pos = verts[tris].reshape(-1, 3)
    a, b, c = pos[0::3], pos[1::3], pos[2::3]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    nrm = np.repeat(n, 3, axis=0)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    span = np.maximum(hi - lo, 1e-9)
    uv = np.column_stack([np.clip((pos[:, 0] - lo[0]) / span[0], 0, 1), np.clip(1 - (pos[:, 1] - lo[1]) / span[1], 0, 1)])
    return pos, nrm, uv, np.arange(len(pos))
