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


# ---- semantic primitives (services/assembly.py) ----

MOULDING_HEIGHT = 0.006  # a raised moulding stands this proud of its face
MOULDING_DEPTH = 0.004  # a recessed one is sunk this deep
TILE = 0.6  # metres one material texture covers (UVs are in real size: no stretching between faces)


def frame(lo, hi, opening) -> mf.Manifold:
    """A frame with a real opening through it (a mirror or glass door's frame): the board minus the opening."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    x0, y0, x1, y1 = opening
    return box(lo, hi) - box([x0, y0, lo[2] - 1e-3], [x1, y1, hi[2] + 1e-3])


def moulding_ring(lo, hi, rect, border: float) -> mf.Manifold:
    """A rectangular ring on the front face: `rect` (fractions of the face, origin top left), `border` (share of the
    face's shorter side), MOULDING_HEIGHT thick, sitting on the face (to add) or into it (to cut)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    s = hi - lo
    x0, x1 = lo[0] + rect[0] * s[0], lo[0] + rect[2] * s[0]
    y1, y0 = hi[1] - rect[1] * s[1], hi[1] - rect[3] * s[1]
    b = max(0.006, border * min(s[0], s[1]))
    if x1 - x0 <= 2 * b or y1 - y0 <= 2 * b:
        return mf.Manifold()
    outer = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    inner = [(x0 + b, y0 + b), (x0 + b, y1 - b), (x1 - b, y1 - b), (x1 - b, y0 + b)]  # clockwise: the hole
    return mf.Manifold.extrude(mf.CrossSection([outer, inner]), MOULDING_HEIGHT)


def with_mouldings(solid: mf.Manifold, lo, hi, mouldings: list[dict[str, Any]]) -> tuple[mf.Manifold, list[dict[str, Any]]]:
    """Every moulding of the face as geometry: raised ones added on the face, recessed ones cut into it."""
    made = []
    front = float(np.asarray(hi, float)[2])
    for m in mouldings or []:
        ring = moulding_ring(lo, hi, m["rect"], float(m.get("border") or 0.04))
        if ring.is_empty():
            continue
        if m.get("profile") == "recessed":
            solid = solid - ring.translate([0, 0, front - MOULDING_DEPTH])
        else:
            solid = solid + ring.translate([0, 0, front - 1e-4])
        made.append({"rect": [round(v, 3) for v in m["rect"]], "profile": m.get("profile") or "raised", "method": "geometry"})
    return solid, made


def face_uv(pos: np.ndarray, nrm: np.ndarray, tile: float = TILE) -> np.ndarray:
    """UVs by the face each triangle lies on, in real size (metres / tile): front and back from x / y, left and right
    from z / y, top and bottom from x / z; each oriented as seen from outside, so nothing is mirrored or stretched."""
    a = np.abs(nrm)
    major = np.argmax(a, axis=1)
    sign = np.sign(nrm[np.arange(len(nrm)), major])
    u, v = np.empty(len(pos)), np.empty(len(pos))
    z, y, x = pos[:, 2], pos[:, 1], pos[:, 0]
    fb, lr, tb = major == 2, major == 0, major == 1
    u[fb], v[fb] = np.where(sign[fb] > 0, x[fb], -x[fb]), y[fb]  # back faces read from behind
    u[lr], v[lr] = np.where(sign[lr] > 0, -z[lr], z[lr]), y[lr]
    u[tb], v[tb] = x[tb], np.where(sign[tb] > 0, -z[tb], z[tb])
    return np.column_stack([u / tile, v / tile])


def mesh_arrays(solid: mf.Manifold, tile: float = TILE) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flat-shaded triangles with face-specific real-size UVs: positions, normals, UVs, indices."""
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, float)[:, :3]
    tris = np.asarray(mesh.tri_verts, np.int64)
    pos = verts[tris].reshape(-1, 3)
    a, b, c = pos[0::3], pos[1::3], pos[2::3]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    nrm = np.repeat(n, 3, axis=0)
    return pos, nrm, face_uv(pos, nrm, tile), np.arange(len(pos))


# ---- hardware ----

def bar_handle(lo, hi, orientation: str, diameter: float, profile: str = "round", posts: int = 2) -> mf.Manifold:
    """A bar handle inside its box: a grip `diameter` thick along the box's long axis at the front of the box, standing
    on `posts` mounting posts that reach back to the mounting face (the box's back). Round or rectangular grip."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    s = hi - lo
    axis = 1 if orientation == "vertical" else 0
    length = s[axis]
    d = min(diameter, s[2], s[1 - axis] if axis == 1 else s[1])
    centre = (lo + hi) / 2
    grip_z = hi[2] - d / 2
    if profile == "round":
        grip = mf.Manifold.cylinder(length, d / 2, d / 2, SEGMENTS)  # along +z: turn it onto the axis
        grip = grip.rotate([-90, 0, 0]) if axis == 1 else grip.rotate([0, 90, 0])
        start = [centre[0], lo[1], grip_z] if axis == 1 else [lo[0], centre[1], grip_z]
        grip = grip.translate(start)
    else:
        half = [d / 2, d / 2, d / 2]
        a = np.array([centre[0] - half[0], lo[1], grip_z - half[2]]) if axis == 1 else np.array([lo[0], centre[1] - half[1], grip_z - half[2]])
        b = np.array([centre[0] + half[0], hi[1], grip_z + half[2]]) if axis == 1 else np.array([hi[0], centre[1] + half[1], grip_z + half[2]])
        grip = box(a, b)
    out = grip
    post_d = max(0.006, 0.6 * d)
    for i in range(max(posts, 1) if length > 0.08 else 1):
        t = 0.5 if posts <= 1 or length <= 0.08 else (0.1 + 0.8 * i / (posts - 1))
        at = lo[axis] + t * length
        post = mf.Manifold.cylinder(grip_z - lo[2], post_d / 2, post_d / 2, SEGMENTS)
        pos = [centre[0], at, lo[2]] if axis == 1 else [at, centre[1], lo[2]]
        out = out + post.translate(pos)
    return out


def knob(lo, hi) -> mf.Manifold:
    """A round knob in its box: a stem from the mounting face and a rounded cap filling the box's front."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    s = hi - lo
    r = 0.5 * min(s[0], s[1])
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    cap_depth = min(s[2] * 0.6, r * 1.2)
    stem = mf.Manifold.cylinder(s[2] - cap_depth / 2, 0.35 * r, 0.35 * r, SEGMENTS).translate([cx, cy, lo[2]])
    cap = mf.Manifold.sphere(1.0, SEGMENTS).scale([r, r, cap_depth / 2]).translate([cx, cy, hi[2] - cap_depth / 2])
    return stem + cap


def recess(lo, hi, depth: float = 0.008) -> mf.Manifold:
    """A finger recess to cut from a door: the box's footprint sunk `depth` into the face at the box's back."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return box([lo[0], lo[1], lo[2] - depth], [hi[0], hi[1], lo[2] + 1e-3])
