"""Converted plan (metres) → DXF: room outlines, wall bodies, openings and dimension notes, one layer each.

DXF y points up, the plan's z points down the image, so z is negated (the drawing reads like the upload).
"""

from __future__ import annotations

import io
from typing import Any

import ezdxf
import numpy as np


def export_plan_dxf(payload: dict[str, Any]) -> bytes:
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    for name, color in (("ROOMS", 3), ("WALLS", 7), ("DOORS", 1), ("WINDOWS", 5), ("DIMENSIONS", 2)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()

    def xy(p):
        return float(p[0]), -float(p[1])

    for w in payload.get("walls") or []:
        msp.add_lwpolyline(_band(xy(w["start"]), xy(w["end"]), w.get("thickness") or 0.1), close=True, dxfattribs={"layer": "WALLS"})
    for o in payload.get("openings") or []:
        layer = "DOORS" if o.get("kind") == "door" else "WINDOWS"
        msp.add_line(xy(o["start"]), xy(o["end"]), dxfattribs={"layer": layer})
    for z in payload.get("zones") or payload.get("rooms") or []:
        msp.add_lwpolyline([xy(p) for p in z["polygon"]], close=True, dxfattribs={"layer": "ROOMS"})
        d = z.get("dimensions") or {}
        at = xy(z.get("label_point") or np.mean(z["polygon"], axis=0))
        approx = "" if z.get("dimension_source") == "printed" else "~"
        lines = [z["name"]]
        if d.get("width"):
            lines.append(f"{approx}{d['width']:.2f} x {d['length']:.2f} m")
        if z.get("area_m2"):
            lines.append(f"{approx}{z['area_m2']:.2f} m2")
        msp.add_mtext("\\P".join(lines), dxfattribs={"layer": "DIMENSIONS", "char_height": 0.15, "insert": at, "attachment_point": 5})
    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode("utf-8")


def _band(a, b, thickness: float) -> list[tuple[float, float]]:
    """A wall segment as its rectangular body."""
    d = np.subtract(b, a)
    n = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1) * thickness / 2
    return [tuple(np.add(a, n)), tuple(np.add(b, n)), tuple(np.subtract(b, n)), tuple(np.subtract(a, n))]
