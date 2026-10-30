from __future__ import annotations

from typing import Any

from app.services.plan.geometry_builder import build_room_glb


def export_room_glb(payload: dict[str, Any]) -> bytes:
    return build_room_glb(payload)


def step_unsupported_reason() -> str:
    return (
        "STEP export needs a CAD kernel (CadQuery/OpenCascade) and watertight solids. "
        "Furniture loaded from GLB is triangle mesh, not parametric CAD. Download GLB instead."
    )
