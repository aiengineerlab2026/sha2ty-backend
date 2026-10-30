from __future__ import annotations

from functools import partial
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.auth import user_id_from_bearer
from app.services.cdn import Cdn, CdnError
from app.services.plan.cad_converter import convert_to_glb
from app.services.plan.geometry_builder import DEFAULT_HEIGHT, DEFAULT_THICKNESS, build_room_glb, rectangular_room, room_extents
from app.services.plan.image_processor import plan_from_image, plan_from_state
from app.services.plan.placement import validate_placement
from app.services.plan.plan_exporter import export_room_glb, step_unsupported_reason
from app.services.plan_store import PlanStore

router = APIRouter(prefix="/api/plans", tags=["plans"])
# Building a plan (from dimensions / an image, correction, validation) is public so guests can design;
# saving, listing, loading and exporting plans require a signed-in user.
ALLOWED = {"image/png", "image/jpeg", "image/jpg"}


class DimensionsBody(BaseModel):
    width: float = Field(gt=0, le=200)
    depth: float = Field(gt=0, le=200)
    height: float = Field(default=DEFAULT_HEIGHT, gt=0, le=20)
    wall_thickness: float = Field(default=DEFAULT_THICKNESS, gt=0, le=2)
    name: str = "Room"


class PlanSaveBody(BaseModel):
    name: str = "Untitled plan"
    source: Literal["image", "dimensions"] = "dimensions"
    source_image_url: str | None = None
    geometry: dict[str, Any]
    items: list[dict[str, Any]] = Field(default_factory=list)
    camera: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    allow_invalid: bool = False


class CorrectionBody(BaseModel):
    """Editable pixel-space plan from the correction view. Rooms are recomputed from these walls."""

    image_size: dict[str, int]
    thickness_px: float | None = None
    walls_px: list[dict[str, Any]]
    openings_px: list[dict[str, Any]] = Field(default_factory=list)
    labels: list[dict[str, Any]] = Field(default_factory=list)
    zone_lines: list[dict[str, Any]] = Field(default_factory=list)
    straight_lines: str | None = None
    drawing_unit: str | None = None
    image_scale: float = 1.0
    meters_per_pixel: float | None = Field(default=None, gt=0)
    scale_confirmed: bool = False
    height: float = Field(default=DEFAULT_HEIGHT, gt=0, le=20)
    wall_thickness: float = Field(default=DEFAULT_THICKNESS, gt=0, le=2)

    def run(self) -> dict[str, Any]:
        state = self.model_dump(exclude={"meters_per_pixel", "scale_confirmed", "height", "wall_thickness"})
        return plan_from_state(
            state,
            wall_height=self.height,
            wall_thickness=self.wall_thickness,
            meters_per_pixel=self.meters_per_pixel,
            scale_confirmed=self.scale_confirmed,
        )


class ValidateBody(BaseModel):
    geometry: dict[str, Any]
    item: dict[str, Any]
    others: list[dict[str, Any]] = Field(default_factory=list)


async def current_user(request: Request, authorization: Annotated[str | None, Header()] = None) -> str:
    return await user_id_from_bearer(authorization, get_settings(), request.app.state.auth_http)


def store(request: Request) -> PlanStore:
    return PlanStore(get_settings(), request.app.state.auth_http)


@router.get("/models")  # public: the home page shows the catalogue to visitors too
async def list_models(
    request: Request,
    q: str = "",
    vendor_id: str | None = None,
    limit: int = 40,
    offset: int = 0,
    id: UUID | None = None,
) -> dict[str, Any]:
    return await store(request).catalog(q, vendor_id, min(limit, 100), max(offset, 0), str(id) if id else None)


@router.post("/from-dimensions")
async def from_dimensions(body: DimensionsBody) -> dict[str, Any]:
    payload = rectangular_room(body.width, body.depth, body.height, body.wall_thickness, body.name)
    width, depth, height = room_extents(payload)
    payload["extents"] = {"width": width, "depth": depth, "height": height}
    return payload


@router.post("/from-image")
async def from_image(
    image: Annotated[UploadFile, File()],
    height: Annotated[float, Form()] = DEFAULT_HEIGHT,
    wall_thickness: Annotated[float, Form()] = DEFAULT_THICKNESS,
    width: Annotated[float | None, Form()] = None,
    depth: Annotated[float | None, Form()] = None,
    debug: Annotated[str, Form()] = "false",
) -> dict[str, Any]:
    raw = await image.read()
    if len(raw) > get_settings().max_image_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Images must be 20 MB or smaller.")
    mime = (image.content_type or "").split(";")[0].strip().lower()
    if mime not in ALLOWED:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a PNG or JPG floor plan.")
    try:
        payload = await run_in_threadpool(
            partial(
                plan_from_image,
                raw,
                wall_height=height,
                wall_thickness=wall_thickness,
                width_m=width,
                depth_m=depth,
                debug=str(debug).lower() in {"1", "true", "yes"},
            )
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Could not convert this floor plan ({exc}).") from exc
    width, depth, h = room_extents(payload)
    payload["extents"] = {"width": width, "depth": depth, "height": h}
    return payload


@router.post("/segment")
async def resegment(body: CorrectionBody) -> dict[str, Any]:
    """Preview: rooms, zones, scale and validation for edited walls. Never blocks."""
    return await run_in_threadpool(body.run)


@router.post("/from-geometry")
async def from_geometry(body: CorrectionBody) -> dict[str, Any]:
    """Build the 3D apartment. Refuses (422) while validation has blocking errors: no fallback box."""
    payload = await run_in_threadpool(body.run)
    if not payload["validation"]["ok"]:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {"message": "Fix these problems before generating 3D.", "errors": payload["validation"]["errors"]},
        )
    width, depth, h = room_extents(payload)
    payload["extents"] = {"width": width, "depth": depth, "height": h}
    return payload


@router.post("/validate")
async def validate(body: ValidateBody) -> dict[str, Any]:
    geometry = body.geometry
    height = float((geometry.get("height") or DEFAULT_HEIGHT))
    warnings = validate_placement(body.item, geometry.get("rooms") or [], geometry.get("walls") or [], body.others, height)
    return {"warnings": warnings, "valid": len(warnings) == 0}


@router.get("")
async def list_plans(request: Request, user_id: Annotated[str, Depends(current_user)]) -> dict[str, Any]:
    return {"plans": await store(request).list_plans(user_id)}


@router.post("")
async def create_plan(request: Request, user_id: Annotated[str, Depends(current_user)], body: PlanSaveBody) -> dict[str, Any]:
    _reject_invalid(body)
    row = {
        "owner_id": user_id,
        "name": body.name,
        "source": body.source,
        "source_image_url": body.source_image_url,
        "geometry": body.geometry,
        "items": body.items,
        "camera": body.camera,
        "warnings": body.warnings,
    }
    return await store(request).insert_plan(row)


@router.get("/{plan_id}")
async def get_plan(plan_id: str, request: Request, user_id: Annotated[str, Depends(current_user)]) -> dict[str, Any]:
    return await store(request).get_plan(plan_id, user_id)


@router.put("/{plan_id}")
async def update_plan(
    plan_id: str,
    request: Request,
    user_id: Annotated[str, Depends(current_user)],
    body: PlanSaveBody,
) -> dict[str, Any]:
    _reject_invalid(body)
    return await store(request).update_plan(
        plan_id,
        user_id,
        {
            "name": body.name,
            "source": body.source,
            "source_image_url": body.source_image_url,
            "geometry": body.geometry,
            "items": body.items,
            "camera": body.camera,
            "warnings": body.warnings,
        },
    )


@router.delete("/{plan_id}")
async def delete_plan(plan_id: str, request: Request, user_id: Annotated[str, Depends(current_user)]) -> dict[str, str]:
    await store(request).delete_plan(plan_id, user_id)
    return {"status": "deleted"}


@router.post("/{plan_id}/export")
async def export_plan(
    plan_id: str,
    request: Request,
    user_id: Annotated[str, Depends(current_user)],
    format: Literal["glb", "step"] = "glb",
) -> Response:
    if format == "step":
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, step_unsupported_reason())
    plan = await store(request).get_plan(plan_id, user_id)
    glb = export_room_glb(plan.get("geometry") or {})
    cdn: Cdn = request.app.state.cdn
    try:
        url = cdn.upload(f"{user_id}/plans/{plan_id}/room.glb", glb, "model/gltf-binary")
    except CdnError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    await store(request).update_plan(plan_id, user_id, {"glb_url": url})
    return Response(content=glb, media_type="model/gltf-binary", headers={"X-Cdn-Url": url, "Content-Disposition": 'attachment; filename="plan-room.glb"'})


@router.post("/convert-cad")
async def convert_cad(
    request: Request,
    user_id: Annotated[str, Depends(current_user)],
    file: Annotated[UploadFile, File()],
) -> dict[str, str]:
    raw = await file.read()
    cache = get_settings().storage_dir / "cad-cache"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        glb, how = convert_to_glb(raw, file.filename or "model.bin", cache)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    cdn: Cdn = request.app.state.cdn
    url = cdn.upload(f"{user_id}/converted/{uuid4().hex}.glb", glb, "model/gltf-binary")
    return {"glb_url": url, "via": how}


def _reject_invalid(body: PlanSaveBody) -> None:
    if body.allow_invalid:
        return
    blocking = [w for w in body.warnings if w]
    item_warnings = [w for item in body.items for w in (item.get("warnings") or [])]
    if blocking or item_warnings:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "This plan has placement warnings. Fix them or save with allow_invalid=true.",
        )
