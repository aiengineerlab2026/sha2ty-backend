from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import math
import re
from functools import partial
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.auth import user_id_from_bearer
from app.services.cdn import Cdn, CdnError
from app.services.plan.cad_converter import convert_to_glb
from app.services.plan.cad_exporter import export_plan_dxf
from app.services.plan.geometry_builder import DEFAULT_HEIGHT, DEFAULT_THICKNESS, build_room_glb, rectangular_room, room_extents
from app.services.plan.image_processor import plan_from_image, plan_from_state
from app.services.plan.vision_analysis import analyze_plan
from app.services.plan.placement import validate_placement
from app.services.plan.plan_exporter import export_room_glb, step_unsupported_reason
from app.services.plan_store import PlanStore

logger = logging.getLogger(__name__)
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
    # Measured once on upload (they need the drawing) and carried through every correction.
    scale_refs: list[dict[str, Any]] = Field(default_factory=list)
    fixtures_px: list[dict[str, Any]] = Field(default_factory=list)
    analysis: dict[str, Any] | None = None  # the vision model's validated JSON
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
    require_3d: bool = True,
) -> dict[str, Any]:
    return await store(request).catalog(q, vendor_id, min(limit, 100), max(offset, 0), str(id) if id else None, require_3d)


@router.post("/from-dimensions")
async def from_dimensions(body: DimensionsBody) -> dict[str, Any]:
    payload = rectangular_room(body.width, body.depth, body.height, body.wall_thickness, body.name)
    width, depth, height = room_extents(payload)
    payload["extents"] = {"width": width, "depth": depth, "height": height}
    return payload


@router.post("/convert-to-3d")
async def convert_to_3d(
    request: Request,
    image: Annotated[UploadFile, File()],
    height: Annotated[float, Form()] = DEFAULT_HEIGHT,
    wall_thickness: Annotated[float, Form()] = DEFAULT_THICKNESS,
    debug: Annotated[str, Form()] = "false",
) -> StreamingResponse:
    """Computer vision only (OpenCV + OCR + geometry), no LLM: the plan plus GLB and DXF download URLs."""
    return await _convert(request, image, height, wall_thickness, None, None, debug, vision=False)


@router.get("/files/{name}")
async def converted_file(name: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{24}\.(glb|dxf)", name):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")
    path = get_settings().storage_dir / "plan-exports" / name
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")
    media = "model/gltf-binary" if name.endswith(".glb") else "image/vnd.dxf"
    return FileResponse(path, media_type=media, filename=f"floor-plan{path.suffix}")


@router.post("/from-image")
async def from_image(
    request: Request,
    image: Annotated[UploadFile, File()],
    height: Annotated[float, Form()] = DEFAULT_HEIGHT,
    wall_thickness: Annotated[float, Form()] = DEFAULT_THICKNESS,
    width: Annotated[float | None, Form()] = None,
    depth: Annotated[float | None, Form()] = None,
    debug: Annotated[str, Form()] = "false",
) -> StreamingResponse:
    return await _convert(request, image, height, wall_thickness, width, depth, debug, vision=True)


async def _convert(request: Request, image: UploadFile, height, wall_thickness, width, depth, debug, vision: bool) -> StreamingResponse:
    raw = await image.read()
    if len(raw) > get_settings().max_image_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Images must be 20 MB or smaller.")
    mime = (image.content_type or "").split(";")[0].strip().lower()
    if mime not in ALLOWED:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a PNG or JPG floor plan.")
    # The vision model reads the whole layout first (cached per image); OpenCV then measures and refines.
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            size = probe.size
    except OSError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not read this image.") from exc
    settings = get_settings()
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()

    def stage(name: str, percent: int) -> None:  # called from the conversion thread
        loop.call_soon_threadsafe(events.put_nowait, {"stage": name, "progress": percent})

    async def stream():
        """NDJSON progress lines {"stage", "progress"} (0–100), then {"success": true, "plan": …} or {"error": …}.

        The model call has no progress of its own: its percentage follows the elapsed time against the
        usual latency (OPENAI_PLAN_EXPECTED_SECONDS) and never passes 70 until it answers. The later
        stages are reported by the conversion as it reaches them.
        """
        yield json.dumps({"stage": "Analyzing", "progress": 2}) + "\n"
        task = asyncio.create_task(analyze_plan(raw, mime, size, settings, request.app.state.openai_http) if vision else asyncio.sleep(0))
        started = loop.time()
        while not task.done():
            await asyncio.wait({task}, timeout=1.0)
            elapsed = loop.time() - started
            yield json.dumps({"stage": "Analyzing", "progress": int(2 + 68 * (1 - math.exp(-elapsed / settings.openai_plan_expected_seconds)))}) + "\n"
        analysis = task.result()
        conversion = asyncio.create_task(
            run_in_threadpool(
                partial(
                    plan_from_image, raw, wall_height=height, wall_thickness=wall_thickness, width_m=width, depth_m=depth,
                    debug=str(debug).lower() in {"1", "true", "yes"}, analysis=analysis, on_stage=stage,
                )
            )
        )
        while not conversion.done() or not events.empty():
            getter = asyncio.create_task(events.get())
            done, _ = await asyncio.wait({getter, conversion}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield json.dumps(getter.result()) + "\n"
            else:
                getter.cancel()
        try:
            payload = conversion.result()
        except ValueError as exc:
            yield json.dumps({"error": str(exc), "status": 400}) + "\n"
            return
        except Exception as exc:  # noqa: BLE001 - reported to the user, logged below
            logger.exception("Floor plan conversion failed")
            yield json.dumps({"error": f"Could not convert this floor plan ({exc}).", "status": 500}) + "\n"
            return
        if vision and analysis is None and settings.openai_plan_model and settings.astra_configured:
            payload["warnings"] = ["The vision model was unavailable, so this plan was read with OpenCV only.", *payload["warnings"]]
        w, d, h = room_extents(payload)
        payload["extents"] = {"width": w, "depth": d, "height": h}
        logger.info("Plan 3D extents %.2f × %.2f × %.2f m, %d walls, %d floors, %d fixtures", w, d, h, len(payload["walls"]), len(payload["zones"]), len(payload.get("fixtures") or []))
        if not vision and payload["walls"]:
            payload.update(await run_in_threadpool(_write_exports, raw, payload))
        yield json.dumps({"stage": "Done", "progress": 100, "success": True, "plan": payload}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


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


def _write_exports(raw: bytes, payload: dict[str, Any]) -> dict[str, str]:
    """GLB + DXF next to each other, named by the upload's hash (same image → same files)."""
    folder = get_settings().storage_dir / "plan-exports"
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(raw).hexdigest()[:24]
    (folder / f"{key}.glb").write_bytes(build_room_glb(payload))
    (folder / f"{key}.dxf").write_bytes(export_plan_dxf(payload))
    return {"glb_url": f"/api/plans/files/{key}.glb", "dxf_url": f"/api/plans/files/{key}.dxf"}


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
