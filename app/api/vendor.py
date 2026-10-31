from __future__ import annotations

from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.services.astra_service import AstraError, AstraService
from app.services.auth import create_confirmed_user, create_vendor, user_id_from_bearer
from app.services.carpet_extract import CarpetNotFound, extract_carpet
from app.services.generation_prompt import astra_prompt, meshy_texture_prompt
from app.services.cdn import Cdn, CdnError
from app.schemas.vendor import GenerationOptions
from app.services.meshy_service import MeshyError, MeshyService
from app.services.quota import generation_usage, use_generation

router = APIRouter(prefix="/api/vendor", tags=["vendor"])

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
Provider = Literal["meshy", "astra"]


class GenerateAccepted(BaseModel):
    task_id: str
    provider: Provider
    status: str = "PENDING"
    remaining: int | None = None  # tries left for this provider after this call


class GenerateReady(BaseModel):
    task_id: str
    provider: Provider
    status: Literal["SUCCEEDED"] = "SUCCEEDED"
    glb_url: str
    remaining: int | None = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int = Field(ge=0, le=100)
    glb_url: str | None = None
    error: str | None = None
    retry_after_seconds: float | None = None


def sniff_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def get_meshy(request: Request) -> MeshyService:
    return request.app.state.meshy


def get_cdn(request: Request) -> Cdn:
    return request.app.state.cdn


def get_astra(request: Request) -> AstraService:
    return request.app.state.astra


class RegisterBody(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    # "customer" shops and plans; "vendor" sells (approved on signup; the admin can still suspend).
    kind: Literal["vendor", "customer"] = "vendor"
    name: str = Field(default="", max_length=120)  # company / studio name (vendors)
    location: str = Field(default="", max_length=200)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lng: float | None = Field(default=None, ge=-180, le=180)
    full_name: str = Field(default="", max_length=120)  # the person signing up; kept in auth metadata


class RegisterResponse(BaseModel):
    user_id: str
    status: Literal["active"]


@router.post("/register", response_model=RegisterResponse)
async def register_vendor(body: RegisterBody, request: Request) -> RegisterResponse:
    settings = get_settings()
    vendor = body.kind == "vendor"
    if vendor and (not body.name.strip() or not body.location.strip()):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Company name and location are required.")
    email = body.email.strip().lower()
    client = request.app.state.auth_http
    user_id = await create_confirmed_user(
        email,
        body.password,
        {"role": body.kind, "name": body.name.strip(), "location": body.location.strip(), "full_name": body.full_name.strip()},
        settings,
        client,
    )
    if vendor:
        # The vendor row is written here (service role); vendors cannot set their own status
        # (see supabase/vendor_approval.sql), so only the admin can suspend them later.
        await create_vendor(
            {"id": user_id, "name": body.name.strip(), "location": body.location.strip(), "full_name": body.full_name.strip(), "email": email, "lat": body.lat, "lng": body.lng},
            settings,
            client,
        )
    return RegisterResponse(user_id=user_id, status="active")


async def current_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    return await user_id_from_bearer(authorization, get_settings(), request.app.state.auth_http)


class QuotaResponse(BaseModel):
    limit: int
    remaining: dict[str, int]


@router.get("/generation-quota", response_model=QuotaResponse)
async def generation_quota(request: Request, user_id: Annotated[str, Depends(current_user)]) -> QuotaResponse:
    settings = get_settings()
    used = await generation_usage(user_id, settings, request.app.state.auth_http)
    return QuotaResponse(limit=settings.generation_limit, remaining={p: max(settings.generation_limit - n, 0) for p, n in used.items()})


async def take_try(request: Request, user_id: str, provider: Provider, settings: Settings) -> int:
    """Reserve one generation try; 429 when the vendor has used them all. Returns tries left."""
    used = await use_generation(user_id, provider, 1, settings, request.app.state.auth_http)
    if used is None:
        name = "GPT Astra" if provider == "astra" else "Meshy"
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, f"You have used all {settings.generation_limit} {name} generations for this account.")
    return settings.generation_limit - used


@router.post("/generate")
async def generate_model(
    request: Request,
    meshy: Annotated[MeshyService, Depends(get_meshy)],
    astra: Annotated[AstraService, Depends(get_astra)],
    cdn: Annotated[Cdn, Depends(get_cdn)],
    settings: Annotated[Settings, Depends(get_settings)],
    user_id: Annotated[str, Depends(current_user)],
    image: Annotated[UploadFile | None, File()] = None,
    image_url: Annotated[str | None, Form()] = None,
    provider: Annotated[Provider, Form()] = "meshy",
    model_id: Annotated[str | None, Form()] = None,
    should_texture: Annotated[str, Form()] = "true",
    category: Annotated[str, Form(max_length=40)] = "furniture",
    title: Annotated[str, Form(max_length=200)] = "",
) -> GenerateAccepted | GenerateReady:
    raw = await image.read() if image is not None else b""
    public_url = (image_url or "").strip() or None
    if raw:
        if len(raw) > settings.max_image_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Images must be 20 MB or smaller.")
        sniffed = sniff_image_mime(raw)
        declared = (image.content_type or "").split(";")[0].strip().lower() if image else ""
        if sniffed not in ALLOWED_IMAGE_TYPES or (declared and declared not in ALLOWED_IMAGE_TYPES and declared != "application/octet-stream"):
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a PNG, JPG or WEBP image.")
        mime = sniffed or "image/png"
    elif public_url:
        mime = "image/png"
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Upload a photo or provide image_url.")
    object_id = model_id or uuid4().hex
    textured = should_texture.strip().lower() not in {"0", "false", "no"}

    if provider == "astra":
        if not settings.astra_configured:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not configured.")
        if not raw:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Astra needs the photo file, not only a URL.")
        remaining = await take_try(request, user_id, "astra", settings)
        try:
            glb = await astra.generate_glb(raw, mime, astra_prompt(category, title))
            url = cdn.upload(f"{user_id}/{object_id}.glb", glb, "model/gltf-binary")
        except (AstraError, CdnError) as exc:
            # ponytail: a failed call gives the try back; refunds are best-effort, not transactional.
            await use_generation(user_id, "astra", -1, settings, request.app.state.auth_http)
            raise HTTPException(exc.status_code, exc.message) from exc
        return GenerateReady(task_id=object_id, provider="astra", glb_url=url, remaining=remaining)

    if not settings.meshy_configured:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "MESHY_API_KEY is not configured.")
    remaining = await take_try(request, user_id, "meshy", settings)
    try:
        task_id = await meshy.create_image_to_3d_task(
            GenerationOptions(should_texture=textured, enable_pbr=False),
            image=None if public_url else raw,
            mime_type=mime,
            image_url=public_url,
            texture_prompt=meshy_texture_prompt(category, title),
        )
    except MeshyError as exc:
        await use_generation(user_id, "meshy", -1, settings, request.app.state.auth_http)  # Meshy refused: give the try back
        raise HTTPException(exc.status_code, exc.message) from exc
    request.app.state.pending_cdn[task_id] = {"user_id": user_id, "model_id": object_id}
    return GenerateAccepted(task_id=task_id, provider="meshy", remaining=remaining)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(
    task_id: str,
    request: Request,
    meshy: Annotated[MeshyService, Depends(get_meshy)],
    cdn: Annotated[Cdn, Depends(get_cdn)],
    user_id: Annotated[str, Depends(current_user)],
) -> TaskStatusResponse:
    try:
        task, retry_after = await meshy.get_image_to_3d_task(task_id)
    except MeshyError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc

    status_value = str(task.get("status") or "IN_PROGRESS")
    progress = int(task.get("progress") or 0)
    if status_value in {"FAILED", "CANCELED", "EXPIRED"}:
        err = task.get("task_error") or task.get("error") or "Generation failed."
        if isinstance(err, dict):
            err = err.get("message") or str(err)
        return TaskStatusResponse(
            task_id=task_id,
            status="FAILED" if status_value == "EXPIRED" else status_value,
            progress=progress,
            error=str(err),
        )
    if status_value != "SUCCEEDED":
        return TaskStatusResponse(
            task_id=task_id,
            status=status_value,
            progress=progress,
            retry_after_seconds=retry_after or 5,
        )

    pending = request.app.state.pending_cdn.get(task_id) or {}
    if pending.get("user_id") and pending["user_id"] != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found.")
    owner = pending.get("user_id") or user_id
    object_id = pending.get("model_id") or task_id
    model_urls = task.get("model_urls") or {}
    glb_url = model_urls.get("glb")
    if not glb_url:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Meshy finished but did not return a GLB.")
    dest = request.app.state.storage.models_dir / f"{task_id}.glb"
    try:
        await meshy.download_file(glb_url, dest, get_settings().max_model_bytes)
        url = cdn.upload(f"{owner}/{object_id}.glb", dest.read_bytes(), "model/gltf-binary")
    except (MeshyError, CdnError) as exc:
        raise HTTPException(getattr(exc, "status_code", 502), str(getattr(exc, "message", exc))) from exc
    request.app.state.pending_cdn.pop(task_id, None)
    return TaskStatusResponse(task_id=task_id, status="SUCCEEDED", progress=100, glb_url=url)


MAX_CARPET_PHOTO = 15 * 1024 * 1024


@router.post("/carpets/extract")
async def extract_carpet_image(
    _user: Annotated[str, Depends(current_user)],
    image: Annotated[UploadFile, File()],
    width_m: Annotated[float | None, Form(gt=0, le=50)] = None,
    length_m: Annotated[float | None, Form(gt=0, le=50)] = None,
) -> Response:
    """Photo of a carpet → PNG of just the carpet, straightened top-down at its real aspect ratio."""
    data = await image.read(MAX_CARPET_PHOTO + 1)
    if len(data) > MAX_CARPET_PHOTO:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Photo is larger than 15 MB.")
    try:
        png = await run_in_threadpool(extract_carpet, data, width_m, length_m)
    except CarpetNotFound as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return Response(content=png, media_type="image/png")
