from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.events import bus
from app.domain.errors import DomainError
from app.domain.events import GenerationFailed, GenerationStarted, UserRegistered
from app.domain.generation import Provider, QuotaExhausted, remaining as tries_left
from app.domain.vendor import Vendor
from app.services.astra_service import AstraError, AstraService
from app.services.auth import create_confirmed_user, user_id_from_bearer
from app.services.carpet_extract import CarpetNotFound, extract_carpet
from app.schemas.product_details import ProductDetails
from app.services import vision
from app.services.generation_prompt import astra_prompt, meshy_texture_prompt
from app.services.product_details import extract_product_details, in_language
from app.services.cdn import Cdn, CdnError
from app.schemas.vendor import GenerationOptions
from app.services.meshy_service import MeshyError, MeshyService
from app.services.quota import generation_usage, use_generation

router = APIRouter(prefix="/api/vendor", tags=["vendor"])
GLB_CACHE_VERSION = "vendor-glb-v1"

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


class GenerateAccepted(BaseModel):
    task_id: str
    provider: Provider
    status: str = "PENDING"
    remaining: int | None = None  # tries left for this provider after this call
    details: ProductDetails | None = None  # the vision model's reading of the photo (sizes are estimates)


class GenerateReady(BaseModel):
    task_id: str
    provider: Provider
    status: Literal["SUCCEEDED"] = "SUCCEEDED"
    glb_url: str
    remaining: int | None = None
    details: ProductDetails | None = None


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
    email = body.email.strip().lower()
    try:
        profile = (
            Vendor.register("", body.name, body.location, full_name=body.full_name.strip(), email=email, lat=body.lat, lng=body.lng)
            if body.kind == "vendor"
            else None
        )
    except DomainError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    client = request.app.state.auth_http
    user_id = await create_confirmed_user(
        email,
        body.password,
        {"role": body.kind, "name": body.name.strip(), "location": body.location.strip(), "full_name": body.full_name.strip()},
        settings,
        client,
    )
    # Handlers create the vendor profile (app/handlers.py).
    await bus.publish(UserRegistered(user_id, body.kind, profile and replace(profile, id=user_id)), request.app.state)
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
    return QuotaResponse(limit=settings.generation_limit, remaining={p: tries_left(settings.generation_limit, n) for p, n in used.items()})


def glb_cache_key(raw: bytes, user_id: str, provider: str, category: str, title: str, textured: bool, kind: str = "") -> str:
    """Bump GLB_CACHE_VERSION when generation changes (prompts, providers): older models are not reused."""
    other = f"|{kind.strip()}" if kind.strip() else ""  # only typed furniture carries a kind: older keys stay valid
    return hashlib.sha256(raw + f"|{user_id}|{provider}|{category}|{title.strip()}|{textured}{other}|{GLB_CACHE_VERSION}".encode()).hexdigest()


def read_upload(raw: bytes, declared: str, settings: Settings) -> str:
    """The photo's real type (sniffed), or 413 / 415."""
    if len(raw) > settings.max_image_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Images must be 20 MB or smaller.")
    sniffed = sniff_image_mime(raw)
    declared = declared.split(";")[0].strip().lower()
    if sniffed not in ALLOWED_IMAGE_TYPES or (declared and declared not in ALLOWED_IMAGE_TYPES and declared != "application/octet-stream"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a PNG, JPG or WEBP image.")
    return sniffed or "image/png"


async def read_photo(request: Request, settings: Settings, photo: bytes, mime: str, category: str, title: str, kind: str, lang: Literal["ar", "en"] = "ar") -> ProductDetails | None:
    """Workflow B: the admin-selected vision model reads the photo into JSON before 3D generation (cached
    per photo, model and object description, so measuring first and converting after costs one call)."""
    if not photo:
        return None  # an image URL only: Meshy fetches it, nothing to read here
    model = await vision.current_model(settings, request.app.state.auth_http)
    return await extract_product_details(
        photo, mime, category, title, settings, vision.client_for(request.app.state, model), model, db=request.app.state.auth_http, kind=kind or None, lang=lang
    )


@router.post("/product-details", response_model=ProductDetails)
async def product_details(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    user_id: Annotated[str, Depends(current_user)],
    image: Annotated[UploadFile, File()],
    category: Annotated[str, Form(max_length=40)] = "furniture",
    title: Annotated[str, Form(max_length=200)] = "",
    kind: Annotated[str, Form(max_length=60)] = "",
    lang: Annotated[Literal["ar", "en"], Form()] = "ar",  # the listing (name, description) in the vendor's interface language
) -> ProductDetails:
    """Measure before converting: the vision model's estimate (size in metres, the object's box in the photo,
    parts, materials) for the vendor to check and correct. Uses no generation try."""
    raw = await image.read()
    mime = read_upload(raw, image.content_type or "", settings)
    details = await read_photo(request, settings, raw, mime, category, title, kind, lang)
    if details is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The photo could not be measured right now. Enter the size yourself.")
    return details


async def take_try(request: Request, user_id: str, provider: Provider, settings: Settings) -> int:
    """Reserve one generation try; 429 when the vendor has used them all. Returns tries left."""
    used = await use_generation(user_id, provider, 1, settings, request.app.state.auth_http)
    if used is None:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(QuotaExhausted(provider, settings.generation_limit)))
    left = tries_left(settings.generation_limit, used)
    await bus.publish(GenerationStarted(user_id, provider, left), request.app.state)
    return left


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
    kind: Annotated[str, Form(max_length=60)] = "",  # furniture: the type the vendor typed ("corner sofa")
    lang: Annotated[Literal["ar", "en"], Form()] = "ar",  # the listing (name, description) in the vendor's interface language
) -> GenerateAccepted | GenerateReady:
    raw = await image.read() if image is not None else b""
    public_url = (image_url or "").strip() or None
    if raw:
        mime = read_upload(raw, image.content_type if image else "", settings)
    elif public_url:
        mime = "image/png"
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Upload a photo or provide image_url.")
    object_id = model_id or uuid4().hex
    textured = should_texture.strip().lower() not in {"0", "false", "no"}
    db = request.app.state.auth_http
    # The same photo, vendor and options → the model generated before (model_cache): no vision call,
    # no Meshy / Astra job, no try used. Per vendor: a model is never served from another studio's files.
    cache_key = glb_cache_key(raw, user_id, provider, category, title, textured, kind) if raw else None
    hit = await vision.cache_get(settings, db, cache_key, table="model_cache") if cache_key else None
    if hit and hit.get("glb_url"):
        details = ProductDetails.model_validate(hit["details"]) if hit.get("details") else None
        # The 3D model is reused; its listing must be in the language asked for now: read again only when it is not
        # (the reading is cached per language, and costs no generation try).
        if details is None or not all(in_language(t, lang) for t in (details.name, details.description) if t):
            details = await read_photo(request, settings, raw, mime, category, title, kind, lang) or details
        return GenerateReady(task_id=hit.get("model_id") or object_id, provider=provider, glb_url=hit["glb_url"], details=details)

    async def remember(url: str, details: ProductDetails | None) -> None:
        if cache_key:
            result = {"glb_url": url, "model_id": object_id, "details": details.model_dump(mode="json") if details else None}
            await vision.cache_put(settings, db, cache_key, "vendor_glb", provider, result, table="model_cache")

    if provider == "astra":
        if not settings.astra_configured:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not configured.")
        if not raw:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Astra needs the photo file, not only a URL.")
        remaining = await take_try(request, user_id, "astra", settings)
        try:
            details = await read_photo(request, settings, raw, mime, category, title, kind, lang)
            glb = await astra.generate_glb(raw, mime, astra_prompt(category, title, details, kind or None))
            url = cdn.upload(f"{user_id}/{object_id}.glb", glb, "model/gltf-binary")
            await remember(url, details)
        except (AstraError, CdnError) as exc:
            await bus.publish(GenerationFailed(user_id, "astra"), request.app.state)  # refunds the try
            raise HTTPException(exc.status_code, exc.message) from exc
        return GenerateReady(task_id=object_id, provider="astra", glb_url=url, remaining=remaining, details=details)

    if not settings.meshy_configured:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "MESHY_API_KEY is not configured.")
    remaining = await take_try(request, user_id, "meshy", settings)
    details = await read_photo(request, settings, raw, mime, category, title, kind, lang) if raw else None
    try:
        task_id = await meshy.create_image_to_3d_task(
            GenerationOptions(should_texture=textured, enable_pbr=False),
            image=None if public_url else raw,
            mime_type=mime,
            image_url=public_url,
            texture_prompt=meshy_texture_prompt(category, title, details, kind or None),
        )
    except MeshyError as exc:
        await bus.publish(GenerationFailed(user_id, "meshy"), request.app.state)  # Meshy refused: refunds the try
        raise HTTPException(exc.status_code, exc.message) from exc
    request.app.state.pending_cdn[task_id] = {
        "user_id": user_id, "model_id": object_id, "cache_key": cache_key,
        "details": details.model_dump(mode="json") if details else None,  # cached with the GLB when Meshy finishes
    }
    return GenerateAccepted(task_id=task_id, provider="meshy", remaining=remaining, details=details)


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
    if pending.get("cache_key"):
        result = {"glb_url": url, "model_id": object_id, "details": pending.get("details")}
        await vision.cache_put(get_settings(), request.app.state.auth_http, pending["cache_key"], "vendor_glb", "meshy", result, table="model_cache")
    request.app.state.pending_cdn.pop(task_id, None)
    return TaskStatusResponse(task_id=task_id, status="SUCCEEDED", progress=100, glb_url=url)


MAX_CARPET_PHOTO = 15 * 1024 * 1024


@router.post("/carpets/extract")
async def extract_carpet_image(
    _user: Annotated[str, Depends(current_user)],
    image: Annotated[UploadFile, File()],
    width_m: Annotated[float | None, Form(gt=0, le=50)] = None,
    length_m: Annotated[float | None, Form(gt=0, le=50)] = None,
    noun: Annotated[str, Form()] = "carpet",
) -> Response:
    """Photo of a carpet, door, or window → PNG of just that item at its real aspect ratio."""
    data = await image.read(MAX_CARPET_PHOTO + 1)
    if len(data) > MAX_CARPET_PHOTO:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Photo is larger than 15 MB.")
    try:
        png = await run_in_threadpool(extract_carpet, data, width_m, length_m, noun)
    except CarpetNotFound as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return Response(content=png, media_type="image/png")
