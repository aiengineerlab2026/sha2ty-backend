from __future__ import annotations

import base64
import hashlib
import json
import logging
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
from app.services import furniture_design, vision, wardrobe
from app.services.image_match import ImageMatchError, reference_png
from app.services.generation_prompt import astra_prompt, furniture_family, furniture_type, object_description, meshy_texture_prompt, split_prompt, target_polycount
from app.services.glb_size import MAX_GLB_BYTES, fit_glb
from app.services.glb_parts import inspect_glb, segment_glb, write_names
from app.services.model_check import check_glb, check_states, with_appearance
from app.services.part_names import name_components
from app.services.platform import is_platform
from app.services.product_details import extract_product_details, in_language
from app.services.cdn import Cdn, CdnError
from app.schemas.vendor import GenerationOptions
from app.services.meshy_service import MeshyError, MeshyService
from app.services import generator_setting
from app.services.quota import generation_usage, use_generation

router = APIRouter(prefix="/api/vendor", tags=["vendor"])
logger = logging.getLogger(__name__)
GLB_CACHE_VERSION = "vendor-glb-v4"  # v4: Meshy Image to Image with the furniture type (v3: remeshed, under 8 MB)

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
    components: dict | None = None  # meshes that actually exist in the GLB (glb_parts.inspect_glb)
    source_glb_url: str | None = None  # the untouched generated GLB, when glb_url is its split (modular) version
    # Other states of the same product (supabase/model_states.sql): {"open": {"glb_url", "components"}}.
    states: dict | None = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int = Field(ge=0, le=100)
    glb_url: str | None = None
    error: str | None = None
    retry_after_seconds: float | None = None
    components: dict | None = None
    source_glb_url: str | None = None


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
    provider: Provider = "meshy"  # the generator in use (the admin's choice)


@router.get("/generation-quota", response_model=QuotaResponse)
async def generation_quota(request: Request, user_id: Annotated[str, Depends(current_user)]) -> QuotaResponse:
    settings = get_settings()
    used = await generation_usage(user_id, settings, request.app.state.auth_http)
    return QuotaResponse(
        limit=settings.generation_limit,
        remaining={p: tries_left(settings.generation_limit, n) for p, n in used.items()},
        provider=await generator_setting.current(settings, request.app.state.auth_http),
    )


def builds_wardrobe(provider: str, category: str, kind: str, two_states: bool) -> bool:
    """Whether the photo-layout builder (services/wardrobe.py) makes this model instead of the admin's generator. The
    admin's choice is kept: with Meshy chosen, Meshy makes every single-photo model, wardrobes too. The builder makes
    wardrobes for the GPT generators, and a wardrobe's open and closed states always (only it lines them up exactly)."""
    return furniture_family(category, kind or None) == "wardrobe" and (two_states or provider != "meshy")


def glb_cache_key(raw: bytes, user_id: str, provider: str, category: str, title: str, textured: bool, kind: str = "", procedural: bool = False) -> str:
    """Bump GLB_CACHE_VERSION when generation changes (prompts, providers): older models are not reused."""
    other = f"|{kind.strip()}" if kind.strip() else ""  # only typed furniture carries a kind: older keys stay valid
    if procedural:  # a built wardrobe is never served for a request the admin's generator makes, nor the reverse
        other += "|procedural-2"  # 2: colours, decoration and the appearance check (1 made white wardrobes)
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
    """Reserve one generation try; 429 when the vendor has used them all. Returns tries left.
    The platform's own catalogue (admins as Sha2ty) has no limit."""
    if await is_platform(user_id, settings, request.app.state.auth_http):
        return settings.generation_limit
    used = await use_generation(user_id, provider, 1, settings, request.app.state.auth_http)
    if used is None:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(QuotaExhausted(provider, settings.generation_limit)))
    left = tries_left(settings.generation_limit, used)
    await bus.publish(GenerationStarted(user_id, provider, left), request.app.state)
    return left


def modular(glb: bytes, category: str | None, kind: str | None) -> tuple[bytes, bytes | None, dict]:
    """The GLB to save, the untouched generated one when it was split (else None), and its component document.

    The meshes the file really has are listed first (Astra can write separate parts; Meshy's image-to-3D has no
    part option and usually returns one mesh). One mesh: split from its geometry where it can be
    (glb_parts.segment_glb); otherwise it stays one editable object and the document says so. The part names
    follow the furniture type the vendor selected, never what the photo reading says."""
    family = furniture_family(category, kind or None)
    info = inspect_glb(glb, family)
    if info.get("separated"):
        return glb, None, {**info, "method": "file"}
    split = segment_glb(glb, family)
    if split is None:
        logger.info("No parts for this %s: one mesh, and its geometry shows no separable parts", family)
        return glb, None, {**info, "method": "none", "limits": "Not split into parts: the generator returned one fused mesh and its geometry shows no separate parts. Color and texture change the whole model."}
    modular_glb, doc = split
    return modular_glb, glb, doc


async def sized(glb: bytes, source_glb: bytes | None, components: dict) -> tuple[bytes, bytes | None, dict]:
    """The GLB made lighter (glb_size: textures and vertex data compacted, parts untouched), aiming for MAX_GLB_BYTES.
    No size limit: a model still over the target is saved as it is. The untouched original is compacted too."""
    glb, report = await run_in_threadpool(fit_glb, glb)
    if report["after"] > MAX_GLB_BYTES:
        logger.info("Generated model kept at %.1f MB (over the %d MB target)", report["after"] / 1048576, MAX_GLB_BYTES // 1048576)
    if source_glb:
        source_glb, _ = await run_in_threadpool(fit_glb, source_glb)
    return glb, source_glb, {**components, "file": {"bytes": report["after"], "originalBytes": report["before"], "optimised": report["steps"]}}


async def named(request: Request, glb: bytes, doc: dict, category: str | None, kind: str | None) -> tuple[bytes, dict]:
    """Descriptive part names (services/part_names.py), saved in the component document and in the GLB itself."""
    settings = get_settings()
    model = await vision.current_model(settings, request.app.state.auth_http)

    async def ask(prompt: str, schema: dict) -> str | None:
        if not vision.configured(settings, model):
            return None
        return await vision.ask_json([{"type": "input_text", "text": prompt}], schema, "part_names", settings, vision.client_for(request.app.state, model), model)

    doc = await name_components(doc, furniture_family(category, kind or None), ask)
    if not doc.get("separated"):
        return glb, doc
    try:
        return await run_in_threadpool(write_names, glb, doc["components"]), doc
    except (ValueError, KeyError) as exc:
        logger.warning("Could not write part names into the GLB: %s", exc)
        return glb, doc


def validated(glb: bytes, components: dict, category: str | None, kind: str | None, photo: bytes | None = None, bbox: list[float] | None = None) -> dict:
    """The model's check (services/model_check.py) in its component document, its colours against the photo
    included: a damaged or wrongly coloured model is flagged, not presented as a success (the review shows why)."""
    return {**components, "validation": with_appearance(check_glb(glb, furniture_family(category, kind or None)), glb, photo, bbox)}


class StatesUnusable(Exception):
    """Two states were asked for and cannot be made: said to the vendor (no image-to-3D fallback, which has one)."""


async def procedural_wardrobe(
    request: Request, settings: Settings, photo: bytes, mime: str, category: str, title: str, kind: str, lang: str,
    details: ProductDetails | None, open_photo: bytes | None = None, open_mime: str = "image/png",
) -> tuple[bytes, dict, tuple[bytes, dict] | None] | None:
    """A wardrobe rebuilt from its photo's layout as real, separately editable panels (services/wardrobe.py): the
    closed GLB and its document, and with `open_photo` the open state too (same panels and origin, doors turned and
    drawers pulled out, the inside as the open photo shows it). None when the layout cannot be read or fails its
    check (then the image-to-3D path runs instead); StatesUnusable when two states were asked for and fail."""
    model = await vision.current_model(settings, request.app.state.auth_http)
    if not vision.configured(settings, model):
        if open_photo:
            raise StatesUnusable("The vision model is not configured, so the two states cannot be read.")
        return None
    image = {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(photo).decode('ascii')}"}
    client = vision.client_for(request.app.state, model)

    async def ask(prompt: str, schema: dict) -> str | None:
        return await vision.ask_json([{"type": "input_text", "text": prompt}, image], schema, "wardrobe_layout", settings, client, model)

    what = object_description(category, title, kind or None)
    spec = await wardrobe.read_spec(ask, what, kind, details)
    if spec is None:
        if open_photo:
            raise StatesUnusable("The wardrobe's layout could not be read from the closed photo. Try a straight-on photo of the whole wardrobe.")
        return None
    reading, notes = None, []
    if open_photo:
        opened = {"type": "input_image", "image_url": f"data:{open_mime};base64,{base64.b64encode(open_photo).decode('ascii')}"}

        async def ask_both(prompt: str, schema: dict) -> str | None:
            return await vision.ask_json([{"type": "input_text", "text": prompt}, image, opened], schema, "wardrobe_open", settings, client, model)

        reading = await wardrobe.read_open(ask_both, spec, what)
        if reading is None:
            raise StatesUnusable("The open photo could not be read. Try a photo of the whole wardrobe with its doors open.")
        spec, notes = wardrobe.with_inside(spec, reading)
    glb, names = await run_in_threadpool(wardrobe.build, spec, lang)
    report = await run_in_threadpool(check_glb, glb, "wardrobe")
    if not report["ok"]:
        logger.warning("Procedural wardrobe rejected: %s (layout %s)", report["errors"], spec.model_dump_json())
        if open_photo:
            raise StatesUnusable("The wardrobe built from the photos failed its checks: " + " ".join(report["errors"][:3]))
        return None
    report = await run_in_threadpool(with_appearance, report, glb, photo, details.bbox if details else None)
    closed = {**wardrobe.components_doc(glb, names), "validation": report, "layout": spec.model_dump(mode="json")}
    if not open_photo:
        return glb, closed, None
    open_glb, open_names = await run_in_threadpool(wardrobe.build, spec, lang, "open")
    open_report = await run_in_threadpool(check_glb, open_glb, "wardrobe")
    pair = await run_in_threadpool(check_states, glb, open_glb)
    if not open_report["ok"] or not pair["ok"]:
        raise StatesUnusable("The open state failed its checks: " + " ".join((open_report["errors"] + pair["errors"])[:3]))
    open_report = await run_in_threadpool(with_appearance, open_report, open_glb, open_photo, None)
    if reading and not reading.same_wardrobe:  # flagged, not refused: the vendor sees why and can replace a photo
        open_report = {**open_report, "ok": False, "errors": [f"The two photos may not show the same wardrobe: {reading.differences or 'they differ'}."] + open_report["errors"]}
    open_report = {**open_report, "warnings": open_report["warnings"] + notes + pair["warnings"], "states": pair}
    return glb, closed, (open_glb, {**wardrobe.components_doc(open_glb, open_names), "validation": open_report})


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
    # Ignored: the generator is the admin's choice (app/services/generator_setting.py). Kept so older clients work.
    provider: Annotated[Provider | None, Form()] = None,
    model_id: Annotated[str | None, Form()] = None,
    should_texture: Annotated[str, Form()] = "true",
    category: Annotated[str, Form(max_length=40)] = "furniture",
    title: Annotated[str, Form(max_length=200)] = "",
    kind: Annotated[str, Form(max_length=60)] = "",  # furniture: the type the vendor typed ("corner sofa")
    lang: Annotated[Literal["ar", "en"], Form()] = "ar",  # the listing (name, description) in the vendor's interface language
    # Wardrobes with an open and a closed state: the photo of it open (`image` is the closed one).
    open_image: Annotated[UploadFile | None, File()] = None,
) -> GenerateAccepted | GenerateReady:
    provider = await generator_setting.current(settings, request.app.state.auth_http)
    logger.info("Generate: generator=%s (admin setting) category=%r kind=%r", provider, category, kind)
    raw = await image.read() if image is not None else b""
    open_raw = await open_image.read() if open_image is not None else b""
    open_mime = read_upload(open_raw, open_image.content_type if open_image else "", settings) if open_raw else "image/png"
    if open_raw and (not raw or furniture_family(category, kind or None) != "wardrobe"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Open and closed states are for wardrobes, with both photos uploaded.")
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
    cache_key = glb_cache_key(raw + open_raw, user_id, provider, category, title, textured, kind, builds_wardrobe(provider, category, kind, bool(open_raw))) if raw else None
    hit = await vision.cache_get(settings, db, cache_key, table="model_cache") if cache_key else None
    if hit and hit.get("glb_url"):
        details = ProductDetails.model_validate(hit["details"]) if hit.get("details") else None
        # The 3D model is reused; its listing must be in the language asked for now: read again only when it is not
        # (the reading is cached per language, and costs no generation try).
        if details is None or not all(in_language(t, lang) for t in (details.name, details.description) if t):
            details = await read_photo(request, settings, raw, mime, category, title, kind, lang) or details
        return GenerateReady(task_id=hit.get("model_id") or object_id, provider=provider, glb_url=hit["glb_url"], details=details, components=hit.get("components"), source_glb_url=hit.get("source_glb_url"), states=hit.get("states"))

    # Only a photo (the app's camera): read it first and take the type from it (furniture: what it is exactly).
    preread: ProductDetails | None = None
    if category == "auto":
        preread = await read_photo(request, settings, raw, mime, "auto", title, kind, lang) if raw else None
        category = preread.category if preread and preread.category else "furniture"
        # What the user typed wins (it goes into the Meshy / Astra prompt); else what the reading saw.
        kind = (kind.strip() or (preread.object_type if preread and category == "furniture" else ""))[:60]

    async def remember(url: str, details: ProductDetails | None, components: dict | None = None, source_url: str | None = None, states: dict | None = None) -> None:
        if cache_key:
            result = {"glb_url": url, "model_id": object_id, "details": details.model_dump(mode="json") if details else None, "components": components, "source_glb_url": source_url, "states": states}
            await vision.cache_put(settings, db, cache_key, "vendor_glb", provider, result, table="model_cache")

    # Wardrobes are flat panels: rebuilt from the photo's layout as real doors, drawers, shelves and handles instead of
    # cut from a fused image-to-3D surface, unless the admin chose Meshy (builds_wardrobe). No generation try is used
    # (only the vision model is called).
    if raw and builds_wardrobe(provider, category, kind, bool(open_raw)):
        details = preread or await read_photo(request, settings, raw, mime, category, title, kind, lang)
        try:
            built = await procedural_wardrobe(request, settings, raw, mime, category, title, kind, lang, details, open_raw or None, open_mime)
        except StatesUnusable as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        if built:
            glb, components, opened = built
            try:
                url = cdn.upload(f"{user_id}/{object_id}.glb", glb, "model/gltf-binary")
                states = {"open": {"glb_url": cdn.upload(f"{user_id}/{object_id}.open.glb", opened[0], "model/gltf-binary"), "components": opened[1]}} if opened else None
            except CdnError as exc:
                raise HTTPException(exc.status_code, exc.message) from exc
            await remember(url, details, components, states=states)
            return GenerateReady(task_id=object_id, provider=provider, glb_url=url, details=details, components=components, states=states)
        preread = details  # read once: the image-to-3D path below reuses it

    if provider in ("astra", "sol"):  # both OpenAI models via the Responses API
        if not settings.astra_configured:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not configured.")
        if not raw:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "The photo file is needed, not only its URL.")
        remaining = await take_try(request, user_id, provider, settings)
        try:
            details = preread or await read_photo(request, settings, raw, mime, category, title, kind, lang)
            glb = await astra.generate_glb(raw, mime, astra_prompt(category, title, details, kind or None), settings.openai_sol_model if provider == "sol" else None)
            glb, source_glb, components = await run_in_threadpool(modular, glb, category, kind)
            glb, components = await named(request, glb, components, category, kind)
            glb, source_glb, components = await sized(glb, source_glb, components)
            components = await run_in_threadpool(validated, glb, components, category, kind, raw, details.bbox if details else None)
            url = cdn.upload(f"{user_id}/{object_id}.glb", glb, "model/gltf-binary")
            source_url = cdn.upload(f"{user_id}/{object_id}.source.glb", source_glb, "model/gltf-binary") if source_glb else None
            await remember(url, details, components, source_url)
        except (AstraError, CdnError) as exc:
            await bus.publish(GenerationFailed(user_id, provider), request.app.state)  # refunds the try
            raise HTTPException(exc.status_code, exc.message) from exc
        return GenerateReady(task_id=object_id, provider=provider, glb_url=url, remaining=remaining, details=details, components=components, source_glb_url=source_url)

    if not settings.meshy_configured:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "MESHY_API_KEY is not configured.")
    remaining = await take_try(request, user_id, "meshy", settings)
    details = preread or (await read_photo(request, settings, raw, mime, category, title, kind, lang) if raw else None)
    # Image to 3D is the selected Meshy model. Image to Image only redraws a 2D picture (nano-banana / GPT Image)
    # and is not a 3D conversion. The uploaded bytes go in the request: a public URL Meshy cannot fetch never starts a task.
    kind_type = furniture_type(category, kind or None, details)
    mesh = {
        "textured": textured,
        "texture_prompt": meshy_texture_prompt(category, title, details, kind or None),
        "target_polycount": target_polycount(category, kind or None),
    }
    try:
        task_id = await meshy.create_image_to_3d_task(
            GenerationOptions(should_texture=textured, enable_pbr=textured),
            image=raw or None,
            mime_type=mime,
            image_url=None if raw else public_url,
            texture_prompt=mesh["texture_prompt"] if textured else None,
            target_polycount=mesh["target_polycount"],
        )
    except MeshyError as exc:
        await bus.publish(GenerationFailed(user_id, "meshy"), request.app.state)  # Meshy refused: refunds the try
        raise HTTPException(exc.status_code, exc.message) from exc
    logger.info("Meshy image-to-3d %s: furniture_type=%r category=%r", task_id, kind_type, category)
    request.app.state.pending_cdn[task_id] = {
        "stage": "mesh",
        "mesh_task_id": task_id,
        "mesh": mesh,
        "mime": mime,
        "image_url": public_url,
        "user_id": user_id, "model_id": object_id, "cache_key": cache_key,
        "details": details.model_dump(mode="json") if details else None,  # cached with the GLB when Meshy finishes
        "category": category,
        "kind": kind,
        "split": split_prompt(category, kind or None) if settings.meshy_auto_split else None,  # Auto Split parts
        "textured": textured,
        "photo": raw or None,  # the reference for the appearance check when the model arrives (dropped with the task)
    }
    return GenerateAccepted(task_id=task_id, provider="meshy", remaining=remaining, details=details)


IMAGE_STAGE = 20  # % of the progress bar the Image to Image step takes


async def conditioned_image(request: Request, meshy: MeshyService, task_id: str, pending: dict) -> TaskStatusResponse | None:
    """The Image to Image step: its progress while it runs; once done, Image to 3D started from its image (None: poll
    the 3D task). If it failed, Image to 3D starts from the photo itself and the model's notes say so."""
    try:
        image = await meshy.get_image_to_image_task(task_id)
    except MeshyError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    state = str(image.get("status") or "IN_PROGRESS")
    if state in {"PENDING", "IN_PROGRESS"}:
        return TaskStatusResponse(task_id=task_id, status="IN_PROGRESS", progress=int(image.get("progress") or 0) * IMAGE_STAGE // 100, retry_after_seconds=4)
    conditioned = state == "SUCCEEDED" and len(image.get("image_urls") or []) == 1  # Image to 3D needs exactly one image
    if not conditioned:
        err = image.get("task_error") or {}
        reason = err.get("message") if isinstance(err, dict) else str(err)
        logger.warning("Meshy Image to Image %s ended %s (%s): Image to 3D starts from the photo itself", task_id, state, reason)
        pending["conditioning"] = "The furniture type could not shape this model (Meshy's image step failed); it was made from the photo alone."
    mesh = pending["mesh"]
    try:
        pending["mesh_task_id"] = await meshy.create_image_to_3d_task(
            # PBR: metallic, roughness and normal maps besides the colour, so gold handles stay metal (glb_size keeps the
            # file within its limit).
            GenerationOptions(should_texture=mesh["textured"], enable_pbr=mesh["textured"]),
            image=None if conditioned or pending.get("image_url") else pending.get("photo"),
            mime_type=pending.get("mime") or "image/png",
            image_url=None if conditioned else pending.get("image_url"),
            texture_prompt=mesh["texture_prompt"],
            target_polycount=mesh["target_polycount"],
            input_task_id=task_id if conditioned else None,
        )
    except MeshyError as exc:
        if request.app.state.pending_cdn.pop(task_id, None) is not None:  # nothing more will come of it: the try is given back
            await bus.publish(GenerationFailed(pending.get("user_id") or "", "meshy"), request.app.state)
        return TaskStatusResponse(task_id=task_id, status="FAILED", progress=IMAGE_STAGE, error=exc.message)
    pending["stage"] = "mesh"
    pending["image_step"] = True  # this task spent the first part of the bar on Image to Image
    return TaskStatusResponse(task_id=task_id, status="IN_PROGRESS", progress=IMAGE_STAGE, retry_after_seconds=5)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(
    task_id: str,
    request: Request,
    meshy: Annotated[MeshyService, Depends(get_meshy)],
    cdn: Annotated[Cdn, Depends(get_cdn)],
    user_id: Annotated[str, Depends(current_user)],
) -> TaskStatusResponse:
    pending = request.app.state.pending_cdn.get(task_id) or {}
    if pending.get("user_id") and pending["user_id"] != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found.")
    if pending.get("stage") == "image":
        waiting = await conditioned_image(request, meshy, task_id, pending)
        if waiting:
            return waiting
    mesh_id = pending.get("mesh_task_id") or task_id  # without a pending entry: an Image to 3D id from before the chain
    try:
        task, retry_after = await meshy.get_image_to_3d_task(mesh_id)
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
    splitting = bool(pending.get("split"))
    if status_value != "SUCCEEDED":
        # A task that started on the old image step already used the first 20%. A direct Image to 3D task has not.
        base = IMAGE_STAGE if pending.get("image_step") else 0
        return TaskStatusResponse(
            task_id=task_id,
            status=status_value,
            progress=base + progress * ((90 if splitting else 100) - base) // 100,
            retry_after_seconds=retry_after or 5,
        )

    # Meshy image-to-3D returns one mesh; Auto Split cuts it into the vendor's furniture parts. Any split
    # failure keeps the generated model (split from its geometry by `modular`, or left whole).
    split_glb_url = None
    if splitting:
        try:
            if not pending.get("split_task_id"):
                pending["split_task_id"] = await meshy.create_split_task(mesh_id, pending["split"], bool(pending.get("textured")))
            split = await meshy.get_split_task(pending["split_task_id"])
        except MeshyError as exc:
            logger.warning("Meshy Auto Split for %s failed: %s", task_id, exc.message)
            pending["split"] = None
        else:
            split_status = str(split.get("status"))
            if split_status in {"PENDING", "IN_PROGRESS"}:
                return TaskStatusResponse(task_id=task_id, status="IN_PROGRESS", progress=90 + int(split.get("progress") or 0) // 10, retry_after_seconds=5)
            split_glb_url = (split.get("model_urls") or {}).get("glb") if split_status == "SUCCEEDED" else None
            logger.info("Meshy Auto Split %s: status=%s part_count=%s prompt_ignored=%s error=%s", pending["split_task_id"], split_status, split.get("part_count"), split.get("prompt_ignored"), split.get("task_error"))
    owner = pending.get("user_id") or user_id
    object_id = pending.get("model_id") or task_id
    model_urls = task.get("model_urls") or {}
    glb_url = model_urls.get("glb")
    if not glb_url:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Meshy finished but did not return a GLB.")
    dest = request.app.state.storage.models_dir / f"{task_id}.glb"
    try:
        await meshy.download_file(glb_url, dest, get_settings().max_model_bytes)
        data = dest.read_bytes()
        components = None
        if split_glb_url:
            split_dest = dest.with_suffix(".split.glb")
            await meshy.download_file(split_glb_url, split_dest, get_settings().max_model_bytes)
            info = await run_in_threadpool(inspect_glb, split_dest.read_bytes(), furniture_family(pending.get("category"), pending.get("kind") or None))
            logger.info("Meshy Auto Split GLB for %s: separated=%s components=%s", task_id, info.get("separated"), [c["name"] for c in info.get("components", [])])
            if info.get("separated"):  # independently editable meshes in the file, not just a reported count
                data, source_glb, components = split_dest.read_bytes(), data, {**info, "method": "meshy_auto_split"}
        if components is None:
            data, source_glb, components = await run_in_threadpool(modular, data, pending.get("category"), pending.get("kind"))
        data, components = await named(request, data, components, pending.get("category"), pending.get("kind"))
        if pending.get("conditioning"):  # said, not hidden: the type did not shape this model's image
            components = {**components, "limits": " ".join(x for x in (components.get("limits"), pending["conditioning"]) if x)}
        data, source_glb, components = await sized(data, source_glb, components)
        components = await run_in_threadpool(validated, data, components, pending.get("category"), pending.get("kind"), pending.get("photo"), (pending.get("details") or {}).get("bbox"))
        url = cdn.upload(f"{owner}/{object_id}.glb", data, "model/gltf-binary")
        source_url = cdn.upload(f"{owner}/{object_id}.source.glb", source_glb, "model/gltf-binary") if source_glb else None
    except (MeshyError, CdnError) as exc:
        raise HTTPException(getattr(exc, "status_code", 502), str(getattr(exc, "message", exc))) from exc
    if pending.get("cache_key"):
        result = {"glb_url": url, "model_id": object_id, "details": pending.get("details"), "components": components, "source_glb_url": source_url}
        await vision.cache_put(get_settings(), request.app.state.auth_http, pending["cache_key"], "vendor_glb", "meshy", result, table="model_cache")
    request.app.state.pending_cdn.pop(task_id, None)
    return TaskStatusResponse(task_id=task_id, status="SUCCEEDED", progress=100, glb_url=url, components=components, source_glb_url=source_url)


class StateBoxes(BaseModel):
    closed: list[float] | None = Field(default=None, description="The wardrobe shown closed: x_min, y_min, x_max, y_max, normalised 0..1")
    open: list[float] | None = Field(default=None, description="The same wardrobe shown open (doors open): x_min, y_min, x_max, y_max, normalised 0..1")


SPLIT_PROMPT = """This image shows one wardrobe twice, side by side or one above the other: once closed (doors shut) and
once open (doors open, the inside visible). Give the box around each depiction, the whole wardrobe and nothing of the
other, in normalised image coordinates (0..1, origin top left). Null for a depiction the image does not have."""


def crop_states(photo: bytes, boxes: StateBoxes) -> dict[str, str]:
    """Each state's depiction cut out (a little margin kept) as a JPEG data URL."""
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(photo)).convert("RGB")
    w, h = img.size
    out = {}
    for name in ("closed", "open"):
        box = getattr(boxes, name)
        if not box or len(box) != 4 or box[2] - box[0] < 0.05 or box[3] - box[1] < 0.05:
            continue
        x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in box)
        pad_x, pad_y = (x1 - x0) * 0.03, (y1 - y0) * 0.03
        part = img.crop((int(max(0, x0 - pad_x) * w), int(max(0, y0 - pad_y) * h), int(min(1, x1 + pad_x) * w), int(min(1, y1 + pad_y) * h)))
        buf = io.BytesIO()
        part.save(buf, format="JPEG", quality=92)
        out[name] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return out


@router.post("/split-states")
async def split_states(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    _user: Annotated[str, Depends(current_user)],
    image: Annotated[UploadFile, File()],
) -> dict[str, str]:
    """One photo showing a wardrobe both closed and open, cut into the two photos ({"closed", "open"} data URLs) for
    the vendor to check before generating. Uses no generation try."""
    raw = await image.read()
    mime = read_upload(raw, image.content_type or "", settings)
    model = await vision.current_model(settings, request.app.state.auth_http)
    if not vision.configured(settings, model):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The vision model is not configured.")
    content = [{"type": "input_text", "text": SPLIT_PROMPT}, {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"}]
    text = await vision.ask_json(content, StateBoxes.model_json_schema(), "wardrobe_states", settings, vision.client_for(request.app.state, model), model)
    try:
        boxes = StateBoxes.model_validate_json(text or "")
    except ValueError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The photo could not be read. Try again, or upload the two photos separately.") from exc
    images = await run_in_threadpool(crop_states, raw, boxes)
    if set(images) != {"closed", "open"}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Both the closed and the open wardrobe were not found in this photo. Upload the two photos separately.")
    return images


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


# ---- Admin: Generate furniture (services/furniture_design.py) ----


async def platform_user(request: Request, user_id: Annotated[str, Depends(current_user)]) -> str:
    """Only the Sha2ty account (an admin signed in from the admin console) generates furniture."""
    if not await is_platform(user_id, get_settings(), request.app.state.auth_http):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only admins can generate furniture.")
    return user_id


class FurnitureBody(BaseModel):
    kind: furniture_design.Kind


@router.post("/furniture/design", response_model=furniture_design.FurnitureSpec)
async def design_furniture(body: FurnitureBody, request: Request, _admin: Annotated[str, Depends(platform_user)]) -> furniture_design.FurnitureSpec:
    """A new random design of that type (nothing is saved: the admin reviews it first)."""
    settings = get_settings()
    if not settings.openai_api_key.get_secret_value():
        raise HTTPException(status.HTTP_409_CONFLICT, "Add OPENAI_API_KEY on the server to use GPT Sol.")
    try:
        return await furniture_design.design(body.kind, settings, request.app.state.openai_http)
    except ValueError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc


MAX_RENDER = 10 * 1024 * 1024


@router.post("/furniture/photos")
async def furniture_photos(
    request: Request,
    _admin: Annotated[str, Depends(platform_user)],
    render: Annotated[UploadFile, File()],
    kind: Annotated[furniture_design.Kind, Form()],
    title: Annotated[str, Form(max_length=120)] = "",
    style: Annotated[str, Form(max_length=200)] = "",
    looks: Annotated[str, Form(max_length=3000)] = "{}",  # the design's {part: {color, material}}
    which: Annotated[int | None, Form(ge=0, lt=len(furniture_design.SHOTS))] = None,
) -> dict[str, object]:
    """Product photos of the generated model from its render (studio, front, in a room, detail; or only shot `which`).
    Returned, not saved."""
    data = await render.read(MAX_RENDER + 1)
    if len(data) > MAX_RENDER:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "The render is larger than 10 MB.")
    try:
        parts = {k: v for k, v in json.loads(looks).items() if isinstance(v, dict)}
    except (ValueError, AttributeError):
        parts = {}
    try:
        images, errors = await furniture_design.photos(kind, title, reference_png(data), get_settings(), request.app.state.openai_http, style, furniture_design.finishes(parts), which)
    except ImageMatchError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    return {"images": {str(i): base64.b64encode(image).decode("ascii") for i, image in images.items()}, "shots": len(furniture_design.SHOTS), "mime": "image/jpeg", "errors": errors}
