from __future__ import annotations

import asyncio

import base64
import hashlib
import json
import logging
import time
from dataclasses import replace
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError
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
from app.services.generation_prompt import astra_prompt, furniture_family, furniture_type, object_description, meshy_clean_prompt, meshy_texture_prompt, split_prompt, target_polycount
from app.services.glb_size import MAX_GLB_BYTES, fit_glb
from app.services.glb_parts import inspect_glb, segment_glb, write_names
from app.services.model_check import check_glb, check_states, with_appearance
from app.services.part_names import name_components
from app.services.platform import is_platform
from app.services.product_details import in_language
from app.services import fidelity, furniture_spec, material_spec, structural, surface_finish
from app.services.cdn import Cdn, CdnError
from app.schemas.vendor import GenerationOptions
from app.services import animation, component_grouping, geometry_quality, mesh_analysis
from app.services.meshy_service import MeshyError, MeshyService, image_to_3d_payload, nearest_aspect_ratio, quality_options, texture_conditioning
from app.services import generator_setting
from app.services.quota import generation_usage, use_generation

router = APIRouter(prefix="/api/vendor", tags=["vendor"])
logger = logging.getLogger(__name__)
GLB_CACHE_VERSION = "vendor-glb-v7"  # v7: one vision spec, structural model from it, raw GLB kept (v6: animation)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
# Bound now, not by name at the first request: with postponed annotations FastAPI would resolve `get_settings` from
# this module's globals lazily, so a test that had monkeypatched it pinned every later request to its settings.
SettingsDep = Annotated[Settings, Depends(get_settings)]


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


class GenerationStep(BaseModel):
    id: Literal["image", "mesh"]
    status: str
    progress: int = 0


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int = Field(ge=0, le=100)
    glb_url: str | None = None
    error: str | None = None
    retry_after_seconds: float | None = None
    components: dict | None = None
    source_glb_url: str | None = None
    # Image to image, then image to 3D. Absent on a direct 3D task (cleaning turned off).
    steps: list[GenerationStep] | None = None
    # Meshy: image_preprocessing, image_validation, mesh_generation, auto_split, postprocessing, completed or failed.
    phase: str | None = None
    warnings: list[str] | None = None  # e.g. cleaning skipped / failed / rejected: the original photo was meshed
    clean_image_url: str | None = None  # the Meshy-cleaned image actually meshed (absent when the original was)


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
    """Whether the photo-layout builder (services/wardrobe.py) makes this model instead of the admin's generator. Only
    for a wardrobe the vendor uploaded open AND closed (only the builder lines both states up exactly). A single photo
    always goes to the selected generator, Meshy or Astra / Sol: a wardrobe is never silently built procedurally."""
    return furniture_family(category, kind or None) == "wardrobe" and two_states


def glb_cache_key(raw: bytes, user_id: str, provider: str, category: str, title: str, textured: bool, kind: str = "", procedural: bool = False) -> str:
    """Bump GLB_CACHE_VERSION when generation changes (prompts, providers): older models are not reused."""
    other = f"|{kind.strip()}" if kind.strip() else ""  # only typed furniture carries a kind: older keys stay valid
    if procedural:  # a built wardrobe is never served for a request the admin's generator makes, nor the reverse
        other += "|procedural-2"  # 2: colours, decoration and the appearance check (1 made white wardrobes)
    elif provider == "meshy":
        other += "|meshy-i2i-1"  # Meshy Image to Image cleaning, no vision reading: earlier Meshy models are not reused
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


read_furniture_spec = furniture_spec.read  # the one vision call (tests replace it)


async def read_spec(request: Request, settings: Settings, photo: bytes, mime: str, category: str, title: str, kind: str, lang: Literal["ar", "en"] = "ar") -> furniture_spec.FurnitureSpec | None:
    """The photo's furniture spec (services/furniture_spec.py): the one vision call, cached per photo, model, object
    description and language, so measuring first (/product-details) and converting after costs one call."""
    if not photo:
        return None  # an image URL only: Meshy fetches it, nothing to read here
    model = await vision.current_model(settings, request.app.state.auth_http)
    return await read_furniture_spec(
        photo, mime, category, title, settings, vision.client_for(request.app.state, model), model, db=request.app.state.auth_http, kind=kind or None, lang=lang
    )


async def read_photo(request: Request, settings: Settings, photo: bytes, mime: str, category: str, title: str, kind: str, lang: Literal["ar", "en"] = "ar") -> ProductDetails | None:
    """The listing part of the spec (name, category, estimated size): the same cached call as read_spec."""
    spec = await read_spec(request, settings, photo, mime, category, title, kind, lang)
    return spec.details if spec else None


@router.post("/product-details", response_model=ProductDetails)
async def product_details(
    request: Request,
    settings: SettingsDep,
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
    No size limit: a model still over the target is saved as it is. The untouched original is compacted with its
    upload, after the response (schedule_source)."""
    glb, report = await run_in_threadpool(fit_glb, glb)
    if report["after"] > MAX_GLB_BYTES:
        logger.info("Generated model kept at %.1f MB (over the %d MB target)", report["after"] / 1048576, MAX_GLB_BYTES // 1048576)
    return glb, source_glb, {**components, "file": {"bytes": report["after"], "originalBytes": report["before"], "optimised": report["steps"]}}


async def named(request: Request, glb: bytes, doc: dict, category: str | None, kind: str | None, allow_vision: bool = True) -> tuple[bytes, dict]:
    """Descriptive part names (services/part_names.py), saved in the component document and in the GLB itself.
    allow_vision=False (Meshy without the optional analysis): names from geometry and the furniture type only."""
    settings = get_settings()
    use_vision = allow_vision and settings.visual_part_ai
    model = await vision.current_model(settings, request.app.state.auth_http) if use_vision else ""

    async def ask(prompt: str, schema: dict) -> str | None:
        # Geometry names every part without a call; the vision model only when the admin turned it on.
        if not use_vision or not vision.configured(settings, model):
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


async def grouped(request: Request, glb: bytes, source_glb: bytes | None, doc: dict, photo: bytes | None, details: dict | None,
                  allow_vision: bool = True) -> tuple[bytes, bytes | None, dict]:
    """The generated meshes grouped into carpenter-named components (services/component_grouping.py), checked
    against the geometry. Anything unsafe or unidentified keeps the model as it was; the file before grouping is
    kept as the source (the fallback) when there was none."""
    settings = get_settings()
    ask = None
    model = await vision.current_model(settings, request.app.state.auth_http) if allow_vision and settings.visual_part_ai else ""
    if model and vision.configured(settings, model):  # an extra vision call: off unless the admin turns it on
        client = vision.client_for(request.app.state, model)

        async def ask(content: list[dict], schema: dict) -> str | None:
            return await vision.ask_json(content, schema, "component_grouping", settings, client, model)

    hints = [str(c.get("name") or c.get("id")) for c in ((doc.get("analysis") or {}).get("components") or []) if c.get("name") or c.get("id")]
    try:
        out = await component_grouping.group(glb, photo, ask, hints[:30], (details or {}).get("height_m"))
    except Exception as exc:  # an optional step: the split model is kept on any failure
        logger.warning("Component grouping failed: %s", str(exc)[:300])
        out = None
    if out is None:
        return glb, source_glb, doc
    grouped_glb, extra = out
    warnings = extra["grouping"]["warnings"]
    logger.info("Component grouping: %d → %d components %s", extra["grouping"]["partsBefore"], extra["grouping"]["components"], [c["name"] for c in extra["components"]])
    return grouped_glb, source_glb or glb, {
        **doc, "separated": True, "components": extra["components"], "grouping": extra["grouping"],
        "method": f"{doc.get('method') or 'file'}+grouped",
        **({"limits": " ".join(x for x in (doc.get("limits"), *warnings) if x)} if warnings else {}),
    }


async def quality_checked(glb: bytes, doc: dict, details: dict | None) -> tuple[bytes, dict, dict | None]:
    """Geometry checked and safely repaired, each part with its own material, its class and size, and the quality
    report (services/geometry_quality.py); also what the AR copy is made from, or None when it is not worth one."""
    try:
        return await run_in_threadpool(geometry_quality.process, glb, doc, (details or {}).get("height_m"), False)
    except Exception as exc:  # an optional step: the model is kept as it was, and says so
        logger.warning("Geometry quality step failed: %s", str(exc)[:300])
        return glb, {**doc, "quality": {"version": geometry_quality.VERSION, "unresolved": [f"Geometry not checked: {str(exc)[:120]}"]}}, None


def build_ar_copy(job: dict, cdn: Cdn, key: str) -> None:
    """Background: the lighter AR copy, compacted like the model and uploaded where its document already points. If it
    cannot be made (or is not lighter enough), nothing is uploaded and AR keeps opening the full model."""
    try:
        result = geometry_quality.ar_copy(job)
        if result is None:
            logger.info("AR copy for %s not made: not light enough to be worth it, or a check failed", key)
            return
        glb, info = result
        glb, _ = fit_glb(glb)
        cdn.upload(key, glb, "model/gltf-binary")
        logger.info("AR copy %s: %d → %d triangles, %.1f MB", key, info["original"], info["triangles"], len(glb) / 1048576)
    except Exception:  # noqa: BLE001 - background work: logged, the full model is the fallback
        logger.exception("AR copy for %s failed", key)


def upload_source(source_glb: bytes, cdn: Cdn, key: str) -> None:
    """Background: the untouched generated model (the fallback and the "Original GLB" download), compacted and uploaded."""
    try:
        glb, _ = fit_glb(source_glb)
        cdn.upload(key, glb, "model/gltf-binary")
        logger.info("Original model %s uploaded (%.1f MB)", key, len(glb) / 1048576)
    except Exception:  # noqa: BLE001 - background work: logged; the processed model is already saved
        logger.exception("Original model upload %s failed", key)


def schedule_source(background: BackgroundTasks, source_glb: bytes | None, cdn: Cdn, key: str) -> str | None:
    """The original's address now, its upload after the response: the processed model is what the vendor waits for
    (only it is opened in the editor); the original is a fallback and a download, and doubled the upload time."""
    if not source_glb:
        return None
    background.add_task(upload_source, source_glb, cdn, key)
    return cdn.public_url(key)


def schedule_ar(background: BackgroundTasks, job: dict | None, cdn: Cdn, owner: str, object_id: str, doc: dict) -> dict:
    """The AR copy is made after the response (big models take a while): its address is known now and goes in the
    document; the AR viewer falls back to the full model until (or unless) the file is there."""
    if job is None:
        return doc
    key = f"{owner}/{object_id}.ar.glb"
    background.add_task(build_ar_copy, job, cdn, key)
    return {**doc, "ar": {"url": cdn.public_url(key), "status": "processing"}}


def structural_model(spec: dict | None, category: str | None, kind: str | None, cdn: Cdn, key: str) -> dict:
    """The structural model from the photo's spec alone (services/structural.py), uploaded beside the visual one. The
    visual model is never read or changed. Its report, with `url` when it was built and `uncertain` always."""
    spec = spec or {}
    try:
        glb, report = structural.build(spec.get("details"), spec.get("analysis"), category, kind)
        if glb:
            report["url"] = cdn.upload(key, glb, "model/gltf-binary")
    except Exception as exc:  # noqa: BLE001 - the visual model is the product; this one must not fail it
        logger.exception("Structural model %s was not built", key)
        report = {"source": "vision_spec", "built": False, "uncertain": [f"The structural model could not be built ({str(exc)[:100]})."]}
    return report


def with_structural(components: dict, report: dict | None, spec: dict | None) -> dict:
    """The structural model (`carpenter`, read by the editor's structural mode), its status and the shared material
    spec, in the component document. The visual parts keep `analysisId`: the link to the same spec component."""
    report = report or {"built": False, "uncertain": ["No photo reading: no structural model."]}
    status = {k: report.get(k) for k in ("source", "built", "uncertain", "excludedLoose", "sizeM", "scaleKnown") if k in report}
    out = {**components, "structural": status, "materials": material_spec.for_analysis((spec or {}).get("analysis"))}
    if report.get("built") and report.get("url"):
        out["carpenter"] = {k: v for k, v in report.items() if k not in ("materials",)}
    else:
        out.pop("carpenter", None)
    if spec:
        out["spec"] = {"looseObjects": spec.get("loose_objects") or [], "designFeatures": spec.get("design_features") or []}
    return out


def upload_raw(raw: bytes, cdn: Cdn, key: str) -> None:
    """Background: the provider's GLB exactly as it arrived (no compaction): the reference every check compares with."""
    try:
        cdn.upload(key, raw, "model/gltf-binary")
    except Exception:  # noqa: BLE001 - background work: logged, the processed model is already saved
        logger.exception("Raw provider model %s was not uploaded", key)


def fidelity_checked(components: dict, raw: bytes, glb: bytes, photo: bytes | None, cleaned: bytes | None, bbox: list[float] | None,
                     cleaning: dict | None = None, split: dict | None = None, expected: list[float] | None = None, layout: bool = False) -> dict:
    """Where the visual model lost fidelity (services/fidelity.py), stage by stage. A failed stage flags the model
    (quality "low", its issues in the validation warnings) instead of passing it as a success. `cleaning`: the check
    already made before meshing; `split`: Auto Split's; `layout`: the camera-aligned render checks (Meshy)."""
    stages = []
    if cleaning:
        stages.append(cleaning)
    elif cleaned and photo:
        stages.append(fidelity.compare_images(photo, cleaned, bbox))
    stages.append(fidelity.provider(raw, photo, bbox, cleaned, expected, layout))
    if split:
        stages.append(split)
    stages += fidelity.compare_glbs(raw, glb)
    stages.append(fidelity.structural(components.get("structural") and {**(components.get("carpenter") or {}), **components["structural"]}))
    result = fidelity.report(stages)
    out = {**components, "fidelity": result}
    if not result["ok"]:
        lost = [f"{s['stage']}: {issue}" for s in stages if s.get("ok") is False and s["stage"] != "structural" for issue in s["issues"]]
        validation = out.get("validation") or {"ok": True, "errors": [], "warnings": [], "parts": 0}
        out["validation"] = {**validation, "ok": False, "warnings": [*validation.get("warnings", []), *lost[:8]]}
    return out


def validated(glb: bytes, components: dict, category: str | None, kind: str | None, photo: bytes | None = None, bbox: list[float] | None = None) -> dict:
    """The model's check (services/model_check.py) in its component document, its colours against the photo
    included: a damaged or wrongly coloured model is flagged, not presented as a success (the review shows why).
    Loose objects still in the model (books, bedding, vases...) are flagged too: cleaning the image may have missed them."""
    validation = with_appearance(check_glb(glb, furniture_family(category, kind or None)), glb, photo, bbox)
    loose = loose_objects(components)
    if loose:
        validation = {**validation, "loose_objects": loose, "warnings": [*validation.get("warnings", []), f"Loose objects still in the model: {', '.join(loose[:8])}."]}
    return {**components, "validation": validation}


def loose_objects(components: dict) -> list[str]:
    """Names of the components classed decorative (not part of the furniture's structure: geometry_quality.classify)."""
    return [str(c.get("name") or c.get("id")) for c in components.get("components") or [] if geometry_quality.classify(c) == "decorative"]


def cleaning_record(pending: dict, cdn: Cdn, key: str) -> dict:
    """What the image-cleaning step did: its prompt, its check against the original, and the cleaned image (used or
    rejected), copied to the CDN."""
    cleaned = pending.get("reference") is not None
    record: dict = {"prompt": pending.get("clean_prompt"), "cleaned": cleaned, "used": "cleaned" if cleaned else "original",
                    "provider": "meshy-image-to-image" if pending.get("image_task_id") or pending.get("cleaning_fallback") else None,
                    "imageTaskId": pending.get("image_task_id")}
    if pending.get("cleaning_fallback"):
        record["fallback"] = pending["cleaning_fallback"]
    if pending.get("cleaning_check"):
        record["check"] = pending["cleaning_check"]
        record["rejected"] = bool(pending["cleaning_check"].get("rejected"))
    if cleaned and pending.get("clean_image_url"):
        record["image_url"] = pending["clean_image_url"]  # uploaded when it was validated
        return record
    image = pending.get("reference") or pending.get("rejected_clean")
    if image:
        try:
            record["image_url"] = cdn.upload(key, image, "image/png")
        except CdnError as exc:
            logger.warning("Cleaned reference image not uploaded: %s", exc.message)
    return record


def meshy_input(image: bytes | None, mime: str, source: str, mesh: dict, url: str | None = None, input_task_id: str | None = None) -> dict:
    """What Meshy Image to 3D was given: which image (its hash and size; the Image to Image task when the cleaned image
    went by input_task_id), and what it textures from, computed from the same payload builder the request uses."""
    settings = get_settings()
    payload = image_to_3d_payload(GenerationOptions(should_texture=mesh["textured"], enable_pbr=mesh["textured"]),
                                  {"input_task_id": "<task>"} if input_task_id else {"image_url": "<image>"},
                                  mesh.get("texture_prompt"), mesh.get("target_polycount"), settings.meshy_texture_source, **quality_options(settings))
    return {
        "source": source, "mime": mime, "bytes": len(image) if image else None, "url": url, "inputTaskId": input_task_id,
        "sha256": hashlib.sha256(image).hexdigest() if image else None,
        "texture": texture_conditioning(payload), "textureSource": settings.meshy_texture_source,
        "fields": sorted(payload),
    }


def meshy_input_uploaded(record: dict | None, image: bytes | None, cdn: Cdn, key: str) -> dict | None:
    """The exact image Meshy meshed, kept beside the model (its record gets the address)."""
    if not record or not image:
        return record
    ext = {"image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}.get(record.get("mime") or "", "png")
    try:
        return {**record, "url": cdn.upload(f"{key}.{ext}", image, record.get("mime") or "image/png")}
    except CdnError as exc:
        logger.warning("Meshy input image not uploaded: %s", exc.message)
        return record


def meshy_parts(glb: bytes, category: str | None, kind: str | None) -> tuple[bytes, None, dict]:
    """The meshes Meshy's file really has. One fused mesh stays one: it is never cut into guessed parts by its
    geometry (that is Astra's `modular`); the structural model is where the components are edited."""
    family = furniture_family(category, kind or None)
    info = inspect_glb(glb, family)
    if info.get("separated"):
        return glb, None, {**info, "method": "file"}
    return glb, None, {**info, "method": "none", "limits": "Meshy returned one fused mesh: the visual model is not modular. "
                       "Edit its components in the structural model."}


def analysis_checked(components: dict, pending: dict) -> dict:
    """The generated meshes matched to the vision analysis (names in English and Arabic; geometry untouched), the
    analysis and where its prompt went stored with them for later editing and regeneration."""
    raw = pending.get("analysis")
    analysis = mesh_analysis.MeshAnalysis.model_validate(raw) if raw else None
    components = mesh_analysis.match(components, analysis)
    if pending.get("meshy_conditioning"):
        components = {**components, "conditioning": pending["meshy_conditioning"]}
    if (components.get("modularity") or {}).get("level") == "partial":
        components = {**components, "limits": " ".join(x for x in (components.get("limits"), components["modularity"]["note"]) if x)}
    if analysis is None:
        why = ("Parts are named from the model's own geometry and the furniture type: Meshy generation makes no vision analysis of the photo."
               if pending.get("vision_off") else "The photo was not analysed into components before generation (no vision model answer).")
        components = {**components, "limits": " ".join(x for x in (components.get("limits"), why) if x)}
    return components


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
    _, closed = await run_in_threadpool(animation.configure, glb, wardrobe.components_doc(glb, names))  # doors and drawers by name
    closed = {**closed, "validation": report, "layout": spec.model_dump(mode="json")}
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
    background: BackgroundTasks,
    meshy: Annotated[MeshyService, Depends(get_meshy)],
    astra: Annotated[AstraService, Depends(get_astra)],
    cdn: Annotated[Cdn, Depends(get_cdn)],
    settings: SettingsDep,
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
    # Meshy reads nothing with the vision model unless the optional analysis is on (the two-state wardrobe builder,
    # which is not Meshy, still reads its photos).
    vision_reading = provider != "meshy" or settings.meshy_vision_analysis
    # The same photo, vendor and options → the model generated before (model_cache): no vision call,
    # no Meshy / Astra job, no try used. Per vendor: a model is never served from another studio's files.
    cache_key = glb_cache_key(raw + open_raw, user_id, provider, category, title, textured, kind, builds_wardrobe(provider, category, kind, bool(open_raw))) if raw else None
    hit = await vision.cache_get(settings, db, cache_key, table="model_cache") if cache_key else None
    if hit and hit.get("glb_url"):
        details = ProductDetails.model_validate(hit["details"]) if hit.get("details") else None
        # The 3D model is reused; its listing must be in the language asked for now: read again only when it is not
        # (the reading is cached per language, and costs no generation try).
        if vision_reading and (details is None or not all(in_language(t, lang) for t in (details.name, details.description) if t)):
            details = await read_photo(request, settings, raw, mime, category, title, kind, lang) or details
        return GenerateReady(task_id=hit.get("model_id") or object_id, provider=provider, glb_url=hit["glb_url"], details=details, components=hit.get("components"), source_glb_url=hit.get("source_glb_url"), states=hit.get("states"))

    # Only a photo (the app's camera): read it first and take the type from it (furniture: what it is exactly).
    preread: ProductDetails | None = None
    prespec: furniture_spec.FurnitureSpec | None = None
    if category == "auto":
        prespec = await read_spec(request, settings, raw, mime, "auto", title, kind, lang) if raw and (vision_reading or open_raw) else None
        preread = prespec.details if prespec else None
        category = preread.category if preread and preread.category else "furniture"
        # What the user typed wins (it goes into the Meshy / Astra prompt); else what the reading saw.
        kind = (kind.strip() or (preread.object_type if preread and category == "furniture" else ""))[:60]

    async def remember(url: str, details: ProductDetails | None, components: dict | None = None, source_url: str | None = None, states: dict | None = None) -> None:
        if cache_key:
            result = {"glb_url": url, "model_id": object_id, "details": details.model_dump(mode="json") if details else None, "components": components, "source_glb_url": source_url, "states": states}
            await vision.cache_put(settings, db, cache_key, "vendor_glb", provider, result, table="model_cache")

    # A wardrobe photographed open and closed: both states rebuilt from the photos' layout, so they line up exactly
    # (builds_wardrobe). A single photo always goes to the selected generator below.
    if raw and builds_wardrobe(provider, category, kind, bool(open_raw)):
        details = preread or await read_photo(request, settings, raw, mime, category, title, kind, lang)
        try:
            built = await procedural_wardrobe(request, settings, raw, mime, category, title, kind, lang, details, open_raw or None, open_mime)
        except StatesUnusable as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        if built:
            glb, components, opened = built
            try:
                url = await run_in_threadpool(cdn.upload, f"{user_id}/{object_id}.glb", glb, "model/gltf-binary")
                states = {"open": {"glb_url": await run_in_threadpool(cdn.upload, f"{user_id}/{object_id}.open.glb", opened[0], "model/gltf-binary"), "components": opened[1]}} if opened else None
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
            spec = prespec or await read_spec(request, settings, raw, mime, category, title, kind, lang)
            details = spec.details if spec else preread
            spec_doc = spec.model_dump(mode="json") if spec else None
            # The visual model (the provider) and the structural one (deterministic, from the spec) at the same time.
            glb, built = await asyncio.gather(
                astra.generate_glb(raw, mime, astra_prompt(category, title, details, kind or None), settings.openai_sol_model if provider == "sol" else None),
                run_in_threadpool(structural_model, spec_doc, category, kind, cdn, f"{user_id}/{object_id}.carpenter.glb"),
            )
            raw_glb = glb  # the provider's file before any change, kept and compared with
            background.add_task(upload_raw, raw_glb, cdn, f"{user_id}/{object_id}.raw.glb")
            glb, surfaces = await run_in_threadpool(surface_finish.flatten, glb)  # photographed lighting tiled on panels
            glb, source_glb, components = await run_in_threadpool(modular, glb, category, kind)
            components = {**mesh_analysis.match(components, spec.analysis if spec else None), **({"surfaces": surfaces} if surfaces else {})}
            glb, source_glb, components = await grouped(request, glb, source_glb, components, raw, details.model_dump(mode="json") if details else None)
            glb, components = await named(request, glb, components, category, kind)
            glb, components, ar_job = await quality_checked(glb, components, details.model_dump(mode="json") if details else None)
            components = schedule_ar(background, ar_job, cdn, user_id, object_id, components)
            glb, components = await run_in_threadpool(animation.configure, glb, components)
            glb, source_glb, components = await sized(glb, source_glb, components)
            components = await run_in_threadpool(validated, glb, components, category, kind, raw, details.bbox if details else None)
            components = {**with_structural(components, built, spec_doc), "raw_glb_url": cdn.public_url(f"{user_id}/{object_id}.raw.glb")}
            components = await run_in_threadpool(fidelity_checked, components, raw_glb, glb, raw, None, details.bbox if details else None)
            url = await run_in_threadpool(cdn.upload, f"{user_id}/{object_id}.glb", glb, "model/gltf-binary")
            source_url = schedule_source(background, source_glb, cdn, f"{user_id}/{object_id}.source.glb")
            await remember(url, details, components, source_url)
        except (AstraError, CdnError) as exc:
            await bus.publish(GenerationFailed(user_id, provider), request.app.state)  # refunds the try
            raise HTTPException(exc.status_code, exc.message) from exc
        return GenerateReady(task_id=object_id, provider=provider, glb_url=url, remaining=remaining, details=details, components=components, source_glb_url=source_url)

    if not settings.meshy_configured:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "MESHY_API_KEY is not configured.")
    remaining = await take_try(request, user_id, "meshy", settings)
    # Meshy makes no vision-model call unless the admin turned on the optional analysis (meshy_vision_analysis): the
    # vendor's category, type and title are the metadata; parts come from the GLB's own geometry.
    spec = prespec or (await read_spec(request, settings, raw, mime, category, title, kind, lang) if raw and settings.meshy_vision_analysis else None)
    details = spec.details if spec else preread
    analysis = spec.analysis if spec else None
    spec_doc = spec.model_dump(mode="json") if spec else None
    # Original photo → Meshy Image to Image (furniture cleaning prompt) → Image to 3D from that task → Auto Split →
    # grouping, naming, checks. Image to 3D has no geometry prompt, so what is removed is removed in the image.
    kind_type = furniture_type(category, kind or None, details)
    clean_prompt = meshy_clean_prompt(category, kind or None, settings.meshy_clean_prompt, settings.meshy_clean_hints,
                                      spec.loose_objects if spec and spec.needs_cleaning else None) if settings.meshy_clean_image else None
    base_texture = meshy_texture_prompt(category, title, details, kind or None)
    split = split_prompt(category, kind or None)  # the vendor's furniture type's parts: no photo reading needed
    conditioning = None
    if analysis:
        base_texture = mesh_analysis.texture_prompt(analysis, base_texture)
        split = mesh_analysis.split_prompt(analysis) or split
        conditioning = mesh_analysis.conditioning(mesh_analysis.meshy_prompt(analysis), base_texture, split, settings.meshy_auto_split, clean_prompt)
    mesh = {
        "textured": textured,
        "texture_prompt": base_texture,
        "target_polycount": target_polycount(category, kind or None),
    }
    pending: dict = {
        "stage": "mesh", "mesh_task_id": None, "image_task_id": None, "clean_prompt": clean_prompt,
        "mesh": mesh, "mime": mime, "image_url": public_url,
        "user_id": user_id, "model_id": object_id, "cache_key": cache_key,
        "details": details.model_dump(mode="json") if details else None,  # cached with the GLB when Meshy finishes
        "category": category, "kind": kind,
        "split": split if settings.meshy_auto_split else None,  # Auto Split parts
        "analysis": analysis.model_dump(mode="json") if analysis else None,
        "meshy_conditioning": conditioning,
        "spec": spec_doc,
        "vision_off": not settings.meshy_vision_analysis,
        "textured": textured,
        "photo": raw or None,  # the reference for the checks when the model arrives (dropped with the task)
        "warnings": [],
    }
    task_id = None
    if clean_prompt:
        try:
            pending["image_task_id"] = await meshy.create_image_to_image_task(
                clean_prompt, image=raw or None, mime_type=mime, image_url=None if raw else public_url, aspect_ratio=nearest_aspect_ratio(raw, settings.meshy_image_model),
            )
        except MeshyError as exc:
            if not settings.meshy_use_original_on_bad_cleaning:
                await bus.publish(GenerationFailed(user_id, "meshy"), request.app.state)  # refunds the try
                raise HTTPException(exc.status_code, f"{IMAGE_FAILED} {exc.message}") from exc
            pending["warnings"].append(cleaning_warning("failed", exc.message))
            pending["cleaning_fallback"] = {"what": "failed", "reason": exc.message}
            pending["image_step_status"] = "SKIPPED"
        else:
            task_id = uuid4().hex  # ours: the Meshy ids are kept in the pending entry
            pending.update(stage="image", clean_started=time.monotonic())
    if task_id is None:
        pending["meshy_input"] = meshy_input(raw or None, mime, "original", mesh, None if raw else public_url)
        pending["meshy_input_bytes"] = raw or None
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
        pending["mesh_task_id"] = task_id
    logger.info("Meshy %s %s: furniture_type=%r category=%r analysed=%s split=%r", "image-to-image" if pending["stage"] == "image" else "image-to-3d",
                task_id, kind_type, category, bool(analysis), split)
    # The structural model needs a validated spec (vision analysis); without one it is reported unavailable, not guessed.
    pending["structural"] = await run_in_threadpool(structural_model, spec_doc, category, kind, cdn, f"{user_id}/{object_id}.carpenter.glb") if spec_doc else STRUCTURAL_OFF
    request.app.state.pending_cdn[task_id] = pending
    return GenerateAccepted(task_id=task_id, provider="meshy", remaining=remaining, details=details)


IMAGE_STAGE = 20  # % of the progress bar the Image to Image step takes
IMAGE_FAILED = "Image to image failed, so the 3D model was not generated."
TRANSIENT = {429, 502, 503, 504}  # a poll that hit these is retried later, never counted as the task failing
FINISHED_TTL = 3600.0  # finished task answers kept this long, so repeated or concurrent polls never redo the work
STRUCTURAL_OFF = {"source": "none", "built": False, "uncertain": [
    "No structural specification: Meshy generation makes no vision analysis (MESHY_VISION_ANALYSIS is off), so no "
    "carpenter-ready model was built. The visual model's parts are not a structural plan."]}


def cleaning_warning(what: str, reason: str) -> str:
    """Said whenever the original photo was meshed instead of a cleaned one: never presented as cleaned."""
    reason = reason.strip().rstrip(".")
    return (f"Image cleaning {what}{f' ({reason})' if reason else ''}. The original photo was used for 3D, so pillows, bedding "
            "or other accessories in it may be part of the model.")


def _steps(image: tuple[str, int] | None, mesh: tuple[str, int] | None) -> list[GenerationStep] | None:
    rows = []
    if image:
        rows.append(GenerationStep(id="image", status=image[0], progress=image[1]))
    if mesh:
        rows.append(GenerationStep(id="mesh", status=mesh[0], progress=mesh[1]))
    return rows or None


def _mesh_steps(pending: dict, status: str, progress: int) -> list[GenerationStep] | None:
    """Image to 3D, with the cleaning step's outcome when this task had one (SUCCEEDED, or SKIPPED: original used)."""
    done = pending.get("image_step_status")
    return _steps((done, 100 if done == "SUCCEEDED" else 0) if done else None, (status, progress))


def _status(task_id: str, pending: dict, phase: str, **kw) -> TaskStatusResponse:
    """A task answer with its phase, the warnings so far and the cleaned image (when one is used)."""
    return TaskStatusResponse(task_id=task_id, phase=phase, warnings=list(pending.get("warnings") or []) or None,
                              clean_image_url=pending.get("clean_image_url"), **kw)


def _finish(request: Request, task_id: str, pending: dict, response: TaskStatusResponse) -> TaskStatusResponse:
    """The task's final answer, kept (bytes dropped) so a later or concurrent poll returns it instead of polling Meshy
    and processing again. ponytail: in-process dict with a TTL; a shared store if the API runs on several hosts."""
    lock = pending.get("_lock")
    keep = {"user_id": pending.get("user_id"), "result": response, "finished": time.monotonic(), "_lock": lock}
    pending.clear()
    pending.update(keep)
    tasks = request.app.state.pending_cdn
    tasks[task_id] = pending
    now = time.monotonic()
    for key in [k for k, v in tasks.items() if now - v.get("finished", now) > FINISHED_TTL]:
        tasks.pop(key, None)
    return response


async def _refund(request: Request, pending: dict) -> None:
    """The try back, once per task, when no model comes of it."""
    owner = pending.get("user_id")
    if owner and not pending.get("refunded"):
        pending["refunded"] = True
        await bus.publish(GenerationFailed(owner, "meshy"), request.app.state)


async def _clean_failed(request: Request, task_id: str, pending: dict, reason: str, progress: int) -> TaskStatusResponse:
    logger.warning("Image cleaning %s failed (%s): 3D generation was not started", task_id, reason)
    await _refund(request, pending)
    return _finish(request, task_id, pending, _status(
        task_id, pending, "failed", status="FAILED", progress=progress,
        error=f"{IMAGE_FAILED} {reason}".strip() if reason else IMAGE_FAILED,
        steps=_steps(("FAILED", progress), ("PENDING", 0)),
    ))


async def _start_mesh(request: Request, meshy: MeshyService, task_id: str, pending: dict, source: str, image: bytes | None = None,
                      url: str | None = None, input_task_id: str | None = None) -> TaskStatusResponse:
    """Image to 3D from the cleaned image (its Image to Image task, no re-upload) or the original photo."""
    mesh, mime = pending["mesh"], (sniff_image_mime(image) if image else None) or pending.get("mime") or "image/png"
    if source == "cleaned" and (not input_task_id or input_task_id != pending.get("image_task_id")):
        # never mesh "the cleaned image" from anything but the Image to Image task that produced it
        raise RuntimeError(f"cleaned handoff without its Image to Image task ({input_task_id!r} != {pending.get('image_task_id')!r})")
    record = meshy_input(image, mime, source, mesh, url, input_task_id)
    logger.info("Meshy image-to-3d for %s: image_source=%s image_to_image_task=%s input_task_id=%s", task_id,
                "input_task_id" if input_task_id else ("image_url (original, URL)" if url else "image_url (original, data URI)"),
                pending.get("image_task_id"), input_task_id)  # ids only: no signed URL, no image data
    try:
        mesh_id = await meshy.create_image_to_3d_task(
            # PBR: metallic, roughness and normal maps besides the colour, so gold handles stay metal.
            GenerationOptions(should_texture=mesh["textured"], enable_pbr=mesh["textured"]),
            image=None if input_task_id else image, mime_type=mime, image_url=None if input_task_id else url, input_task_id=input_task_id,
            texture_prompt=mesh["texture_prompt"] if mesh["textured"] else None, target_polycount=mesh["target_polycount"],
        )
    except MeshyError as exc:
        await _refund(request, pending)  # nothing more will come of it
        return _finish(request, task_id, pending, _status(task_id, pending, "failed", status="FAILED", progress=IMAGE_STAGE, error=exc.message,
                                                          steps=_steps((pending.get("image_step_status") or "SUCCEEDED", 100), ("FAILED", 0))))
    logger.info("Meshy image-to-3d task %s created for %s from %s", mesh_id, task_id, input_task_id or "the original photo")
    pending.update(stage="mesh", mesh_task_id=mesh_id, image_step=True, meshy_input=record, meshy_input_bytes=image)
    return _status(task_id, pending, "mesh_generation", status="IN_PROGRESS", progress=IMAGE_STAGE, retry_after_seconds=5,
                   steps=_mesh_steps(pending, "IN_PROGRESS", 0))


async def _use_original(request: Request, meshy: MeshyService, task_id: str, pending: dict, what: str, reason: str) -> TaskStatusResponse:
    """The configured fallback when cleaning failed or its image was rejected: the original photo is meshed and the
    model says so (meshy_use_original_on_bad_cleaning), or the conversion stops and the try is refunded."""
    photo, url = pending.get("photo"), pending.get("image_url")
    if not (get_settings().meshy_use_original_on_bad_cleaning and (photo or url)):
        return await _clean_failed(request, task_id, pending, reason, 0)
    logger.warning("Meshy image cleaning %s %s (%s): meshing the original photo", task_id, what, reason[:200])
    pending["warnings"].append(cleaning_warning(what, reason))
    pending["cleaning_fallback"] = {"what": what, "reason": reason}
    pending["image_step_status"] = "SKIPPED"
    pending.pop("clean_image_url", None)
    pending.pop("reference", None)
    return await _start_mesh(request, meshy, task_id, pending, "original", image=photo, url=None if photo else url)


def _task_error(task: dict) -> str:
    err = task.get("task_error") or task.get("error")
    return str((err.get("message") if isinstance(err, dict) else err) or "")


async def conditioned_image(request: Request, meshy: MeshyService, cdn: Cdn | None, task_id: str, pending: dict) -> TaskStatusResponse:
    """The Meshy Image to Image step, one poll at a time (the caller holds the task's lock):
    image_preprocessing — the Image to Image task runs; image_validation — its one image is downloaded, checked as an
    image and against the original (fidelity.compare_images, removal mode), then Image to 3D starts from that task."""
    if pending["stage"] == "image":
        try:
            task = await meshy.get_image_to_image_task(pending["image_task_id"])
        except MeshyError as exc:
            if exc.status_code in TRANSIENT:  # rate limited / Meshy busy: ask again later, nothing is lost
                return _status(task_id, pending, "image_preprocessing", status="IN_PROGRESS", progress=0, retry_after_seconds=exc.retry_after or 10,
                               steps=_steps(("IN_PROGRESS", 0), ("PENDING", 0)))
            return await _use_original(request, meshy, task_id, pending, "failed", exc.message)
        state = str(task.get("status") or "")
        if state in ("PENDING", "IN_PROGRESS", ""):
            if time.monotonic() - float(pending.get("clean_started") or time.monotonic()) > get_settings().meshy_clean_timeout_seconds:
                return await _use_original(request, meshy, task_id, pending, "timed out", "Meshy Image to Image took too long")
            progress = max(0, min(99, int(task.get("progress") or 0)))
            return _status(task_id, pending, "image_preprocessing", status="IN_PROGRESS", progress=progress * IMAGE_STAGE // 100, retry_after_seconds=4,
                           steps=_steps(("IN_PROGRESS", progress), ("PENDING", 0)))
        if state != "SUCCEEDED":  # FAILED, CANCELED, or anything undocumented
            return await _use_original(request, meshy, task_id, pending, "failed", _task_error(task) or f"Meshy Image to Image ended {state}")
        urls = [u for u in task.get("image_urls") or [] if isinstance(u, str) and u]
        if len(urls) != 1:  # Image to 3D's input_task_id takes a task with exactly one image
            return await _use_original(request, meshy, task_id, pending, "failed", f"Meshy Image to Image returned {len(urls)} images, not one")
        pending.update(stage="validation", clean_url=urls[0])
        return _status(task_id, pending, "image_validation", status="IN_PROGRESS", progress=IMAGE_STAGE, retry_after_seconds=1,
                       steps=_steps(("SUCCEEDED", 100), ("PENDING", 0)))

    cleaned = await meshy.download_bytes(pending.pop("clean_url", "") or "")
    if not cleaned:
        return await _use_original(request, meshy, task_id, pending, "rejected", "The cleaned image could not be downloaded")
    if sniff_image_mime(cleaned) not in ALLOWED_IMAGE_TYPES:
        return await _use_original(request, meshy, task_id, pending, "rejected", "The cleaned image is not a PNG, JPG or WEBP")
    photo = pending.get("photo")
    check = await run_in_threadpool(fidelity.compare_images, photo, cleaned, (pending.get("details") or {}).get("bbox"), True) if photo else \
        {"stage": "cleaning", "mode": "removal", "ok": None, "issues": [], "notes": ["Only an image URL was given: the cleaned image was not compared with the original."]}
    pending["cleaning_check"] = check
    if check.get("ok") is False:
        check["rejected"] = True
        pending["rejected_clean"] = cleaned
        return await _use_original(request, meshy, task_id, pending, "rejected", "The cleaned image changed the furniture: " + "; ".join(check["issues"][:2]))
    # No photo reading (the default Meshy path) means no box to compare layout in: that is recorded in the check's notes,
    # not raised as a warning on every model. Any other unjudged check is.
    if check.get("ok") is None and (not photo or (pending.get("details") or {}).get("bbox")):
        pending["warnings"].append("The cleaned image was used but could not be fully checked against the original photo: " + "; ".join(check.get("notes", [])[:1]))
    # The cleaned image is the reference from here on (grouping, appearance): the photo still shows what was removed.
    pending["reference"] = cleaned
    pending["image_step_status"] = "SUCCEEDED"
    if cdn is not None:
        try:
            pending["clean_image_url"] = await run_in_threadpool(cdn.upload, f"{pending.get('user_id')}/{pending.get('model_id') or task_id}.clean.png", cleaned,
                                                                 sniff_image_mime(cleaned) or "image/png")
        except CdnError as exc:
            logger.warning("Cleaned image not uploaded: %s", exc.message)
    return await _start_mesh(request, meshy, task_id, pending, "cleaned", image=cleaned, input_task_id=pending["image_task_id"])


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(
    task_id: str,
    request: Request,
    meshy: Annotated[MeshyService, Depends(get_meshy)],
    cdn: Annotated[Cdn, Depends(get_cdn)],
    user_id: Annotated[str, Depends(current_user)],
    background: BackgroundTasks,
) -> TaskStatusResponse:
    """image_preprocessing → image_validation → mesh_generation → auto_split → postprocessing → completed (or failed).
    One poll at a time per task (its lock): concurrent polls wait, then get the same answer, so no Meshy task is
    created twice and the model is processed once."""
    registered = task_id in request.app.state.pending_cdn
    pending = request.app.state.pending_cdn.get(task_id) or {}
    if pending.get("user_id") and pending["user_id"] != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found.")
    async with pending.setdefault("_lock", asyncio.Lock()):
        if "result" in pending:
            return pending["result"]
        if pending.get("stage") in ("image", "validation"):
            return await conditioned_image(request, meshy, cdn, task_id, pending)
        return await _mesh_task(task_id, request, meshy, cdn, user_id, background, pending, registered)


async def _mesh_task(task_id: str, request: Request, meshy: MeshyService, cdn: Cdn, user_id: str, background: BackgroundTasks,
                     pending: dict, registered: bool) -> TaskStatusResponse:
    settings = get_settings()
    mesh_id = pending.get("mesh_task_id") or task_id  # without a pending entry: an Image to 3D id from before the chain
    try:
        task, retry_after = await meshy.get_image_to_3d_task(mesh_id)
    except MeshyError as exc:
        if registered and exc.status_code in TRANSIENT:
            return _status(task_id, pending, "mesh_generation", status="IN_PROGRESS", progress=IMAGE_STAGE if pending.get("image_step") else 0,
                           retry_after_seconds=exc.retry_after or 10, steps=_mesh_steps(pending, "IN_PROGRESS", 0))
        raise HTTPException(exc.status_code, exc.message) from exc

    status_value = str(task.get("status") or "IN_PROGRESS")
    progress = int(task.get("progress") or 0)
    if status_value in {"FAILED", "CANCELED", "EXPIRED"}:
        if registered:
            await _refund(request, pending)  # no model: the try is given back (once)
        failed = _status(task_id, pending, "failed", status="FAILED" if status_value == "EXPIRED" else status_value, progress=progress,
                         error=_task_error(task) or "Generation failed.", steps=_mesh_steps(pending, "FAILED", progress))
        return _finish(request, task_id, pending, failed) if registered else failed
    splitting = bool(pending.get("split"))
    if status_value != "SUCCEEDED":
        # A task that went through the image step already used the first 20%. A direct Image to 3D task has not.
        base = IMAGE_STAGE if pending.get("image_step") else 0
        return _status(task_id, pending, "mesh_generation", status=status_value, progress=base + progress * ((90 if splitting else 95) - base) // 100,
                       retry_after_seconds=retry_after or 5, steps=_mesh_steps(pending, "IN_PROGRESS", progress))

    # Auto Split (by_parts, the furniture type's parts): its result is checked against the raw model below and used
    # only when it really is that model in several meshes. Its actual status, error and prompt_ignored are kept.
    split_glb_url = None
    split_task: dict | None = None
    if splitting:
        split_task = {"prompt": pending["split"], "status": None}
        try:
            if not pending.get("split_task_id"):
                pending["split_task_id"] = await meshy.create_split_task(mesh_id, pending["split"], bool(pending.get("textured")))
            split = await meshy.get_split_task(pending["split_task_id"])
        except MeshyError as exc:
            if exc.status_code in TRANSIENT and pending.get("split_task_id") and registered:
                return _status(task_id, pending, "auto_split", status="IN_PROGRESS", progress=90, retry_after_seconds=exc.retry_after or 10,
                               steps=_mesh_steps(pending, "SUCCEEDED", 100))
            logger.warning("Meshy Auto Split for %s failed: %s", task_id, exc.message)
            split_task.update(status="REQUEST_FAILED", error=exc.message, taskId=pending.get("split_task_id"))
            pending["split"] = None
            pending["split_failed"] = split_task
        else:
            split_status = str(split.get("status"))
            if split_status in {"PENDING", "IN_PROGRESS"}:
                return _status(task_id, pending, "auto_split", status="IN_PROGRESS", progress=90 + int(split.get("progress") or 0) // 20, retry_after_seconds=5,
                               steps=_mesh_steps(pending, "SUCCEEDED", 100))
            split_glb_url = (split.get("model_urls") or {}).get("glb") if split_status == "SUCCEEDED" else None
            err = split.get("task_error")
            split_task.update(status=split_status, taskId=pending["split_task_id"], partCount=split.get("part_count"), promptIgnored=split.get("prompt_ignored"),
                              error=(err.get("message") if isinstance(err, dict) else err) or None)
            logger.info("Meshy Auto Split %s: status=%s part_count=%s prompt_ignored=%s error=%s", pending["split_task_id"], split_status, split.get("part_count"), split.get("prompt_ignored"), err)
    elif pending.get("split_failed"):
        split_task = pending["split_failed"]
    # Everything Meshy makes is ready: say so (one short poll) before the processing request, which takes a while.
    if registered and not pending.get("finalizing"):
        pending["finalizing"] = True
        return _status(task_id, pending, "postprocessing", status="IN_PROGRESS", progress=96, retry_after_seconds=1,
                       steps=_mesh_steps(pending, "SUCCEEDED", 100))
    owner = pending.get("user_id") or user_id
    object_id = pending.get("model_id") or task_id
    model_urls = task.get("model_urls") or {}
    glb_url = model_urls.get("glb")
    if not glb_url:
        if registered:
            await _refund(request, pending)
        failed = _status(task_id, pending, "failed", status="FAILED", progress=100, error="Meshy finished but did not return a GLB.",
                         steps=_mesh_steps(pending, "FAILED", 100))
        return _finish(request, task_id, pending, failed) if registered else failed
    allow_vision = settings.meshy_vision_analysis  # downstream vision calls only with the optional analysis
    dest = request.app.state.storage.models_dir / f"{task_id}.glb"
    try:
        split_dest = dest.with_suffix(".split.glb")
        # The model and its split at the same time (tens of MB each).
        await asyncio.gather(meshy.download_file(glb_url, dest), *([meshy.download_file(split_glb_url, split_dest)] if split_glb_url else []))
        data = dest.read_bytes()
        raw_glb = data  # the provider's file before any change, kept full quality (raw.glb) and compared with
        background.add_task(upload_raw, raw_glb, cdn, f"{owner}/{object_id}.raw.glb")
        components = None
        split_report = None
        if split_task is not None:
            split_bytes = split_dest.read_bytes() if split_glb_url else None
            split_report = await run_in_threadpool(fidelity.split_check, raw_glb, split_bytes, split_task)
            logger.info("Meshy Auto Split check for %s: accepted=%s meshes %s → %s %s", task_id, split_report["accepted"], split_report.get("meshesBefore"), split_report.get("meshesAfter"), split_report["issues"])
            if split_report["accepted"]:
                info = await run_in_threadpool(inspect_glb, split_bytes, furniture_family(pending.get("category"), pending.get("kind") or None))
                data, source_glb, components = split_bytes, raw_glb, {**info, "method": "meshy_auto_split"}  # the unsplit original kept as source
        data, surfaces = await run_in_threadpool(surface_finish.flatten, data)  # a no-op on UV atlases (not tiled)
        if components is None:
            data, source_glb, components = await run_in_threadpool(meshy_parts, data, pending.get("category"), pending.get("kind"))
            if split_report and not split_report["accepted"]:
                source_glb = None
                components = {**components, "limits": " ".join(x for x in (components.get("limits"), "Auto Split was not used: " + " ".join(split_report["issues"][:2])) if x)}
        if surfaces:
            components = {**components, "surfaces": surfaces}
        components = analysis_checked(components, pending)
        reference = pending.get("reference") or pending.get("photo")
        data, source_glb, components = await grouped(request, data, source_glb, components, reference, pending.get("details"), allow_vision)
        data, components = await named(request, data, components, pending.get("category"), pending.get("kind"), allow_vision)
        data, components, ar_job = await quality_checked(data, components, pending.get("details"))
        components = schedule_ar(background, ar_job, cdn, owner, object_id, components)
        data, components = await run_in_threadpool(animation.configure, data, components)  # movable parts, estimated interior
        warnings = list(pending.get("warnings") or [])
        if pending.get("conditioning"):
            warnings.append(pending["conditioning"])
        if warnings:  # said, not hidden: what did not go as asked
            components = {**components, "limits": " ".join(x for x in (components.get("limits"), *warnings) if x)}
        data, source_glb, components = await sized(data, source_glb, components)
        # The photo reading's box is in the original photo: the cleaned image is framed differently, so no box for it.
        bbox = None if pending.get("reference") else (pending.get("details") or {}).get("bbox")
        components = await run_in_threadpool(validated, data, components, pending.get("category"), pending.get("kind"), reference, bbox)
        if warnings:
            components["validation"] = {**components["validation"], "warnings": [*warnings, *components["validation"].get("warnings", [])]}
        components = {**components, "cleaning": await run_in_threadpool(cleaning_record, pending, cdn, f"{owner}/{object_id}.clean.png")}
        # Built from the spec when the task was submitted (older tasks without one: built now, from the same spec).
        built = pending.get("structural") if "structural" in pending else await run_in_threadpool(structural_model, pending.get("spec"), pending.get("category"), pending.get("kind"), cdn, f"{owner}/{object_id}.carpenter.glb")
        components = {**with_structural(components, built, pending.get("spec")), "raw_glb_url": cdn.public_url(f"{owner}/{object_id}.raw.glb")}
        details = pending.get("details") or {}
        components = await run_in_threadpool(fidelity_checked, components, raw_glb, data, pending.get("photo"), pending.get("reference"), details.get("bbox"),
                                             pending.get("cleaning_check"), split_report, details.get("proportions"), True)
        components = {**components, "meshyInput": await run_in_threadpool(meshy_input_uploaded, pending.get("meshy_input"), pending.get("meshy_input_bytes"), cdn, f"{owner}/{object_id}.meshy-input"),
                      **({"split": split_report} if split_report else {})}
        url = await run_in_threadpool(cdn.upload, f"{owner}/{object_id}.glb", data, "model/gltf-binary")
        source_url = schedule_source(background, source_glb, cdn, f"{owner}/{object_id}.source.glb")
    except (MeshyError, CdnError) as exc:
        pending.pop("finalizing", None)
        raise HTTPException(getattr(exc, "status_code", 502), str(getattr(exc, "message", exc))) from exc
    if pending.get("cache_key"):
        result = {"glb_url": url, "model_id": object_id, "details": pending.get("details"), "components": components, "source_glb_url": source_url}
        await vision.cache_put(settings, request.app.state.auth_http, pending["cache_key"], "vendor_glb", "meshy", result, table="model_cache")
    done = _status(task_id, pending, "completed", status="SUCCEEDED", progress=100, glb_url=url, components=components, source_glb_url=source_url,
                   steps=_mesh_steps(pending, "SUCCEEDED", 100))
    return _finish(request, task_id, pending, done) if registered else done


class StructuralBody(BaseModel):
    """A corrected spec (sizes, parts, materials) for a model already generated: only the structural model is rebuilt."""

    model_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    details: dict = Field(default_factory=dict)  # width_m, height_m, depth_m, dimension_source, proportions, category
    analysis: dict  # the spec's components (services/mesh_analysis.MeshAnalysis)
    category: str = Field(default="furniture", max_length=40)
    kind: str = Field(default="", max_length=60)


@router.post("/structural")
async def rebuild_structural(body: StructuralBody, cdn: Annotated[Cdn, Depends(get_cdn)], user_id: Annotated[str, Depends(current_user)]) -> dict:
    """The structural model again from an edited spec: no vision call and no new 3D generation (the visual model is
    untouched). The file gets a new name per spec, so no viewer shows a cached older one."""
    try:
        analysis = mesh_analysis.MeshAnalysis.model_validate(body.analysis).model_dump(mode="json")
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"The spec's components are invalid: {str(exc)[:200]}") from exc
    spec = {"details": body.details, "analysis": analysis}
    tag = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    report = await run_in_threadpool(structural_model, spec, body.category, body.kind, cdn, f"{user_id}/{body.model_id}.carpenter.{tag}.glb")
    return with_structural({}, report, spec)


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
    settings: SettingsDep,
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
