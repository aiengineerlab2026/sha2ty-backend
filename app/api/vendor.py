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
from app.services.generation_prompt import astra_prompt, furniture_family, furniture_type, object_description, meshy_prompt, meshy_texture_prompt, split_prompt, target_polycount
from app.services.glb_size import MAX_GLB_BYTES, fit_glb
from app.services.glb_parts import inspect_glb, segment_glb, write_names
from app.services.model_check import check_glb, check_states, with_appearance
from app.services.part_names import name_components
from app.services.platform import is_platform
from app.services.product_details import in_language
from app.services import fidelity, furniture_spec, material_spec, structural
from app.services.cdn import Cdn, CdnError
from app.schemas.vendor import GenerationOptions
from app.services import animation, component_grouping, geometry_quality, mesh_analysis
from app.services.meshy_service import MeshyError, MeshyService
from app.services import generator_setting
from app.services.quota import generation_usage, use_generation

router = APIRouter(prefix="/api/vendor", tags=["vendor"])
logger = logging.getLogger(__name__)
GLB_CACHE_VERSION = "vendor-glb-v7"  # v7: one vision spec, structural model from it, raw GLB kept (v6: animation)

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
    No size limit: a model still over the target is saved as it is. The untouched original is compacted with its
    upload, after the response (schedule_source)."""
    glb, report = await run_in_threadpool(fit_glb, glb)
    if report["after"] > MAX_GLB_BYTES:
        logger.info("Generated model kept at %.1f MB (over the %d MB target)", report["after"] / 1048576, MAX_GLB_BYTES // 1048576)
    return glb, source_glb, {**components, "file": {"bytes": report["after"], "originalBytes": report["before"], "optimised": report["steps"]}}


async def named(request: Request, glb: bytes, doc: dict, category: str | None, kind: str | None) -> tuple[bytes, dict]:
    """Descriptive part names (services/part_names.py), saved in the component document and in the GLB itself."""
    settings = get_settings()
    model = await vision.current_model(settings, request.app.state.auth_http)

    async def ask(prompt: str, schema: dict) -> str | None:
        # Geometry names every part without a call; the vision model only when the admin turned it on.
        if not settings.visual_part_ai or not vision.configured(settings, model):
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


async def grouped(request: Request, glb: bytes, source_glb: bytes | None, doc: dict, photo: bytes | None, details: dict | None) -> tuple[bytes, bytes | None, dict]:
    """The generated meshes grouped into carpenter-named components (services/component_grouping.py), checked
    against the geometry. Anything unsafe or unidentified keeps the model as it was; the file before grouping is
    kept as the source (the fallback) when there was none."""
    settings = get_settings()
    model = await vision.current_model(settings, request.app.state.auth_http)
    ask = None
    if settings.visual_part_ai and vision.configured(settings, model):  # an extra vision call: off unless the admin turns it on
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


def fidelity_checked(components: dict, raw: bytes, glb: bytes, photo: bytes | None, cleaned: bytes | None, bbox: list[float] | None) -> dict:
    """Where the visual model lost fidelity (services/fidelity.py), stage by stage. A failed stage flags the model
    (quality "low", its issues in the validation warnings) instead of passing it as a success."""
    stages = []
    if cleaned and photo:
        stages.append(fidelity.compare_images(photo, cleaned, bbox))
    stages.append(fidelity.provider(raw, photo, bbox, cleaned))
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
    """What the image-cleaning step did: its prompt and the cleaned image, copied to the CDN."""
    record: dict = {"prompt": pending.get("clean_prompt"), "cleaned": bool(pending.get("clean_image_url"))}
    if pending.get("reference"):
        try:
            record["image_url"] = cdn.upload(key, pending["reference"], "image/png")
        except CdnError as exc:
            logger.warning("Cleaned reference image not uploaded: %s", exc.message)
    return record


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
        components = {**components, "limits": " ".join(x for x in (components.get("limits"), "The photo was not analysed into components before generation (no vision model answer).") if x)}
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


def _log_clean_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.warning("Vision image cleaning failed: %s", getattr(exc, "message", exc))


@router.post("/generate")
async def generate_model(
    request: Request,
    background: BackgroundTasks,
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
    prespec: furniture_spec.FurnitureSpec | None = None
    if category == "auto":
        prespec = await read_spec(request, settings, raw, mime, "auto", title, kind, lang) if raw else None
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
            glb, source_glb, components = await run_in_threadpool(modular, glb, category, kind)
            components = mesh_analysis.match(components, spec.analysis if spec else None)
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
    # The listing reading and the component analysis read the same photo independently: asked at the same time
    # (each takes 30–60 s), not one after the other.
    # One vision call (cached): the listing, the components and what is not part of the product.
    spec = prespec or (await read_spec(request, settings, raw, mime, category, title, kind, lang) if raw else None)
    details = spec.details if spec else preread
    analysis = spec.analysis if spec else None
    spec_doc = spec.model_dump(mode="json") if spec else None
    # Original photo → the configured vision model cleans it (loose objects and contents removed) → Image to 3D from
    # that image → Auto Split → grouping, naming, checks. Image to 3D has no geometry prompt, so what is removed
    # is removed in the image. A public URL is downloaded here: the vision model needs the bytes.
    kind_type = furniture_type(category, kind or None, details)
    # Cleaning is another image-model call that redraws the photo: only when the reading found loose objects on or in
    # the product. A clean product photo goes to Image to 3D as it is (the most faithful input).
    clean_prompt = meshy_prompt(category, kind or None, details) if settings.meshy_clean_image and spec and spec.needs_cleaning else None
    if clean_prompt:
        clean_prompt += f" In this photo, remove: {', '.join(spec.loose_objects[:12])}."
    clean_model = None
    # Vision mesh analysis first: the photo's components shape the prompt. Meshy takes it where it can act on it
    # (texture_prompt, Auto Split's part list); `conditioning` records where each part went, nothing is dropped.
    base_texture = meshy_texture_prompt(category, title, details, kind or None)
    split = split_prompt(category, kind or None)
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
    stage = "mesh"
    task_id = None
    clean_job = None
    if clean_prompt:
        clean_model = await vision.current_model(settings, request.app.state.auth_http)
        photo = raw
        try:
            if not photo and public_url:
                photo = await vision.fetch_image(public_url)
            if not photo or not vision.configured(settings, clean_model):
                raise vision.VisionImageError("The vision model is not configured." if not vision.configured(settings, clean_model) else "Provide a photo to clean.")
        except vision.VisionImageError as exc:
            logger.warning("Image cleaning did not start (%s): 3D generation was not started", exc.message)
            await bus.publish(GenerationFailed(user_id, "meshy"), request.app.state)
            raise HTTPException(status.HTTP_400_BAD_REQUEST, IMAGE_FAILED) from exc
        task_id = uuid4().hex
        clean_job = asyncio.create_task(vision.clean_image(
            clean_prompt, photo, mime, settings, vision.client_for(request.app.state, clean_model), clean_model,
        ))
        clean_job.add_done_callback(_log_clean_failure)
        stage = "image"
    try:
        if task_id is None:
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
    logger.info("%s %s: furniture_type=%r category=%r analysed=%s split=%r vision_model=%s loose=%s", "Vision image cleaning" if stage == "image" else "Meshy image-to-3d", task_id, kind_type, category, bool(analysis), split, clean_model, spec.loose_objects if spec else None)
    # The structural model needs only the spec: built now, while the provider works on the visual one.
    built = await run_in_threadpool(structural_model, spec_doc, category, kind, cdn, f"{user_id}/{object_id}.carpenter.glb")
    request.app.state.pending_cdn[task_id] = {
        "stage": stage,
        "mesh_task_id": task_id if stage == "mesh" else None,
        "clean_prompt": clean_prompt,
        "clean_job": clean_job,
        "clean_started": time.monotonic() if clean_job else None,
        "mesh": mesh,
        "mime": mime,
        "image_url": public_url,
        "user_id": user_id, "model_id": object_id, "cache_key": cache_key,
        "details": details.model_dump(mode="json") if details else None,  # cached with the GLB when Meshy finishes
        "category": category,
        "kind": kind,
        "split": split if settings.meshy_auto_split else None,  # Auto Split parts
        "analysis": analysis.model_dump(mode="json") if analysis else None,
        "meshy_conditioning": conditioning,
        "spec": spec_doc,
        "structural": built,
        "textured": textured,
        "photo": raw or None,  # the reference for the appearance check when the model arrives (dropped with the task)
    }
    if clean_job:
        await asyncio.sleep(0)  # the cleaning request starts before this response is sent
    return GenerateAccepted(task_id=task_id, provider="meshy", remaining=remaining, details=details)


IMAGE_STAGE = 20  # % of the progress bar the Image to Image step takes
IMAGE_FAILED = "Image to image failed, so the 3D model was not generated."


def _steps(image: tuple[str, int] | None, mesh: tuple[str, int] | None) -> list[GenerationStep] | None:
    rows = []
    if image:
        rows.append(GenerationStep(id="image", status=image[0], progress=image[1]))
    if mesh:
        rows.append(GenerationStep(id="mesh", status=mesh[0], progress=mesh[1]))
    return rows or None


def _mesh_steps(pending: dict, status: str, progress: int) -> list[GenerationStep] | None:
    """Image to 3D, with the cleaning step marked done when this task went through it."""
    image = ("SUCCEEDED", 100) if pending.get("image_step") or pending.get("clean_prompt") else None
    return _steps(image, (status, progress))


async def _clean_failed(request: Request, task_id: str, pending: dict, reason: str, progress: int) -> TaskStatusResponse:
    logger.warning("Image cleaning %s failed (%s): 3D generation was not started", task_id, reason)
    owner = pending.get("user_id") or ""
    request.app.state.pending_cdn.pop(task_id, None)
    if owner:
        await bus.publish(GenerationFailed(owner, "meshy"), request.app.state)
    return TaskStatusResponse(
        task_id=task_id, status="FAILED", progress=progress,
        error=f"{IMAGE_FAILED} {reason}".strip() if reason else IMAGE_FAILED,
        steps=_steps(("FAILED", progress), ("PENDING", 0)),
    )


async def conditioned_image(request: Request, meshy: MeshyService, task_id: str, pending: dict) -> TaskStatusResponse | None:
    """The vision model's cleaning step. While it runs, report progress. When it returns one image, Image to 3D
    starts from those bytes. A failed cleaning step stops the conversion: 3D is not generated from the original photo."""
    lock = pending.setdefault("_clean_lock", asyncio.Lock())
    async with lock:
        if pending.get("stage") != "image":
            return None
        job = pending.get("clean_job")
        if not isinstance(job, asyncio.Task):
            return await _clean_failed(request, task_id, pending, "Image cleaning did not start.", 0)
        if not job.done():
            elapsed = time.monotonic() - float(pending.get("clean_started") or time.monotonic())
            image_progress = min(95, int(elapsed / 30 * 100))
            return TaskStatusResponse(
                task_id=task_id, status="IN_PROGRESS", progress=image_progress * IMAGE_STAGE // 100, retry_after_seconds=4,
                steps=_steps(("IN_PROGRESS", image_progress), ("PENDING", 0)),
            )
        exc = job.exception()
        cleaned = None if exc else job.result()
        if exc or not isinstance(cleaned, (bytes, bytearray)) or not cleaned:
            reason = getattr(exc, "message", None) or (str(exc) if exc else "The vision model returned no image.")
            return await _clean_failed(request, task_id, pending, reason, 0)
        pending["clean_image_url"] = "vision"
        # The cleaned image is the reference from here on (grouping, appearance): the photo still shows what was removed.
        pending["reference"] = bytes(cleaned)
        mesh = pending["mesh"]
        try:
            pending["mesh_task_id"] = await meshy.create_image_to_3d_task(
                # PBR: metallic, roughness and normal maps besides the colour, so gold handles stay metal (glb_size keeps the
                # file within its limit). The cleaned image is the only input.
                GenerationOptions(should_texture=mesh["textured"], enable_pbr=mesh["textured"]),
                image=bytes(cleaned),
                mime_type=pending.get("mime") or "image/png",
                texture_prompt=mesh["texture_prompt"],
                target_polycount=mesh["target_polycount"],
            )
        except MeshyError as exc:
            if request.app.state.pending_cdn.pop(task_id, None) is not None:  # nothing more will come of it: the try is given back
                await bus.publish(GenerationFailed(pending.get("user_id") or "", "meshy"), request.app.state)
            return TaskStatusResponse(
                task_id=task_id, status="FAILED", progress=IMAGE_STAGE, error=exc.message,
                steps=_steps(("SUCCEEDED", 100), ("FAILED", 0)),
            )
        pending["stage"] = "mesh"
        pending["image_step"] = True  # this task spent the first part of the bar on image cleaning
        return TaskStatusResponse(
            task_id=task_id, status="IN_PROGRESS", progress=IMAGE_STAGE, retry_after_seconds=5,
            steps=_steps(("SUCCEEDED", 100), ("IN_PROGRESS", 0)),
        )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(
    task_id: str,
    request: Request,
    meshy: Annotated[MeshyService, Depends(get_meshy)],
    cdn: Annotated[Cdn, Depends(get_cdn)],
    user_id: Annotated[str, Depends(current_user)],
    background: BackgroundTasks,
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
            steps=_mesh_steps(pending, "FAILED", progress),
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
            steps=_mesh_steps(pending, "IN_PROGRESS", progress),
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
                return TaskStatusResponse(task_id=task_id, status="IN_PROGRESS", progress=90 + int(split.get("progress") or 0) // 10, retry_after_seconds=5, steps=_mesh_steps(pending, "IN_PROGRESS", 100))
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
        split_dest = dest.with_suffix(".split.glb")
        # The model and its split at the same time (tens of MB each).
        await asyncio.gather(meshy.download_file(glb_url, dest), *([meshy.download_file(split_glb_url, split_dest)] if split_glb_url else []))
        data = dest.read_bytes()
        raw_glb = data  # the provider's file before any change, kept and compared with
        background.add_task(upload_raw, raw_glb, cdn, f"{owner}/{object_id}.raw.glb")
        components = None
        if split_glb_url:
            info = await run_in_threadpool(inspect_glb, split_dest.read_bytes(), furniture_family(pending.get("category"), pending.get("kind") or None))
            logger.info("Meshy Auto Split GLB for %s: separated=%s components=%s", task_id, info.get("separated"), [c["name"] for c in info.get("components", [])])
            if info.get("separated"):  # independently editable meshes in the file, not just a reported count
                data, source_glb, components = split_dest.read_bytes(), data, {**info, "method": "meshy_auto_split"}
        if components is None:
            data, source_glb, components = await run_in_threadpool(modular, data, pending.get("category"), pending.get("kind"))
        components = analysis_checked(components, pending)
        reference = pending.get("reference") or pending.get("photo")
        data, source_glb, components = await grouped(request, data, source_glb, components, reference, pending.get("details"))
        data, components = await named(request, data, components, pending.get("category"), pending.get("kind"))
        data, components, ar_job = await quality_checked(data, components, pending.get("details"))
        components = schedule_ar(background, ar_job, cdn, owner, object_id, components)
        data, components = await run_in_threadpool(animation.configure, data, components)  # movable parts, estimated interior
        if pending.get("conditioning"):  # said, not hidden: the type did not shape this model's image
            components = {**components, "limits": " ".join(x for x in (components.get("limits"), pending["conditioning"]) if x)}
        data, source_glb, components = await sized(data, source_glb, components)
        # The photo reading's box is in the original photo: the cleaned image is framed differently, so no box for it.
        bbox = None if pending.get("reference") else (pending.get("details") or {}).get("bbox")
        components = await run_in_threadpool(validated, data, components, pending.get("category"), pending.get("kind"), reference, bbox)
        components = {**components, "cleaning": await run_in_threadpool(cleaning_record, pending, cdn, f"{owner}/{object_id}.clean.png")}
        # Built from the spec when the task was submitted (older tasks without one: built now, from the same spec).
        built = pending.get("structural") if "structural" in pending else await run_in_threadpool(structural_model, pending.get("spec"), pending.get("category"), pending.get("kind"), cdn, f"{owner}/{object_id}.carpenter.glb")
        components = {**with_structural(components, built, pending.get("spec")), "raw_glb_url": cdn.public_url(f"{owner}/{object_id}.raw.glb")}
        components = await run_in_threadpool(fidelity_checked, components, raw_glb, data, pending.get("photo"), pending.get("reference"), (pending.get("details") or {}).get("bbox"))
        url = await run_in_threadpool(cdn.upload, f"{owner}/{object_id}.glb", data, "model/gltf-binary")
        source_url = schedule_source(background, source_glb, cdn, f"{owner}/{object_id}.source.glb")
    except (MeshyError, CdnError) as exc:
        raise HTTPException(getattr(exc, "status_code", 502), str(getattr(exc, "message", exc))) from exc
    if pending.get("cache_key"):
        result = {"glb_url": url, "model_id": object_id, "details": pending.get("details"), "components": components, "source_glb_url": source_url}
        await vision.cache_put(get_settings(), request.app.state.auth_http, pending["cache_key"], "vendor_glb", "meshy", result, table="model_cache")
    request.app.state.pending_cdn.pop(task_id, None)
    return TaskStatusResponse(task_id=task_id, status="SUCCEEDED", progress=100, glb_url=url, components=components, source_glb_url=source_url)


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
