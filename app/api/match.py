"""POST /api/match: a photorealistic room with the catalogue model and the shopper's item, made by
the admin's GPT (Astra or Sol) from both photos (see app/services/image_match.py)."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from app.api.plans import store
from app.core.config import get_settings
from app.services import image_match as im
from app.services import generator_setting
from app.services.auth import service_headers, user_id_from_bearer

router = APIRouter(prefix="/api/match", tags=["match"])

MAX_UPLOAD = 15 * 1024 * 1024
ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
KEEP_SECONDS = 3600  # ponytail: results live an hour on local disk; move to the CDN if they must persist

VIEW_LABEL = ["Eye-level view", "Corner view", "Opposite corner view", "Wide view", "Elevated view"]
PLACEMENT_LABEL = {
    ("furniture", "left"): "Furniture to the left of the model",
    ("furniture", "right"): "Furniture to the right of the model",
    ("furniture", "front"): "Furniture in front of the model",
    ("furniture", "behind"): "Furniture behind the model",
    ("carpet", None): "Carpet underneath the model",
    ("wall_paint", "behind"): "Wall behind the model painted",
    ("wall_paint", "left"): "Left wall painted",
    ("wall_paint", "right"): "Right wall painted",
    ("wall_paint", "front"): "Wall facing the model painted",
}

OTHER_TYPE = {"model": "furniture", "design": "furniture", "user_model": "furniture", "carpet": "carpet", "wall": "wall_paint"}
DEFAULT_WALL_CM = (300.0, 260.0)  # ponytail: a paint has no wall size; a typical wall stands in


def _sized(url: str | None, w: Any, h: Any, d: Any, name: str | None = None) -> dict[str, Any]:
    if not url or not (w and h):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "That item has no photo or size to match with.")
    return {"url": url, "size": (float(w) * 100, float(h) * 100, float(d) * 100 if d else None), "name": name}


async def other_item(request: Request, kind: str, item_id: str, user: str | None = None) -> dict[str, Any]:
    """A catalogue item (cart, purchases, or picked from the catalogue), or one of the shopper's own customised or
    uploaded models: its photo (or colour) and real size in cm, so the shopper doesn't upload or measure anything."""
    if kind in ("design", "user_model"):
        settings = get_settings()
        rest = f"{settings.supabase_url.rstrip('/')}/rest/v1"
        if kind == "design":
            current = "current:design_versions!designs_current_version_fk(data,thumbnail_url)"
            params = {"id": f"eq.{item_id}", "owner_id": f"eq.{user}", "deleted_at": "is.null", "select": f"title,thumbnail_url,{current}"}
            table = "designs"
        else:
            params = {"id": f"eq.{item_id}", "owner_id": f"eq.{user}", "select": "*"}
            table = "user_models"
        response = await request.app.state.auth_http.get(f"{rest}/{table}", headers=service_headers(settings), params=params)
        rows = response.json() if response.status_code < 400 else []
        if not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "That model is not one of yours.")
        row = rows[0]
        if kind == "user_model":
            return _sized(row.get("thumbnail_url") or row.get("source_image_url"), row.get("width_m"), row.get("height_m"), row.get("depth_m"), row.get("title"))
        version = row.get("current") or {}
        item = (version.get("data") or {}).get("item") or {}
        dims, source = item.get("dimensions") or {}, item.get("source") or {}
        return _sized(
            version.get("thumbnail_url") or row.get("thumbnail_url"),
            dims.get("width") or source.get("width_m"), dims.get("height") or source.get("height_m"), dims.get("depth") or source.get("depth_m"), row.get("title") or source.get("title"),
        )
    if kind == "model":
        catalog = await store(request).catalog(model_id=item_id, limit=1, require_3d=False)
        found = [m for c in catalog.get("companies", []) for m in c.get("models", [])]
        if not found or not found[0].get("thumbnail") or not (found[0].get("width_m") and found[0].get("height_m")):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "That item has no photo or size to match with.")
        m = found[0]
        return {"url": m["thumbnail"], "size": (m["width_m"] * 100, m["height_m"] * 100, (m.get("depth_m") or 0) * 100 or None)}
    settings = get_settings()
    table = "carpets" if kind == "carpet" else "paints"
    response = await request.app.state.auth_http.get(
        f"{settings.supabase_url.rstrip('/')}/rest/v1/{table}", headers=service_headers(settings), params={"select": "*", "id": f"eq.{item_id}"}
    )
    rows = response.json() if response.status_code < 400 else []
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That item is no longer in the catalogue.")
    row = rows[0]
    if kind == "carpet":
        return {"url": row["image_url"], "size": (float(row["width_m"]) * 100, float(row["length_m"]) * 100, None)}
    tile = row.get("tile_width_m") or row.get("tile_m")
    if row.get("kind") == "texture" and row.get("texture_url"):
        return {"url": row["texture_url"], "size": (*DEFAULT_WALL_CM, None), "tile": float(tile) * 100 if tile else None}
    return {"color": row["color"], "size": (*DEFAULT_WALL_CM, None)}


async def read_image(upload: UploadFile) -> bytes:
    if (upload.content_type or "").split(";")[0].lower() not in ALLOWED:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a JPG, PNG or WebP image.")
    data = await upload.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Images must be 15 MB or smaller.")
    return data


def out_dir() -> Path:
    path = get_settings().storage_dir / "match"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup(directory: Path) -> None:
    cutoff = time.time() - KEEP_SECONDS
    for f in directory.glob("*.jpg"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)


@router.post("")
async def create_match(
    request: Request,
    modelId: Annotated[str, Form()],
    matchType: Annotated[Literal["furniture", "wall_paint", "carpet"], Form()],
    width: Annotated[float | None, Form(gt=0, le=5000)] = None,
    height: Annotated[float | None, Form(gt=0, le=5000)] = None,
    unit: Annotated[Literal["cm", "m"], Form()] = "cm",
    variationsCount: Annotated[int, Form(ge=1, le=5)] = 3,
    depth: Annotated[float | None, Form(gt=0, le=5000)] = None,
    position: Annotated[Literal["front", "left", "right", "behind"] | None, Form()] = None,
    color: Annotated[str | None, Form()] = None,
    tileSize: Annotated[float | None, Form(gt=0, le=1000)] = None,
    modelWidth: Annotated[float | None, Form(gt=0, le=5000)] = None,
    modelDepth: Annotated[float | None, Form(gt=0, le=5000)] = None,
    modelHeight: Annotated[float | None, Form(gt=0, le=5000)] = None,
    uploadedImage: Annotated[UploadFile | None, File()] = None,
    # Match with an item from the cart / purchases instead of an upload: its photo and size come from the catalogue.
    otherKind: Annotated[Literal["model", "design", "user_model", "carpet", "wall"] | None, Form()] = None,
    otherId: Annotated[UUID | None, Form()] = None,
    # The item shown with the other one: a catalogue model, or (from the 3D editor) the shopper's design or own model,
    # or "snapshot": the editor's picture of the model as it is now (saved or not), sized by modelWidth/Depth/Height.
    modelKind: Annotated[Literal["model", "design", "user_model", "snapshot"], Form()] = "model",
    modelImage: Annotated[UploadFile | None, File()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Model sizes (modelWidth/Depth/Height, same unit) are only used when the catalogue has none.
    Answers with NDJSON progress lines, then the result (see `stream`)."""
    settings = get_settings()
    if not settings.astra_configured:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Image generation is not configured (OPENAI_API_KEY).")
    user: list[str] = []

    async def owner() -> str:  # the shopper's own models need who is asking; catalogue items don't
        if not user:
            user.append(await user_id_from_bearer(authorization, settings, request.app.state.auth_http))
        return user[0]

    data: bytes | None = None
    other_cm: tuple[float, float, float | None] | None = None
    if otherKind and otherId:
        other = await other_item(request, otherKind, str(otherId), await owner() if otherKind in ("design", "user_model") else None)
        matchType = OTHER_TYPE[otherKind]  # type: ignore[assignment]
        other_cm = other["size"]
        color = other.get("color")
        if other.get("url"):
            fetched = await request.app.state.auth_http.get(other["url"])
            if fetched.status_code >= 400:
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not download that item's photo.")
            data = fetched.content
    elif width is None or height is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Enter the item's width and height.")
    elif uploadedImage is not None:
        data = await read_image(uploadedImage)
    if data is None and not (matchType == "wall_paint" and color):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Upload a texture or choose a paint colour." if matchType == "wall_paint" else "Upload an image to match.")
    if matchType == "furniture" and not position:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Choose where the furniture goes.")
    if matchType == "wall_paint" and not position:
        position = "behind"
    if matchType == "carpet":
        position = None

    snapshot: bytes | None = None
    if modelKind == "snapshot":
        if modelImage is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "This model has no photo to match against.")
        snapshot = await read_image(modelImage)
        models = [{"title": "product", "thumbnail": None, "width_m": None, "height_m": None, "depth_m": None}]
    elif modelKind == "model":
        catalog = await store(request).catalog(model_id=modelId, limit=1, require_3d=False)  # photo-only models match too
        models = [model for company in catalog.get("companies", []) for model in company.get("models", [])]
    else:  # its saved photo and (edited) size, in the catalogue's shape
        mine = await other_item(request, modelKind, modelId, await owner())
        w, h, d = mine["size"]
        models = [{"title": mine["name"], "thumbnail": mine["url"], "width_m": w / 100, "height_m": h / 100, "depth_m": d / 100 if d else None}]
    if not models:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This model is not in the catalogue.")
    model = models[0]
    if snapshot is None:
        photo_url = model.get("thumbnail")
        if not photo_url:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "This model has no photo to match against.")
        photo = await request.app.state.auth_http.get(photo_url)
        if photo.status_code >= 400:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not download the model's photo.")
        snapshot = photo.content

    def model_cm(value: Any, override: float | None, name: str) -> float:
        if value:
            return float(value) * 100  # catalogue sizes are metres
        if override:
            return im.to_cm(override, unit)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"This model has no {name} in the catalogue. Enter it to continue.")

    try:
        existing = im.Item(
            model.get("title") or "product",
            model_cm(model.get("width_m"), modelWidth, "width"),
            model_cm(model.get("height_m"), modelHeight, "height"),
            # Depth is optional: the room-size estimate uses the width when it is unknown.
            float(model["depth_m"]) * 100 if model.get("depth_m") else (im.to_cm(modelDepth, unit) if modelDepth else None),
        )
        if other_cm:
            item = im.Item(matchType, *other_cm)
        else:
            item = im.Item(matchType, im.to_cm(width, unit), im.to_cm(height, unit), im.to_cm(depth, unit) if depth else None)  # type: ignore[arg-type]
        references = [im.reference_png(snapshot), im.reference_png(data) if data else im.swatch_png(color or "")]
    except im.ImageMatchError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc

    warnings: list[str] = []
    if matchType == "wall_paint" and color and variationsCount > 1:
        warnings.append("A solid paint colour is shown from different camera angles; the colour itself is the same in every image.")
    tile = other.get("tile") if other_cm else (im.to_cm(tileSize, unit) if tileSize else None)
    prompts = [im.build_prompt(matchType, position, existing, item, i, color, tile) for i in range(variationsCount)]
    gpt = await generator_setting.current(settings, request.app.state.auth_http, generator_setting.MATCH)
    matcher = im.ImageMatcher(settings, request.app.state.openai_http, settings.openai_astra_model if gpt == "astra" else settings.openai_sol_model)
    label = PLACEMENT_LABEL.get((matchType, position), "")

    async def stream():
        """NDJSON: {"progress": n} lines while generating, then the result (or {"error": …})."""
        last = 0
        try:
            async for kind, value in matcher.variations_with_progress(prompts, references):
                if kind == "progress":
                    last = max(last, int(value))  # type: ignore[arg-type]
                    yield json.dumps({"progress": last}) + "\n"
                    continue
                images, errors = value  # type: ignore[misc]
                if errors:
                    warnings.append(f"{len(errors)} of {variationsCount} images could not be generated: {errors[0]}")
                directory = out_dir()
                cleanup(directory)
                result = []
                for n, (i, png) in enumerate(images, 1):
                    name = f"{uuid4().hex}.jpg"
                    Image.open(io.BytesIO(png)).convert("RGB").save(directory / name, "JPEG", quality=92)
                    result.append({"id": f"variation-{n}", "imageUrl": f"/api/match/files/{name}", "description": f"{label} — {VIEW_LABEL[i % len(VIEW_LABEL)]}"})
                yield json.dumps({"progress": 100, "success": True, "variations": result, "warnings": warnings}) + "\n"
        except im.ImageMatchError as exc:
            yield json.dumps({"error": str(exc), "status": exc.status_code}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.get("/files/{name}")
async def match_file(name: str) -> FileResponse:
    path = out_dir() / name
    if "/" in name or ".." in name or not name.endswith(".jpg") or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This result has expired. Generate it again.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
