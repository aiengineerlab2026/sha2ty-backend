"""POST /api/match: a photorealistic room with the catalogue model and the shopper's item, made by
OpenAI image editing from both photos (see app/services/image_match.py)."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from app.api.plans import store
from app.core.config import get_settings
from app.services import image_match as im

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
    width: Annotated[float, Form(gt=0, le=5000)],
    height: Annotated[float, Form(gt=0, le=5000)],
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
) -> StreamingResponse:
    """Model sizes (modelWidth/Depth/Height, same unit) are only used when the catalogue has none.
    Answers with NDJSON progress lines, then the result (see `stream`)."""
    settings = get_settings()
    if not settings.astra_configured:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Image generation is not configured (OPENAI_API_KEY).")
    data: bytes | None = None
    if uploadedImage is not None:
        if (uploadedImage.content_type or "").split(";")[0].lower() not in ALLOWED:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a JPG, PNG or WebP image.")
        data = await uploadedImage.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Images must be 15 MB or smaller.")
    if data is None and not (matchType == "wall_paint" and color):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Upload a texture or choose a paint colour." if matchType == "wall_paint" else "Upload an image to match.")
    if matchType == "furniture" and not position:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Choose where the furniture goes.")
    if matchType == "wall_paint" and not position:
        position = "behind"
    if matchType == "carpet":
        position = None

    catalog = await store(request).catalog(model_id=modelId, limit=1)
    models = [model for company in catalog.get("companies", []) for model in company.get("models", [])]
    if not models:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This model is not in the catalogue.")
    model = models[0]
    photo_url = model.get("thumbnail")
    if not photo_url:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "This model has no photo to match against.")
    photo = await request.app.state.auth_http.get(photo_url)
    if photo.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not download the model's photo.")

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
            model_cm(model.get("depth_m"), modelDepth, "depth"),
        )
        item = im.Item(matchType, im.to_cm(width, unit), im.to_cm(height, unit), im.to_cm(depth, unit) if depth else None)
        references = [im.reference_png(photo.content), im.reference_png(data) if data else im.swatch_png(color or "")]
    except im.ImageMatchError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc

    warnings: list[str] = []
    if matchType == "wall_paint" and color and variationsCount > 1:
        warnings.append("A solid paint colour is shown from different camera angles; the colour itself is the same in every image.")
    tile = im.to_cm(tileSize, unit) if tileSize else None
    prompts = [im.build_prompt(matchType, position, existing, item, i, color, tile) for i in range(variationsCount)]
    matcher = im.ImageMatcher(settings, request.app.state.openai_http)
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
