"""POST /api/realistic-materials: a copy of a model's GLB with realistic materials (services/realistic_materials.py),
for the mobile AR viewer's optional mode. The original file is never changed; the app falls back to it on any
status but "ready"."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.designs import User
from app.core.config import get_settings
from app.services import realistic_materials, vision
from app.services.cdn import CdnError, file_prefixes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["realistic-materials"])

MAX_GLB = 60 * 1024 * 1024
MAX_PHOTO = 8 * 1024 * 1024


class RealisticBody(BaseModel):
    glb_url: str = Field(max_length=1000)
    # The product's photo (its thumbnail), when there is one: helps tell wood from painted MDF, linen from leather.
    photo_url: str | None = Field(default=None, max_length=1000)


FILES = "/api/files/"


def _stored_key(url: str) -> str | None:
    """The storage key of a file served by this API's /api/files route, whichever host the URL names (a file saved
    through a local dev server reads http://localhost:8010/api/files/…, the same bucket as production)."""
    path = urlsplit(url).path
    return unquote(path.split(FILES, 1)[1]) if FILES in path and ".." not in path else None


def _ours(url: str | None) -> bool:
    return bool(url) and ".." not in url and (url.startswith(file_prefixes(get_settings())) or _stored_key(url) is not None)  # type: ignore[union-attr, arg-type]


async def _fetch(request: Request, url: str, limit: int) -> bytes | None:
    """Our own stored files are read from storage by key (never by fetching an arbitrary host); other platform URLs
    (Supabase Storage, the CDN's public URL) over HTTP."""
    key = _stored_key(url)
    if key is not None:
        def read() -> bytes | None:
            try:
                obj = request.app.state.cdn.get(key)
            except CdnError:
                return None
            body = obj["Body"].read(limit + 1)
            return body if len(body) <= limit else None

        return await asyncio.to_thread(read)
    try:
        response = await request.app.state.auth_http.get(url, timeout=60)
    except httpx.HTTPError:
        return None
    return response.content if response.status_code < 400 and len(response.content) <= limit else None


@router.post("/realistic-materials")
async def realistic_materials_for(body: RealisticBody, request: Request, _user: User) -> dict[str, Any]:
    settings = get_settings()
    if not _ours(body.glb_url):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Only models stored on Sha2ty can be given realistic materials.")
    db: httpx.AsyncClient = request.app.state.auth_http
    data = await _fetch(request, body.glb_url, MAX_GLB)
    if data is None:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not read the 3D model.")
    photo = await _fetch(request, body.photo_url, MAX_PHOTO) if _ours(body.photo_url) else None  # type: ignore[arg-type]
    model = await vision.current_model(settings, db)  # the admin's choice, never hard-coded here
    if not vision.configured(settings, model):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Realistic materials are not available right now.")

    async def ask(content: list[dict[str, Any]], schema: dict[str, Any]) -> str | None:
        return await vision.ask_json(content, schema, "material_profile", settings, vision.client_for(request.app.state, model), model)

    def upload(key: str, glb: bytes) -> str:
        try:
            return request.app.state.cdn.upload(key, glb, "model/gltf-binary")
        except CdnError as exc:
            raise HTTPException(exc.status_code, exc.message) from exc

    async def cache_get(key: str) -> dict[str, Any] | None:
        return await vision.cache_get(settings, db, key, table="model_cache")

    async def cache_put(key: str, result: dict[str, Any]) -> None:
        await vision.cache_put(settings, db, key, "realistic_materials", model, result, table="model_cache")

    result = await realistic_materials.make(data, photo, ask, cache_get, cache_put, upload)
    logger.info("Realistic materials %s: %s", body.glb_url.rsplit("/", 1)[-1], result.get("status"))
    return result

