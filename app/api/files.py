"""The platform's files on Cloudflare R2: served at /api/files/<key> (the bucket is private) and uploaded by a signed-in
user into their own folder (<user id>/…, as Supabase Storage's cads_insert policy allowed)."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from starlette.background import BackgroundTask

from app.api.designs import User
from app.core.config import get_settings
from app.services.cdn import CdnError

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("/{key:path}")
async def read(key: str, request: Request):
    if ".." in key:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")
    try:
        obj = await run_in_threadpool(request.app.state.cdn.get, key)
    except CdnError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    body = obj["Body"]
    headers = {"Content-Length": str(obj["ContentLength"]), "Cache-Control": "public, max-age=3600"}
    return StreamingResponse(body.iter_chunks(1 << 20), media_type=obj.get("ContentType"), headers=headers, background=BackgroundTask(body.close))


@router.post("")
async def upload(user_id: User, request: Request, file: Annotated[UploadFile, File()], folder: Annotated[str, Form()] = "uploads") -> dict[str, str]:
    """Stores the file at <user id>/<folder>/<random>.<ext> and returns its public URL."""
    folder = folder.strip("/")
    if ".." in folder or not folder:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid folder.")
    data = await file.read()
    if len(data) > get_settings().max_model_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "The file is too large.")
    ext = (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else "bin"
    try:
        url = await run_in_threadpool(request.app.state.cdn.upload, f"{user_id}/{folder}/{uuid.uuid4()}.{ext}", data, file.content_type)
    except CdnError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    return {"url": url}
