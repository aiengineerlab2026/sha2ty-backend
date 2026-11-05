"""Temporary: the database passthrough the web app's supabase-js still uses (moved here from the Next.js /api/sb route).

ponytail: an open passthrough, protected only by row-level security; each area of the app moves to its own endpoints
(catalogue, vendor, tasks, carts / orders, reviews, uploads) and this file is deleted with the last of them.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.core.config import get_settings

router = APIRouter(prefix="/api/sb", tags=["database (temporary)"])
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "host", "cookie"}


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"], include_in_schema=False)
async def passthrough(path: str, request: Request):
    if path.split("/", 1)[0] == "auth":
        return JSONResponse({"error": "Use /api/auth."}, status_code=404)
    settings = get_settings()
    headers = {k: v for k, v in request.headers.items() if k not in HOP and k != "x-admin-token"}
    headers["apikey"] = settings.supabase_anon_key.get_secret_value()
    url = f"{settings.supabase_url.rstrip('/')}/{path}"
    client: httpx.AsyncClient = request.app.state.auth_http
    upstream = await client.send(
        client.build_request(request.method, url, params=request.query_params, headers=headers, content=request.stream() if request.method not in ("GET", "HEAD") else None),
        stream=True,
    )
    out = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP}
    return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, headers=out, background=BackgroundTask(upstream.aclose))
