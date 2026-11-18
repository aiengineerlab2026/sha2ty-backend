"""The one-page PDF report of an implementation request (services/request_report.py), for whoever may see the
request: its customer, the vendors it is open to (or who made an offer), and admins."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from pydantic import BaseModel

from app.api.admin import token_valid
from app.api.marketplace import DesignViews, check_views
from app.core.config import get_settings

from app.services import report_photos, request_report
from app.services.auth import service_headers, user_id_from_bearer
from app.services.cdn import file_prefixes

router = APIRouter(prefix="/api/tasks", tags=["requests"])
MAX_PICTURE = 24 * 1024 * 1024  # the parts sheet of a model with many parts is the largest picture


async def _rows(client: httpx.AsyncClient, path: str) -> list[dict[str, Any]]:
    settings = get_settings()
    try:
        response = await client.get(f"{settings.supabase_url.rstrip('/')}/rest/v1/{path}", headers=service_headers(settings))
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not read the request.") from exc
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not read the request.")
    return response.json()


def may_see(task: dict[str, Any], user_id: str, vendor: dict[str, Any] | None, bids: list[dict[str, Any]]) -> bool:
    """The customer; an approved vendor it is open to (published, open or theirs alone) or who made an offer."""
    if task.get("customer_id") == user_id:
        return True
    if not vendor or vendor.get("status", "approved") != "approved":
        return False
    if any(b.get("vendor_id") == user_id for b in bids):
        return True
    return task.get("status") == "published" and (task.get("audience") != "vendor_exclusive" or task.get("exclusive_vendor_id") == user_id)


async def _picture(client: httpx.AsyncClient, url: str | None) -> bytes | None:
    """A picture of the model (its thumbnail, a view), only from the platform's own storage (never an arbitrary URL)."""
    if not url or not url.startswith(file_prefixes(get_settings())) or ".." in url:
        return None
    try:
        response = await client.get(url, timeout=15)
    except httpx.HTTPError:
        return None
    return response.content if response.status_code < 400 and len(response.content) <= MAX_PICTURE else None


async def _customer(client: httpx.AsyncClient, user_id: str | None) -> str | None:
    """The name of who made the request (their account's name; no email), None when it cannot be read."""
    settings = get_settings()
    if not user_id:
        return None
    try:
        response = await client.get(f"{settings.supabase_url.rstrip('/')}/auth/v1/admin/users/{user_id}", headers=service_headers(settings))
    except httpx.HTTPError:
        return None
    meta = (response.json().get("user_metadata") or {}) if response.status_code < 400 else {}
    return (meta.get("full_name") or meta.get("name") or "").strip() or None


class ReportBody(BaseModel):
    """Pictures made by whoever downloads, for this report only (the request's own are older): never stored here."""

    views: DesignViews


@router.post("/{task_id}/report")
async def report_with_views(
    task_id: UUID,
    body: ReportBody,
    request: Request,
    lang: Annotated[Literal["ar", "en"], Query()] = "ar",
    authorization: Annotated[str | None, Header()] = None,
    x_admin_token: Annotated[str | None, Header()] = None,
) -> Response:
    check_views(body.views)
    return await _report(task_id, request, lang, authorization, x_admin_token, body.views.model_dump())


@router.get("/{task_id}/report")
async def report(
    task_id: UUID,
    request: Request,
    lang: Annotated[Literal["ar", "en"], Query()] = "ar",
    authorization: Annotated[str | None, Header()] = None,
    x_admin_token: Annotated[str | None, Header()] = None,
) -> Response:
    return await _report(task_id, request, lang, authorization, x_admin_token)


async def _report(task_id: UUID, request: Request, lang: Literal["ar", "en"], authorization: str | None, x_admin_token: str | None, views_now: dict[str, Any] | None = None) -> Response:
    """The PDF for whoever may see the request (customer, its vendors, admins); `views_now`: newer pictures for it."""
    client: httpx.AsyncClient = request.app.state.auth_http
    admin = bool(x_admin_token and token_valid(x_admin_token, get_settings()))
    user_id = None if admin else await user_id_from_bearer(authorization, get_settings(), client)
    task = (await _rows(client, f"freelance_tasks?id=eq.{task_id}&select=*") or [None])[0]
    bids = await _rows(client, f"task_bids?task_id=eq.{task_id}&select=vendor_id,status") if task else []
    if not task:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found.")
    if not admin:
        vendor = (await _rows(client, f"vendors?id=eq.{user_id}&select=id,status") or [None])[0]
        if not may_see(task, user_id, vendor, bids):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found.")  # 404, not 403: it stays unseen
    picture = await _picture(client, ((task.get("spec") or {}).get("design") or {}).get("thumbnail_url"))
    # The 3D file: the customer and admins; a vendor only once the customer accepted their offer.
    download = admin or task.get("customer_id") == user_id or any(b.get("vendor_id") == user_id and b.get("status") == "accepted" for b in bids)
    design = (task.get("spec") or {}).get("design") or {}
    if views_now and design:
        design["views"] = views_now  # pictures made just now from the request's own model (newer than its stored ones)
    customer = await _customer(client, task.get("customer_id"))
    # The closed / opened pictures the request was sent with: a report never goes out with one of them missing.
    meta = design.get("views") or {}
    views = None
    if meta.get("version", 1) < request_report.VIEWS_VERSION:
        meta = {}  # numbered-marker pictures: the report says they must be made again, never shows them
        design["views"] = None
    if meta:
        urls = (meta.get("closed_url"), *([meta["ortho_url"]] if meta.get("ortho_url") else []), meta.get("open_url"), *(meta.get("detail_urls") or []))
        pictures = [await _picture(client, u) for u in urls]
        if not all(pictures):
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not read the model views for this report. Try again.")
        views = tuple(pictures)
    # Each main part's picture (requests sent before have none) and the wood textures the customer chose (platform
    # storage only; one that cannot be read leaves its swatch out, never the report).
    pictures = await _picture(client, meta.get("pictures_url")) if meta.get("pictures_url") else None
    if meta.get("pictures_url") and not pictures:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not read the part pictures for this report. Try again.")
    textures = {}
    # (the textures saved on the request's parts now, and those the pictures recorded)
    for url in {r.get("texture") for r in meta.get("parts") or [] if r.get("texture")} | {m.get("mapUrl") for m in design.get("materials") or [] if m.get("mapUrl")}:
        image = await _picture(client, url)
        if image:
            textures[url] = image
    # The thumbnail and the part pictures become real photos (gpt-image-2); the dimension views stay exact as drawn.
    picture, pictures = await report_photos.realistic(picture, pictures, meta, get_settings(), request.app.state.openai_http)
    pdf = await run_in_threadpool(request_report.build, task, picture, lang, None, download, customer, views, pictures, textures)
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="sha2ty-request-{str(task_id)[:8]}.pdf"',
        "Cache-Control": "no-store",
    })
