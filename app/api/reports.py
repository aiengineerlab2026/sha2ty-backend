"""The one-page PDF report of an implementation request (services/request_report.py), for whoever may see the
request: its customer, the vendors it is open to (or who made an offer), and admins."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.api.admin import token_valid
from app.core.config import get_settings
import logging

from app.services import part_sizes, request_report
from app.services.auth import service_headers, user_id_from_bearer

router = APIRouter(prefix="/api/tasks", tags=["requests"])
logger = logging.getLogger(__name__)
MAX_PICTURE = 8 * 1024 * 1024


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
    """The model's thumbnail, only from the platform's own storage (never an arbitrary URL)."""
    prefix = f"{get_settings().supabase_url.rstrip('/')}/storage/v1/object/public/"
    if not url or not url.startswith(prefix) or ".." in url:
        return None
    try:
        response = await client.get(url, timeout=15)
    except httpx.HTTPError:
        return None
    return response.content if response.status_code < 400 and len(response.content) <= MAX_PICTURE else None


MAX_GLB = 40 * 1024 * 1024


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


async def _sizes(client: httpx.AsyncClient, design: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Each part's size from the request's 3D file (platform storage only); None when it cannot be read: the report
    still goes out, without sizes."""
    url = design.get("glb_url")
    if not url or not url.startswith(f"{get_settings().supabase_url.rstrip('/')}/storage/v1/object/public/") or ".." in url:
        return None
    try:
        response = await client.get(url, timeout=30)
        if response.status_code >= 400 or len(response.content) > MAX_GLB:
            return None
        return await run_in_threadpool(part_sizes.measure, response.content, design.get("parts") or [], design.get("dimensions"), design.get("materials"))
    except Exception:  # noqa: BLE001 - a file that cannot be measured leaves the sizes out, never the report
        logger.warning("Could not measure the parts of %s", url[:120], exc_info=True)
        return None


@router.get("/{task_id}/report")
async def report(
    task_id: UUID,
    request: Request,
    lang: Annotated[Literal["ar", "en"], Query()] = "ar",
    authorization: Annotated[str | None, Header()] = None,
    x_admin_token: Annotated[str | None, Header()] = None,
) -> Response:
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
    sizes = await _sizes(client, design)
    customer = await _customer(client, task.get("customer_id"))
    pdf = await run_in_threadpool(request_report.build, task, picture, lang, None, download, sizes, customer)
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="sha2ty-request-{str(task_id)[:8]}.pdf"',
        "Cache-Control": "no-store",
    })
