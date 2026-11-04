"""Admin portal: the paint colour palette (supabase/model_paints.sql). Customers paint model parts in these colours
only; vendors' paints are offered in them. Admins are not Supabase users, so these routes write with the service key."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.admin import require_admin
from app.core.config import get_settings
from app.services.auth import service_headers

router = APIRouter(prefix="/api/admin/paint-colors", tags=["admin"], dependencies=[Depends(require_admin)])

COLUMNS = "id,name,hex,created_at"


class ColorBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    hex: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")


def _rest() -> str:
    return f"{get_settings().supabase_url.rstrip('/')}/rest/v1"


def _headers() -> dict[str, str]:
    return {**service_headers(get_settings()), "Prefer": "return=representation"}


@router.get("")
async def list_colors(request: Request) -> dict[str, Any]:
    client: httpx.AsyncClient = request.app.state.auth_http
    rows = await client.get(f"{_rest()}/paint_colors?select={COLUMNS}&order=created_at", headers=_headers())
    if rows.status_code >= 400:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Run supabase/model_paints.sql first.")
    return {"colors": rows.json()}


@router.post("")
async def create_color(body: ColorBody, request: Request) -> dict[str, Any]:
    row = {"name": body.name.strip(), "hex": body.hex.lower()}
    response = await request.app.state.auth_http.post(f"{_rest()}/paint_colors?select={COLUMNS}", headers=_headers(), json=row)
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not save the colour.")
    return {"color": response.json()[0]}


@router.patch("/{color_id}")
async def update_color(color_id: UUID, body: ColorBody, request: Request) -> dict[str, Any]:
    row = {"name": body.name.strip(), "hex": body.hex.lower()}
    response = await request.app.state.auth_http.patch(f"{_rest()}/paint_colors?id=eq.{color_id}&select={COLUMNS}", headers=_headers(), json=row)
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not save the colour.")
    if not response.json():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Colour not found.")
    return {"color": response.json()[0]}


@router.delete("/{color_id}")
async def delete_color(color_id: UUID, request: Request) -> dict[str, Any]:
    """Vendors' paints in this colour go with it; designs already painted keep their colour."""
    response = await request.app.state.auth_http.delete(f"{_rest()}/paint_colors?id=eq.{color_id}", headers=_headers())
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not delete the colour.")
    if not response.json():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Colour not found.")
    return {"deleted": str(color_id)}
