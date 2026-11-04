"""Admin portal: the default woods (supabase/woods.sql), owned by the Sha2ty vendor. Admins are not Supabase users,
so these routes write with the service key, always scoped to the platform vendor's wood rows."""

from __future__ import annotations

import uuid
from typing import Annotated, Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.api.admin import require_admin
from app.core.config import get_settings
from app.services.auth import service_headers
from app.services.cdn import CdnError
from app.services.platform import ensure_platform_vendor

router = APIRouter(prefix="/api/admin/woods", tags=["admin"], dependencies=[Depends(require_admin)])

# frontend/lib/designs/materials.ts WOOD_TYPES
WOOD_TYPES = {"oak", "beech", "pine", "walnut", "teak", "ash", "mahogany", "mdf", "hdf", "plywood", "particleboard", "blockboard", "melamine", "veneer", "laminate", "lacquer"}
UNITS = {"m2", "m3", "sheet"}
IMAGE_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
MAX_IMAGE = 15 * 1024 * 1024
COLUMNS = "id,name,wood_type,description,texture_url,normal_url,tile_m,price,price_unit,thicknesses_mm,sheet_w_m,sheet_h_m,roughness,created_at"


def _rest() -> str:
    return f"{get_settings().supabase_url.rstrip('/')}/rest/v1"


async def _platform(request: Request) -> dict[str, Any]:
    return await ensure_platform_vendor(get_settings(), request.app.state.auth_http, request.app.state.cdn)


async def _image(file: UploadFile | None, vendor_id: str, folder: str, request: Request) -> str | None:
    if file is None or not file.filename:
        return None
    ext = IMAGE_TYPES.get(file.content_type or "")
    if not ext:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Use a JPG, PNG or WebP image.")
    data = await file.read(MAX_IMAGE + 1)
    if len(data) > MAX_IMAGE:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "The image is larger than 15 MB.")
    try:
        return request.app.state.cdn.upload(f"{vendor_id}/woods/{folder}/{uuid.uuid4()}.{ext}", data, file.content_type)
    except CdnError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not store the image.") from exc


def _fields(name: str, wood_type: str, description: str, price: str, price_unit: str, thicknesses: str, sheet_w: str, sheet_h: str, roughness: str) -> dict[str, Any]:
    if wood_type not in WOOD_TYPES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Unknown wood type.")
    if price_unit not in UNITS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Price per m², m³ or sheet.")

    def number(raw: str, label: str, low: float, high: float) -> float | None:
        if not raw.strip():
            return None
        try:
            value = float(raw)
        except ValueError:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{label} must be a number.") from None
        if not low <= value <= high:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{label} is out of range.")
        return value

    mm = [number(x, "Thickness", 0.1, 500) for x in thicknesses.replace("،", ",").split(",") if x.strip()]
    sheet = (number(sheet_w, "Sheet width", 0.1, 10), number(sheet_h, "Sheet length", 0.1, 10)) if price_unit == "sheet" else (None, None)
    if price_unit == "sheet" and None in sheet:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Enter the sheet width and length in metres.")
    return {
        "name": name.strip()[:120] or wood_type,
        "wood_type": wood_type,
        "description": description.strip()[:500] or None,
        "price": number(price, "Price", 0, 1e9),
        "price_unit": price_unit,
        "thicknesses_mm": mm[:20] or None,
        "sheet_w_m": sheet[0],
        "sheet_h_m": sheet[1],
        "roughness": number(roughness, "Surface", 0, 1),
    }


Text = Annotated[str, Form()]


@router.get("")
async def list_woods(request: Request) -> dict[str, Any]:
    """The default woods, and how many vendors work with each."""
    vendor = await _platform(request)
    client: httpx.AsyncClient = request.app.state.auth_http
    headers = service_headers(get_settings())
    rows = await client.get(f"{_rest()}/fabrics?vendor_id=eq.{vendor['id']}&kind=eq.wood&select={COLUMNS}&order=created_at", headers=headers)
    if rows.status_code >= 400:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Run supabase/fabrics.sql and supabase/woods.sql first.")
    picks = await client.get(f"{_rest()}/vendor_woods?select=wood_id", headers=headers)
    counts: dict[str, int] = {}
    for p in picks.json() if picks.status_code < 400 else []:
        counts[p["wood_id"]] = counts.get(p["wood_id"], 0) + 1
    return {"woods": [{**w, "vendors": counts.get(w["id"], 0)} for w in rows.json()]}


@router.post("")
async def create_wood(
    request: Request, name: Text, wood_type: Text, price_unit: Text, image: Annotated[UploadFile, File()],
    description: Text = "", price: Text = "", thicknesses: Text = "", sheet_w: Text = "", sheet_h: Text = "", roughness: Text = "",
    tile_cm: Text = "60", normal: Annotated[UploadFile | None, File()] = None,
) -> dict[str, Any]:
    vendor = await _platform(request)
    row = _fields(name, wood_type, description, price, price_unit, thicknesses, sheet_w, sheet_h, roughness)
    texture = await _image(image, vendor["id"], "grain", request)
    if not texture:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Add a grain photo.")
    row.update(vendor_id=vendor["id"], kind="wood", texture_url=texture, normal_url=await _image(normal, vendor["id"], "normal", request), tile_m=_tile(tile_cm))
    response = await request.app.state.auth_http.post(f"{_rest()}/fabrics?select={COLUMNS}", headers={**service_headers(get_settings()), "Prefer": "return=representation"}, json=row)
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not save the wood.")
    return {"wood": {**response.json()[0], "vendors": 0}}


@router.patch("/{wood_id}")
async def update_wood(
    wood_id: UUID, request: Request, name: Text, wood_type: Text, price_unit: Text,
    description: Text = "", price: Text = "", thicknesses: Text = "", sheet_w: Text = "", sheet_h: Text = "", roughness: Text = "",
    tile_cm: Text = "", image: Annotated[UploadFile | None, File()] = None, normal: Annotated[UploadFile | None, File()] = None,
    remove_normal: Text = "",
) -> dict[str, Any]:
    vendor = await _platform(request)
    row = _fields(name, wood_type, description, price, price_unit, thicknesses, sheet_w, sheet_h, roughness)
    if texture := await _image(image, vendor["id"], "grain", request):
        row["texture_url"] = texture
    if normal_url := await _image(normal, vendor["id"], "normal", request):
        row["normal_url"] = normal_url
    elif remove_normal == "1":
        row["normal_url"] = None
    if tile_cm.strip():
        row["tile_m"] = _tile(tile_cm)
    response = await request.app.state.auth_http.patch(
        f"{_rest()}/fabrics?id=eq.{wood_id}&vendor_id=eq.{vendor['id']}&kind=eq.wood&select={COLUMNS}",
        headers={**service_headers(get_settings()), "Prefer": "return=representation"},
        json=row,
    )
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not save the wood.")
    if not response.json():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Wood not found.")
    return {"wood": response.json()[0]}


@router.delete("/{wood_id}")
async def delete_wood(wood_id: UUID, request: Request) -> dict[str, Any]:
    """Designs that used it keep their saved look and estimate; vendors' ticks for it go with it."""
    vendor = await _platform(request)
    response = await request.app.state.auth_http.delete(
        f"{_rest()}/fabrics?id=eq.{wood_id}&vendor_id=eq.{vendor['id']}&kind=eq.wood", headers={**service_headers(get_settings()), "Prefer": "return=representation"}
    )
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not delete the wood.")
    if not response.json():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Wood not found.")
    return {"deleted": str(wood_id)}


def _tile(raw: str) -> float:
    try:
        cm = float(raw)
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Grain repeat must be a number of cm.") from None
    if not 1 <= cm <= 500:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Grain repeat is between 1 and 500 cm.")
    return cm / 100
