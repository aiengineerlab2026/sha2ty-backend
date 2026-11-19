"""Implementation requests from customised designs, and public vendor profiles (supabase/marketplace.sql).

A request is a freelance task built here from one immutable design version: the specification vendors bid on is
written by the server from that version (dimensions, parts, materials, files), never sent by the client, and
later saves of the design do not change it. Bids, chat and reviews are guarded by row-level security; accepting
a bid and status changes go through the database functions in tasks.sql.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.designs import Designs, User
from app.core.config import get_settings
from app.services.auth import service_headers
from app.services.cdn import file_prefixes

router = APIRouter(prefix="/api", tags=["marketplace"])

OPEN = ("published", "awarded", "in_progress")


class ViewPart(BaseModel):
    """One row of the report's parts table, measured from the saved model in the browser (cm)."""

    n: int = Field(ge=1, le=1000)
    id: str = Field(max_length=200)
    name: str = Field(max_length=200)
    added: bool = False
    quantity: int = Field(default=1, ge=1, le=1000)
    material: str | None = Field(default=None, max_length=120)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    width: float = Field(ge=0, le=10_000)
    height: float = Field(ge=0, le=10_000)
    depth: float = Field(ge=0, le=10_000)
    thickness: float = Field(ge=0, le=10_000)
    # Distances from the compartment's inside faces (added parts): top, bottom, left, right, front, back.
    offsets: dict[str, float] | None = None
    shown: bool = True  # labelled in the opened picture
    # Its name in Arabic, both without position or number; a main furniture part (not hardware); its material's type
    # and finish when the customer set them.
    name_ar: str | None = Field(default=None, max_length=200)
    main_name: str | None = Field(default=None, max_length=200)
    main_name_ar: str | None = Field(default=None, max_length=200)
    main: bool | None = None
    kind: str | None = Field(default=None, max_length=40)
    finish: str | None = Field(default=None, pattern=r"^(matte|satin|gloss)$")
    # The wood chosen in the editor: board type and thickness (mm), the surface on it, a paint's colour, the texture.
    board: str | None = Field(default=None, max_length=40)
    board_mm: float | None = Field(default=None, ge=0, le=200)
    surface: str | None = Field(default=None, max_length=120)
    surface_type: str | None = Field(default=None, max_length=40)
    paint: str | None = Field(default=None, max_length=120)
    texture: str | None = Field(default=None, max_length=1000)
    group: str | None = Field(default=None, pattern=r"^(structure|doors|drawers|shelves|accessories)$")
    pic: int | None = Field(default=None, ge=0, le=1000)  # its picture: cell `pic` of the parts sheet


class Check(BaseModel):
    """One measurement check made before the report (frontend lib/designs/measure.ts checkMeasurements)."""

    level: Literal["critical", "warning", "info", "error"]  # error: pictures made before the three levels (critical)
    code: Literal["approx", "mismatch", "units", "missing", "exceeds", "over_request", "axis", "overlap", "joint", "clearance", "drawer", "shelf"]
    parts: list[str] = Field(default_factory=list, max_length=10)
    params: dict[str, float | str] = Field(default_factory=dict, max_length=10)


class DesignViews(BaseModel):
    """The two rendered pictures of the saved model (platform storage) and the measurements drawn on them."""

    closed_url: str = Field(max_length=1000)
    open_url: str = Field(max_length=1000)
    # Close-ups of the parts whose dimensions do not fit clearly in the opened picture.
    # One a row of the parts table (frontend MAX_DETAILS).
    detail_urls: list[str] = Field(default_factory=list, max_length=60)
    # What each close-up shows: a group of parts (doors, drawers, shelves, structure, accessories).
    detail_groups: list[str] = Field(default_factory=list, max_length=60)
    # Front, side and top orthographic views with the overall size (one picture).
    ortho_url: str | None = Field(default=None, max_length=1000)
    # What the measurements do not agree on (requested vs model size, parts that overlap…), for the checks page.
    checks: list[Check] | None = Field(default=None, max_length=300)
    # 2: dimensions written beside the parts. 1 (or none): numbered markers, which the report no longer shows.
    version: int = Field(default=1, ge=1, le=100)
    # Each main part's picture: one sheet of `pic_size` squares, `pic_cols` to a line.
    pictures_url: str | None = Field(default=None, max_length=1000)
    pic_size: int = Field(default=512, ge=64, le=2048)
    pic_cols: int = Field(default=8, ge=1, le=64)
    scale_known: bool = True
    # The model's own size as measured from its geometry (cm), beside the requested one.
    overall: dict[str, float] | None = None
    opened: int = Field(default=0, ge=0, le=1000)  # movable parts opened in the second picture
    parts: list[ViewPart] = Field(default_factory=list, max_length=1000)


class RequestBody(BaseModel):
    description: str = Field(default="", max_length=4000)
    location: str = Field(default="", max_length=200)
    budget_max: float | None = Field(default=None, ge=0)
    deadline: date | None = None
    # Which parts the customer wants made (component ids of the saved version); None: the whole model.
    part_ids: list[str] | None = Field(default=None, max_length=500)
    # How many pieces of the model the customer wants made.
    quantity: int = Field(default=1, ge=1, le=1000)
    # The closed / opened pictures with dimensions for the report (captured by the editor from the saved version).
    views: DesignViews | None = None


def spec_from_version(design: dict[str, Any], version: dict[str, Any], part_ids: list[str] | None = None) -> dict[str, Any]:
    """What vendors see and bid on: the model as this version has it, and which of its parts to make."""
    item = (version.get("data") or {}).get("item") or {}
    source = item.get("source") or {}
    removed = set(item.get("removed_parts") or [])
    # Removed parts are not part of the model any more: vendors never see them.
    components = [c for c in ((item.get("components") or {}).get("components") or []) if c.get("id") not in removed]
    names = {c.get("id"): c.get("name") for c in components}
    # The vendor's variant the customer picked for a part (its own shape and price difference), when any.
    chosen = item.get("part_variants") or {}
    variants = {
        c["id"]: v
        for c in components
        for v in (c.get("variants") or [])
        if isinstance(v, dict) and chosen.get(c.get("id")) == v.get("id")
    }
    if part_ids is not None:
        unknown = [p for p in part_ids if p not in names]
        if unknown or not part_ids:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Choose at least one part of this model." if not part_ids else f"Unknown parts: {', '.join(unknown[:5])}")
    wanted = set(part_ids) if part_ids is not None else set(names)
    materials = [
        {"part": names.get(part, part), "part_id": part, **{k: v for k, v in edit.items() if k in ("color", "mapUrl", "tile", "mapping", "roughness", "metalness", "material")}}
        for part, edit in (item.get("material_edits") or {}).items()
        if part not in removed
    ]
    # The customer's material estimate for the parts asked for (fabric metres, wood boards/sheets, finishes); null
    # when any of them is priced on request.
    picked = [m["material"] for m in materials if m.get("material") and m["part_id"] in wanted]
    totals = [p.get("total") for p in picked]
    materials_total = round(sum(totals), 2) if picked and all(isinstance(x, (int, float)) for x in totals) else None
    dims = item.get("dimensions") or ({"width": source.get("width_m"), "height": source.get("height_m"), "depth": source.get("depth_m")} if source.get("width_m") else None)
    return {
        "design": {
            "design_id": design["id"],
            "version_id": version["id"],
            "version": version["number"],
            "title": design.get("title") or source.get("title") or "",
            "thumbnail_url": version.get("thumbnail_url") or design.get("thumbnail_url"),
            # The customised file when the version has one; otherwise the catalogue file with the edits listed.
            "glb_url": version.get("glb_url") or source.get("glb_url"),
            "catalogue_glb_url": source.get("glb_url"),
            "source": {"model_id": source.get("model_id"), "title": source.get("title"), "vendor_id": source.get("vendor_id"),
                       "width_m": source.get("width_m"), "height_m": source.get("height_m"), "depth_m": source.get("depth_m")},
            "dimensions": dims,
            "resize_mode": item.get("resize_mode"),
            "scale": item.get("scale"),
            "parts": [
                {"id": k, "name": v, "requested": k in wanted,
                 **({"variant": {key: variants[k].get(key) for key in ("id", "name", "glb_url", "price_delta")}} if k in variants else {})}
                for k, v in names.items()
            ],
            "whole_model": part_ids is None or wanted == set(names),
            # What the request's viewer needs to open its doors and drawers like the editor (read-only): each part's
            # id and names, the movable parts' settings and the parts the customer added.
            "part_document": {
                "components": [{k: c.get(k) for k in ("id", "name", "nameAr") if c.get(k) is not None} | {"editable": False} for c in components],
                **{k: (item.get("components") or {}).get(k) for k in ("animation", "added") if (item.get("components") or {}).get(k)},
            },
            "materials": materials,
            "materials_total": materials_total,
        }
    }


async def request_target(design: dict[str, Any], version: dict[str, Any], client: httpx.AsyncClient) -> dict[str, Any]:
    """Who a request for this design goes to, from the catalogue model it started from (the design's frozen
    source, never its current owner): the platform's own models are open to every approved vendor; a vendor's model
    goes to that vendor only. `available` is false when that vendor cannot take requests (suspended, removed)."""
    settings = get_settings()
    base = f"{settings.supabase_url.rstrip('/')}/rest/v1"
    source = ((version.get("data") or {}).get("item") or {}).get("source") or {}
    if source.get("source_type") == "user":  # the customer's own model: no vendor owns it, every vendor may bid
        return {"audience": "open", "vendor": None, "available": True}
    vendor_id = source.get("vendor_id")
    if not vendor_id:
        raise HTTPException(status.HTTP_409_CONFLICT, "The original vendor of this model is not known, so it cannot be requested.")
    try:
        response = await client.get(f"{base}/vendors?id=eq.{vendor_id}&select=*", headers=service_headers(settings))
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not check the original vendor.") from exc
    vendor = (response.json() or [None])[0] if response.status_code < 400 else None
    platform = source.get("source_type") == "platform" or bool(vendor and vendor.get("platform"))
    if platform:
        return {"audience": "open", "vendor": None, "available": True}
    return {
        "audience": "vendor_exclusive",
        "vendor": {"id": vendor_id, "name": (vendor or {}).get("name") or source.get("title") or ""},
        "available": bool(vendor) and vendor.get("status", "approved") == "approved",
    }


async def _design_version(design_id: UUID, user_id: str, store: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    design = await store.get_design(design_id, user_id)
    if design["kind"] != "item":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Rooms and apartments are offered from the plan builder.")
    if not design.get("current_version_id"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Save the design first.")
    return design, await store.version(design_id, version_id=design["current_version_id"])


@router.get("/designs/{design_id}/request-target")
async def get_request_target(design_id: UUID, request: Request, user_id: User, store: Designs) -> dict[str, Any]:
    design, version = await _design_version(design_id, user_id, store)
    return await request_target(design, version, request.app.state.auth_http)


@router.post("/designs/{design_id}/request", status_code=status.HTTP_201_CREATED)
async def request_implementation(design_id: UUID, body: RequestBody, request: Request, user_id: User, store: Designs) -> dict[str, Any]:
    """Ask vendors to make a saved customised model. One open request per design version (a repeat returns it)."""
    design, version = await _design_version(design_id, user_id, store)
    if body.views:
        check_views(body.views)
    client_: httpx.AsyncClient = request.app.state.auth_http
    target = await request_target(design, version, client_)
    if not target["available"]:
        # Never published to others instead: the model is the vendor's, only they may make it.
        raise HTTPException(status.HTTP_409_CONFLICT, f"{target['vendor']['name'] or 'The original vendor'} is not taking requests right now. Try again later.")
    settings = get_settings()
    base = f"{settings.supabase_url.rstrip('/')}/rest/v1"
    headers = {**service_headers(settings), "Prefer": "return=representation"}
    client: httpx.AsyncClient = request.app.state.auth_http
    try:
        existing = await client.get(
            f"{base}/freelance_tasks?customer_id=eq.{user_id}&design_version_id=eq.{version['id']}&status=in.({','.join(OPEN)})&select=*",
            headers=headers,
        )
        wanted_spec = spec_from_version(design, version, body.part_ids)  # validates the parts before anything is written
        for task in existing.json() if existing.status_code < 400 else []:
            spec = task.get("spec") or {}
            if spec.get("design", {}).get("parts") == wanted_spec["design"]["parts"] and spec.get("quantity", 1) == body.quantity:
                return task  # the same request again (same version, parts and quantity)
        spec = spec_from_version(design, version, body.part_ids)
        if body.views:
            spec["design"]["views"] = body.views.model_dump()
        row = {
            "customer_id": user_id, "title": design.get("title") or "Custom model", "scope": "item", "service": "furniture",
            "description": body.description.strip(), "location": body.location.strip(), "budget_max": body.budget_max,
            "deadline": body.deadline.isoformat() if body.deadline else None, "status": "published",
            "spec": {**spec, "audience": target["audience"], "vendor": target["vendor"], "quantity": body.quantity},
            "design_id": str(design_id), "design_version_id": version["id"],
            "audience": target["audience"], "exclusive_vendor_id": (target["vendor"] or {}).get("id"),
        }
        response = await client.post(f"{base}/freelance_tasks", headers=headers, json=row)
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not publish the request.") from exc
    if response.status_code >= 400:
        missing = any(name in response.text for name in ("design_id", "design_version_id", "deadline")) or ("freelance_tasks" in response.text and "does not exist" in response.text)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE if missing else status.HTTP_502_BAD_GATEWAY,
            "Requests are not set up yet: run backend/supabase/tasks.sql and marketplace.sql." if missing else "Could not publish the request.",
        )
    return response.json()[0]


def check_views(views: DesignViews) -> None:
    if not all(u.startswith(file_prefixes(get_settings())) and ".." not in u and len(u) <= 1000 for u in (views.closed_url, views.open_url, *views.detail_urls, *([u for u in (views.pictures_url, views.ortho_url) if u]))):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "The model views must be uploaded to Sha2ty first.")


@router.put("/designs/{design_id}/request-views")
async def update_request_views(design_id: UUID, views: DesignViews, request: Request, user_id: User, store: Designs) -> dict[str, int]:
    """New pictures for the design's active requests after a save moved them to its newest version (the old ones
    showed the model as it was). Only the owner's requests on the current version change."""
    design, version = await _design_version(design_id, user_id, store)
    check_views(views)
    settings = get_settings()
    base = f"{settings.supabase_url.rstrip('/')}/rest/v1"
    client: httpx.AsyncClient = request.app.state.auth_http
    try:
        found = await client.get(
            f"{base}/freelance_tasks?customer_id=eq.{user_id}&design_id=eq.{design_id}&design_version_id=eq.{version['id']}&status=in.({','.join(OPEN)})&select=id,spec",
            headers=service_headers(settings),
        )
        updated = 0
        for task in found.json() if found.status_code < 400 else []:
            spec = task.get("spec") or {}
            spec = {**spec, "design": {**(spec.get("design") or {}), "views": views.model_dump()}}
            done = await client.patch(f"{base}/freelance_tasks?id=eq.{task['id']}", headers=service_headers(settings), json={"spec": spec})
            updated += done.status_code < 400
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not update the request's model views.") from exc
    return {"updated": updated}


@router.put("/tasks/{task_id}/views")
async def replace_task_views(task_id: UUID, views: DesignViews, request: Request, user_id: User) -> dict[str, int]:
    """The customer's own request gets pictures made again from its frozen model (older pictures had numbered
    markers instead of dimensions). Only the customer; only while the request is active."""
    check_views(views)
    settings = get_settings()
    base = f"{settings.supabase_url.rstrip('/')}/rest/v1"
    client: httpx.AsyncClient = request.app.state.auth_http
    try:
        found = await client.get(f"{base}/freelance_tasks?id=eq.{task_id}&customer_id=eq.{user_id}&status=in.({','.join(OPEN)})&select=id,spec", headers=service_headers(settings))
        task = (found.json() or [None])[0] if found.status_code < 400 else None
        if not task or not (task.get("spec") or {}).get("design"):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found.")
        spec = task["spec"]
        spec = {**spec, "design": {**spec["design"], "views": views.model_dump()}}
        done = await client.patch(f"{base}/freelance_tasks?id=eq.{task_id}", headers=service_headers(settings), json={"spec": spec})
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not update the request's model views.") from exc
    if done.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not update the request's model views.")
    return {"updated": 1}


FURNITURE_KINDS = ("chairs", "sofas", "beds", "tables", "storage", "doors", "other")


class Size(BaseModel):
    """Centimetres, as the customer measured them."""

    width: float = Field(gt=0, le=2000)
    height: float = Field(gt=0, le=2000)
    depth: float = Field(gt=0, le=2000)


class CustomRequestBody(BaseModel):
    """A piece described in words, sizes and pictures, for vendors who make furniture to order (no 3D model)."""

    title: str = Field(min_length=2, max_length=120)
    furniture_type: Literal[FURNITURE_KINDS]  # type: ignore[valid-type]
    dimensions: Size
    quantity: int = Field(default=1, ge=1, le=1000)
    material: str = Field(min_length=2, max_length=200)
    finish: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=4000)
    location: str = Field(min_length=2, max_length=200)
    budget_max: float | None = Field(default=None, ge=0, le=100_000_000)
    deadline: date | None = None
    images: list[str] = Field(default_factory=list, max_length=8)


@router.post("/requests", status_code=status.HTTP_201_CREATED)
async def create_custom_request(body: CustomRequestBody, request: Request, user_id: User) -> dict[str, Any]:
    """Publish a made-to-order request to every approved vendor. The pictures must already be on Sha2ty (POST
    /api/files); the specification is stored as sent (spec.brief) and never changes after publishing."""
    if not all(u.startswith(file_prefixes(get_settings())) and ".." not in u and len(u) <= 1000 for u in body.images):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Upload the reference pictures to Sha2ty first.")
    if body.deadline and body.deadline < date.today():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "The wanted-by date has passed.")
    brief = {
        "furniture_type": body.furniture_type,
        "dimensions": body.dimensions.model_dump(),
        "material": body.material.strip(),
        "finish": body.finish.strip(),
        "images": body.images,
    }
    row = {
        "customer_id": user_id, "title": body.title.strip(), "scope": "item", "service": "furniture",
        "description": body.description.strip(), "location": body.location.strip(), "budget_max": body.budget_max,
        "deadline": body.deadline.isoformat() if body.deadline else None, "status": "published",
        "spec": {"brief": brief, "quantity": body.quantity, "audience": "open", "vendor": None},
        "audience": "open",
    }
    settings = get_settings()
    client: httpx.AsyncClient = request.app.state.auth_http
    try:
        response = await client.post(
            f"{settings.supabase_url.rstrip('/')}/rest/v1/freelance_tasks",
            headers={**service_headers(settings), "Prefer": "return=representation"},
            json=row,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not send the request.") from exc
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not send the request.")
    return response.json()[0]


@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_request(task_id: UUID, request: Request, user_id: User) -> None:
    """The customer deletes their own request, whatever its state: the row is removed from the database and its bids,
    chat, proposals and review go with it (the tables' ON DELETE CASCADE), so it is gone from every list, the
    vendors' included. Their requests cancelled before are cleared too. (Admins: DELETE /api/admin/tasks/{id}.)"""
    settings = get_settings()
    base = f"{settings.supabase_url.rstrip('/')}/rest/v1/freelance_tasks"
    headers = {**service_headers(settings), "Prefer": "return=representation"}
    client: httpx.AsyncClient = request.app.state.auth_http
    try:
        # One statement checks the owner and the status and deletes: nothing slips in between.
        response = await client.delete(f"{base}?id=eq.{task_id}&customer_id=eq.{user_id}", headers=headers)
        await client.delete(f"{base}?customer_id=eq.{user_id}&status=eq.cancelled", headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not cancel the request.") from exc
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not cancel the request.")
    if not response.json():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Only your own request can be deleted.")


@router.get("/vendors/brands")
async def vendor_brands(request: Request, ids: str = "") -> list[dict[str, Any]]:
    """Name and logo of approved vendors (customers cannot read the vendors table), e.g. for the fabrics and woods
    shown in the editor. Public business details only."""
    wanted = []
    for raw in ids.split(",")[:100]:
        try:
            wanted.append(str(UUID(raw.strip())))
        except ValueError:
            continue
    if not wanted:
        return []
    settings = get_settings()
    client: httpx.AsyncClient = request.app.state.auth_http
    try:
        response = await client.get(
            f"{settings.supabase_url.rstrip('/')}/rest/v1/vendors?id=in.({','.join(wanted)})&select=*",
            headers=service_headers(settings),
        )
    except httpx.HTTPError:
        return []
    rows = response.json() if response.status_code < 400 else []
    # `platform`: Sha2ty's own studio (the admins' catalogue), so lists can leave its items out where only vendors' belong.
    return [{"id": r["id"], "name": r.get("name") or "", "logo_url": r.get("logo_url"), "platform": bool(r.get("platform"))} for r in rows if r.get("status", "approved") == "approved"]


@router.get("/vendors/{vendor_id}/profile")
async def vendor_profile(vendor_id: UUID, request: Request) -> dict[str, Any]:
    """A vendor's public page: business details (no email or owner name), their listings, completed projects,
    and customer reviews. Public: shown to customers comparing bids."""
    settings = get_settings()
    base = f"{settings.supabase_url.rstrip('/')}/rest/v1"
    headers = service_headers(settings)
    client: httpx.AsyncClient = request.app.state.auth_http

    async def rows(path: str) -> list[dict[str, Any]]:
        try:
            response = await client.get(f"{base}/{path}", headers=headers)
        except httpx.HTTPError:
            return []
        return response.json() if response.status_code < 400 else []

    vendors = await rows(f"vendors?id=eq.{vendor_id}&select=*")
    vendor = vendors[0] if vendors else None
    if vendor is None or vendor.get("status", "approved") != "approved":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Vendor not found.")
    models = await rows(f"models?vendor_id=eq.{vendor_id}&status=eq.ready&select=id,title,glb_url,width_m,height_m,depth_m,price,price_unit,source_image_url,thumbnail_url&order=created_at.desc&limit=24")
    if not models:  # before model_thumbnail.sql
        models = await rows(f"models?vendor_id=eq.{vendor_id}&status=eq.ready&select=id,title,glb_url,width_m,height_m,depth_m,price,price_unit,source_image_url&order=created_at.desc&limit=24")
    won = await rows(f"task_bids?vendor_id=eq.{vendor_id}&status=eq.accepted&select=price,days,task:freelance_tasks(id,title,status,spec,created_at)")
    projects = [
        {"id": b["task"]["id"], "title": b["task"]["title"], "status": b["task"]["status"], "created_at": b["task"]["created_at"],
         "thumbnail_url": ((b["task"].get("spec") or {}).get("design") or {}).get("thumbnail_url"), "days": b.get("days")}
        for b in won if b.get("task")
    ]
    reviews = await rows(f"vendor_reviews?vendor_id=eq.{vendor_id}&select=id,task_id,rating,comment,created_at&order=created_at.desc&limit=50")
    bids = await rows(f"task_bids?vendor_id=eq.{vendor_id}&select=id")
    completed = [p for p in projects if p["status"] == "completed"]
    rating = round(sum(r["rating"] for r in reviews) / len(reviews), 2) if reviews else None
    return {
        "vendor": {k: vendor.get(k) for k in ("id", "name", "location", "logo_url", "created_at")}
        | {"service_areas": vendor.get("service_areas") or [], "bio": vendor.get("bio") or "",
           # Only pictures stored on Sha2ty: a vendor cannot point the page at someone else's site.
           "portfolio": [p for p in vendor.get("portfolio") or [] if isinstance(p, dict) and str(p.get("url", "")).startswith(file_prefixes(settings))][:12]},
        "stats": {"completed_projects": len(completed), "active_projects": len(projects) - len(completed),
                  "bids": len(bids), "rating": rating, "reviews": len(reviews)},
        "listings": models,
        "projects": projects,
        "reviews": reviews,
    }
