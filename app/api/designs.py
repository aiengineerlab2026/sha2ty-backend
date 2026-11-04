"""Customer designs with an immutable version history (supabase/designs.sql, app/schemas/design.py).

Every write is here: the body is validated against the shared design schema, ownership is checked on each query
(service key), file URLs must be the user's own uploads, and the catalogue source of an item design is taken from
the database, never from the client. Saving, restoring and duplicating only ever add versions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import ValidationError

from app.core.config import get_settings
from app.schemas.design import (
    DesignData,
    DuplicateBody,
    FromModelBody, FromUserModelBody,
    FromPlanBody,
    ItemDesign,
    ItemEdits,
    PartDocument,
    PlanSnapshot,
    RenameBody,
    RestoreBody,
    SaveVersionBody,
    SourceModel,
    plan_kind,
)
from app.services.auth import user_id_from_bearer
from app.services.design_store import DesignStore
from app.services.plan_store import PlanStore

router = APIRouter(prefix="/api/designs", tags=["designs"])


async def current_user(request: Request, authorization: Annotated[str | None, Header()] = None) -> str:
    return await user_id_from_bearer(authorization, get_settings(), request.app.state.auth_http)


def designs(request: Request) -> DesignStore:
    return DesignStore(get_settings(), request.app.state.auth_http)


def plans(request: Request) -> PlanStore:
    return PlanStore(get_settings(), request.app.state.auth_http)


User = Annotated[str, Depends(current_user)]
Designs = Annotated[DesignStore, Depends(designs)]
Plans = Annotated[PlanStore, Depends(plans)]


# ---- file URLs ----


def _public_prefix() -> str:
    return f"{get_settings().supabase_url.rstrip('/')}/storage/v1/object/public/cads/"


def own_file(url: str | None, user_id: str, what: str) -> str | None:
    """A GLB or thumbnail a version points at: only the user's own uploads (cads/<user id>/…)."""
    if url is None:
        return None
    if not url.startswith(f"{_public_prefix()}{user_id}/") or ".." in url:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"The {what} must be a file you uploaded.")
    return url


def check_textures(edits: ItemEdits) -> None:
    """Texture images: any file in the platform's storage (the user's uploads or a vendor's texture library).
    Browser-only blob: URLs would not survive a reload, so they are refused."""
    for part, edit in edits.material_edits.items():
        if edit.mapUrl is not None and (not edit.mapUrl.startswith(_public_prefix()) or ".." in edit.mapUrl):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"The texture of part {part!r} must be uploaded first.")


def validated(data: dict[str, Any]) -> dict[str, Any]:
    try:
        return DesignData.model_validate(data).dump()
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.errors(include_url=False, include_context=False)) from exc


def snapshot(plan: dict[str, Any]) -> PlanSnapshot:
    return PlanSnapshot(
        plan_id=plan["id"], name=plan.get("name") or "", source=plan.get("source") or "dimensions",
        source_image_url=plan.get("source_image_url"), geometry=plan.get("geometry") or {},
        items=plan.get("items") or [], camera=plan.get("camera"), warnings=plan.get("warnings") or [],
    )


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def with_scale(item: dict[str, Any]) -> dict[str, Any]:
    """`scale` from the saved size and the catalogue size (never the client's numbers)."""
    item.pop("scale", None)
    dims, src = item.get("dimensions"), item.get("source") or {}
    base = (src.get("width_m"), src.get("height_m"), src.get("depth_m"))
    if dims and all(base):
        item["scale"] = [round(dims[k] / b, 6) for k, b in zip(("width", "height", "depth"), base)]
    return item


async def _create(store: DesignStore, user_id: str, row: dict[str, Any], data: dict[str, Any], note: str, client_token: UUID | None = None, **version: Any) -> dict[str, Any]:
    if client_token:
        existing = await store.by_token(user_id, client_token)
        if existing:  # the same first save, sent again
            return await _full(store, existing)
        row = {**row, "client_token": str(client_token)}
    design = await store.insert_design({"owner_id": user_id, **row})
    try:
        current = await store.add_version(design["id"], user_id, data, note=note, **{"glb_url": None, "thumbnail_url": None, **version})
    except HTTPException:
        await store.remove_design_row(design["id"], user_id)
        raise
    return {**design, "current_version_id": current["id"], "version_count": current["number"], "current": current}


async def _full(store: DesignStore, design: dict[str, Any]) -> dict[str, Any]:
    current = await store.version(design["id"], version_id=design["current_version_id"]) if design.get("current_version_id") else None
    return {**design, "current": current}


# ---- reads ----


@router.get("")
async def list_designs(
    user_id: User, store: Designs, kind: Literal["item", "room", "apartment"] | None = None, plan_id: UUID | None = None,
    limit: int = 30, offset: int = 0,
) -> dict[str, Any]:
    limit, offset = max(1, min(limit, 100)), max(0, offset)
    return {"designs": await store.list_designs(user_id, kind, plan_id, limit, offset), "limit": limit, "offset": offset}


@router.get("/library")
async def customised_models(user_id: User, store: Designs, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """The planner's "My customised models": the user's item designs with their latest version."""
    limit, offset = max(1, min(limit, 100)), max(0, offset)
    return {"models": await store.library(user_id, limit, offset), "limit": limit, "offset": offset}


@router.get("/{design_id}")
async def get_design(design_id: UUID, user_id: User, store: Designs) -> dict[str, Any]:
    return await _full(store, await store.get_design(design_id, user_id))


@router.get("/{design_id}/versions")
async def list_versions(design_id: UUID, user_id: User, store: Designs) -> dict[str, Any]:
    await store.get_design(design_id, user_id)
    return {"versions": await store.versions(design_id)}


@router.get("/{design_id}/versions/{number}")
async def get_version(design_id: UUID, number: int, user_id: User, store: Designs) -> dict[str, Any]:
    await store.get_design(design_id, user_id)
    return await store.version(design_id, number)


# ---- creating ----


@router.post("/from-model", status_code=status.HTTP_201_CREATED)
async def from_model(body: FromModelBody, user_id: User, store: Designs) -> dict[str, Any]:
    """Customise a catalogue model: the design references the listing; the vendor's model is never changed."""
    model = await store.catalog_model(body.model_id, user_id)
    w, h, d = (_num(model.get(k)) for k in ("width_m", "height_m", "depth_m"))
    platform = bool((model.get("vendors") or {}).get("platform"))
    item = ItemDesign(
        source=SourceModel(
            model_id=model["id"], title=model.get("title") or "", vendor_id=model.get("vendor_id"),
            source_type="platform" if platform else "vendor",
            glb_url=model["glb_url"], source_glb_url=model.get("source_glb_url"), width_m=w, height_m=h, depth_m=d,
        ),
        dimensions={"width": w, "height": h, "depth": d} if w and h and d else None,
        components=PartDocument.model_validate(model["components"]) if isinstance(model.get("components"), dict) else None,
    )
    fields = item.model_dump(mode="json", exclude_none=True)
    if body.item is not None:  # the first save carries the customer's edits: v1 is what they saved
        check_textures(body.item)
        fields = {**fields, **body.item.model_dump(mode="json", exclude_none=True), "source": fields["source"]}
    data = validated({"schema": "sha2ty.design/1", "kind": "item", "item": with_scale(fields)})
    return await _create(
        store, user_id,
        {"kind": "item", "title": body.title.strip() or model.get("title") or "", "source_model_id": str(body.model_id),
         "thumbnail_url": own_file(body.thumbnail_url, user_id, "thumbnail") or model.get("thumbnail_url") or model.get("source_image_url")},
        data, body.note or ("Saved" if body.item is not None else "Started from the catalogue"),
        client_token=body.client_token,
        glb_url=own_file(body.glb_url, user_id, "model file"), thumbnail_url=own_file(body.thumbnail_url, user_id, "thumbnail"),
    )


@router.post("/from-user-model", status_code=status.HTTP_201_CREATED)
async def from_user_model(body: FromUserModelBody, user_id: User, store: Designs) -> dict[str, Any]:
    """Customise the user's own model (user_models) like a catalogue one: sizes, parts and materials, then request
    it from vendors (open to all: the model is the customer's, no vendor owns it). The private model is never changed."""
    model = await store.user_model(body.user_model_id, user_id)
    w, h, d = (_num(model.get(k)) for k in ("width_m", "height_m", "depth_m"))
    item = ItemDesign(
        source=SourceModel(
            model_id=model["id"], title=model.get("title") or "", vendor_id=None, source_type="user",
            glb_url=model["glb_url"], source_glb_url=model.get("source_glb_url"), width_m=w, height_m=h, depth_m=d,
        ),
        dimensions={"width": w, "height": h, "depth": d} if w and h and d else None,
        components=PartDocument.model_validate(model["components"]) if isinstance(model.get("components"), dict) else None,
    )
    data = validated({"schema": "sha2ty.design/1", "kind": "item", "item": with_scale(item.model_dump(mode="json", exclude_none=True))})
    return await _create(
        store, user_id,
        # source_model_id stays empty: it references catalogue listings, and this model is not one.
        {"kind": "item", "title": body.title.strip() or model.get("title") or "", "thumbnail_url": model.get("thumbnail_url") or model.get("source_image_url")},
        data, "Started from my model",
    )


@router.post("/from-plan", status_code=status.HTTP_201_CREATED)
async def from_plan(body: FromPlanBody, user_id: User, store: Designs, plan_store: Plans) -> dict[str, Any]:
    """A room or apartment design from one of the user's saved plans (its first version is the plan as saved)."""
    plan = await plan_store.get_plan(str(body.plan_id), user_id)
    snap = snapshot(plan)
    kind = body.kind or plan_kind(snap)
    data = validated({"schema": "sha2ty.design/1", "kind": kind, "plan": snap.model_dump(mode="json")})
    return await _create(
        store, user_id, {"kind": kind, "title": body.title.strip() or plan.get("name") or "", "plan_id": str(body.plan_id)},
        data, "Saved from the planner",
    )


# ---- versions ----


@router.post("/{design_id}/versions", status_code=status.HTTP_201_CREATED)
async def save_version(design_id: UUID, body: SaveVersionBody, user_id: User, store: Designs, plan_store: Plans) -> dict[str, Any]:
    design = await store.get_design(design_id, user_id)
    previous = (await store.version(design_id, version_id=design["current_version_id"]))["data"]
    if design["kind"] == "item":
        if body.item is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Send the item's edits.")
        check_textures(body.item)
        # The catalogue source stays as it was when the design started.
        item = {**body.item.model_dump(mode="json", exclude_none=True), "source": previous["item"]["source"]}
        data = validated({"schema": "sha2ty.design/1", "kind": "item", "item": with_scale(item)})
    else:
        if not design.get("plan_id"):
            raise HTTPException(status.HTTP_409_CONFLICT, "This design's plan was deleted; restore a version to keep working on it.")
        snap = snapshot(await plan_store.get_plan(str(design["plan_id"]), user_id))
        data = validated({"schema": "sha2ty.design/1", "kind": design["kind"], "plan": snap.model_dump(mode="json")})
    before = {**previous, "item": with_scale(dict(previous["item"]))} if design["kind"] == "item" else previous
    if data == validated(before):
        # Nothing changed: no new version (the editor says so). The current one is returned as it is.
        current = await store.version(design_id, version_id=design["current_version_id"])
        return {**{k: v for k, v in current.items() if k != "data"}, "unchanged": True}
    saved = await store.add_version(
        str(design_id), user_id, data, glb_url=own_file(body.glb_url, user_id, "model file"),
        thumbnail_url=own_file(body.thumbnail_url, user_id, "thumbnail"), note=body.note, client_token=body.client_token,
    )
    if design["kind"] == "item":
        await store.prune_versions(str(design_id), saved["id"])  # a customised model keeps one version: the latest
    return saved


@router.post("/{design_id}/restore", status_code=status.HTTP_201_CREATED)
async def restore_version(design_id: UUID, body: RestoreBody, user_id: User, store: Designs, plan_store: Plans) -> dict[str, Any]:
    """An earlier version becomes the newest one (history is kept). Room and apartment designs also put it back
    into the planner's working plan, so the planner opens what was restored."""
    design = await store.get_design(design_id, user_id)
    old = await store.version(design_id, body.number)
    data = validated(old["data"])
    if design["kind"] != "item":
        plan = data["plan"]
        fields = {k: plan.get(k) for k in ("name", "source", "source_image_url", "geometry", "items", "camera", "warnings")}
        try:
            if not design.get("plan_id"):
                raise HTTPException(status.HTTP_404_NOT_FOUND)
            await plan_store.update_plan(str(design["plan_id"]), user_id, fields)
        except HTTPException as exc:
            if exc.status_code != status.HTTP_404_NOT_FOUND:
                raise
            created = await plan_store.insert_plan({"owner_id": user_id, **fields})  # the working plan was deleted
            await store.update_design(design_id, user_id, {"plan_id": created["id"]})
            data["plan"]["plan_id"] = created["id"]
    return await store.add_version(
        str(design_id), user_id, data, glb_url=old.get("glb_url"), thumbnail_url=old.get("thumbnail_url"),
        note=f"Restored version {body.number}", restored_from=old["id"], client_token=body.client_token,
    )


@router.post("/{design_id}/duplicate", status_code=status.HTTP_201_CREATED)
async def duplicate_design(design_id: UUID, body: DuplicateBody, user_id: User, store: Designs, plan_store: Plans) -> dict[str, Any]:
    """A new design whose first version is a copy of this one's (current or chosen) version. Room and apartment
    copies get their own working plan, so editing one never changes the other."""
    design = await store.get_design(design_id, user_id)
    old = await store.version(design_id, body.number) if body.number else await store.version(design_id, version_id=design["current_version_id"])
    data = validated(old["data"])
    row: dict[str, Any] = {"kind": design["kind"], "title": body.title.strip() or f"{design['title']} (copy)"[:200],
                           "source_model_id": design.get("source_model_id"), "thumbnail_url": old.get("thumbnail_url") or design.get("thumbnail_url")}
    if design["kind"] != "item":
        plan = data["plan"]
        created = await plan_store.insert_plan({"owner_id": user_id, **{k: plan.get(k) for k in ("name", "source", "source_image_url", "geometry", "items", "camera", "warnings")}})
        row["plan_id"] = created["id"]
        data["plan"]["plan_id"] = created["id"]
    return await _create(store, user_id, row, data, f"Copied from {design['title'] or 'a design'} v{old['number']}",
                         glb_url=old.get("glb_url"), thumbnail_url=old.get("thumbnail_url"))


# ---- design row ----


@router.patch("/{design_id}")
async def rename_design(design_id: UUID, body: RenameBody, user_id: User, store: Designs) -> dict[str, Any]:
    return await store.update_design(design_id, user_id, {"title": body.title.strip()})


@router.delete("/{design_id}")
async def delete_design(design_id: UUID, user_id: User, store: Designs) -> dict[str, str]:
    """Soft delete: the design leaves the library; its versions stay for anything that references them."""
    await store.update_design(design_id, user_id, {"deleted_at": datetime.now(timezone.utc).isoformat()})
    return {"status": "deleted"}
