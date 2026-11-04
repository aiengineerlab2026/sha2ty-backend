"""Vendors' proposed changes to a requested model (supabase/proposals.sql).

While a design request is open, a vendor it is open to may load the requested model, change it in the editor and
send the result with a note. The customer sees each proposal as a full specification and accepts it (a new version
of their design, and the request's specification from then on) or rejects it. Every write is here, with the
service key; the editor's item is validated like a design save and its files must be the vendor's own uploads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.designs import Designs, User, check_textures, own_file, validated, with_scale
from app.api.marketplace import spec_from_version
from app.core.config import get_settings
from app.schemas.design import ItemEdits
from app.services.auth import service_headers

router = APIRouter(prefix="/api", tags=["proposals"])


class ProposalBody(BaseModel):
    item: ItemEdits
    glb_url: str | None = Field(default=None, max_length=2000)
    thumbnail_url: str | None = Field(default=None, max_length=2000)
    note: str = Field(default="", max_length=2000)
    # A new version of the model file the vendor uploaded (their own upload): it becomes the design's base file,
    # and `item`'s parts and edits are on it.
    model_url: str | None = Field(default=None, max_length=2000)


class DecisionBody(BaseModel):
    accept: bool


def _rest() -> str:
    return f"{get_settings().supabase_url.rstrip('/')}/rest/v1"


async def _get(request: Request, path: str, what: str) -> list[dict[str, Any]]:
    client: httpx.AsyncClient = request.app.state.auth_http
    try:
        response = await client.get(f"{_rest()}/{path}", headers=service_headers(get_settings()))
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not {what}.") from exc
    if response.status_code >= 400:
        missing = "task_proposals" in response.text
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE if missing else status.HTTP_502_BAD_GATEWAY, "Run supabase/proposals.sql first." if missing else f"Could not {what}.")
    return response.json()


async def _write(request: Request, method: str, path: str, what: str, json: Any = None) -> list[dict[str, Any]]:
    client: httpx.AsyncClient = request.app.state.auth_http
    try:
        response = await client.request(method, f"{_rest()}/{path}", headers={**service_headers(get_settings()), "Prefer": "return=representation"}, json=json)
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not {what}.") from exc
    if response.status_code >= 400:
        if "task_proposals" in response.text:  # the table is missing: say what to run
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Proposals are not set up yet: run backend/supabase/proposals.sql in Supabase.")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not {what}.")
    return response.json() if response.content else []


async def _task(request: Request, task_id: UUID) -> dict[str, Any]:
    rows = await _get(request, f"freelance_tasks?id=eq.{task_id}&select=*", "load the request")
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found.")
    return rows[0]


async def _vendor_task(request: Request, task_id: UUID, user_id: str) -> dict[str, Any]:
    """An open design request this approved vendor may take part in."""
    task = await _task(request, task_id)
    vendor = (await _get(request, f"vendors?id=eq.{user_id}&select=id,status", "check the vendor") or [None])[0]
    if not vendor or vendor.get("status", "approved") != "approved" or (task.get("audience") == "vendor_exclusive" and task.get("exclusive_vendor_id") != user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found.")
    if task["status"] != "published":
        raise HTTPException(status.HTTP_409_CONFLICT, "This request is closed.")
    if not task.get("design_id") or not task.get("design_version_id"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Only requests for a customised model can get proposed changes.")
    return task


def _requested(task: dict[str, Any]) -> list[str] | None:
    design = (task.get("spec") or {}).get("design") or {}
    return None if design.get("whole_model", True) else [p["id"] for p in design.get("parts") or [] if p.get("requested")]


def _spec(task: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    """The proposal as vendors' specifications show it: the request's parts, at the proposed size and materials."""
    design = {"id": task.get("design_id"), "title": task.get("title")}
    version = {"id": proposal["id"], "number": 0, "data": proposal["data"], "glb_url": proposal.get("glb_url"), "thumbnail_url": proposal.get("thumbnail_url")}
    names = {c.get("id") for c in (((proposal["data"].get("item") or {}).get("components") or {}).get("components") or [])}
    wanted = _requested(task)
    try:
        return spec_from_version(design, version, [p for p in wanted if p in names] if wanted else None)["design"]
    except HTTPException:  # the vendor removed every requested part: the whole proposed model
        return spec_from_version(design, version)["design"]


@router.get("/tasks/{task_id}/proposal-base")
async def proposal_base(task_id: UUID, request: Request, user_id: User, store: Designs) -> dict[str, Any]:
    """What the vendor starts editing from: the requested version of the customer's model."""
    task = await _vendor_task(request, task_id, user_id)
    version = await store.version(task["design_id"], version_id=task["design_version_id"])
    return {"item": (version.get("data") or {}).get("item"), "glb_url": version.get("glb_url"), "title": task.get("title")}


@router.post("/tasks/{task_id}/proposals", status_code=status.HTTP_201_CREATED)
async def propose(task_id: UUID, body: ProposalBody, request: Request, user_id: User, store: Designs) -> dict[str, Any]:
    """One open proposal per vendor and request: sending again replaces the pending one."""
    task = await _vendor_task(request, task_id, user_id)
    base = await store.version(task["design_id"], version_id=task["design_version_id"])
    check_textures(body.item)
    source = dict(base["data"]["item"]["source"])
    if body.model_url:
        source.update(glb_url=own_file(body.model_url, user_id, "new model file"), source_glb_url=None)
    item = {**body.item.model_dump(mode="json", exclude_none=True), "source": source}
    data = validated({"schema": "sha2ty.design/1", "kind": "item", "item": with_scale(item)})
    row = {
        "task_id": str(task_id), "vendor_id": user_id, "base_version_id": task["design_version_id"], "data": data,
        "glb_url": own_file(body.glb_url, user_id, "model file"), "thumbnail_url": own_file(body.thumbnail_url, user_id, "thumbnail"),
        "note": body.note.strip(), "status": "pending",
    }
    await _write(request, "DELETE", f"task_proposals?task_id=eq.{task_id}&vendor_id=eq.{user_id}&status=eq.pending", "replace your proposal")
    saved = (await _write(request, "POST", "task_proposals", "send the proposal", json=row))[0]
    return {**saved, "spec": _spec(task, saved)}


@router.get("/tasks/{task_id}/proposals")
async def list_proposals(task_id: UUID, request: Request, user_id: User) -> list[dict[str, Any]]:
    """The customer sees every proposal on their request; a vendor only their own."""
    task = await _task(request, task_id)
    mine = task["customer_id"] == user_id
    rows = await _get(request, f"task_proposals?task_id=eq.{task_id}{'' if mine else f'&vendor_id=eq.{user_id}'}&select=*&order=created_at.desc", "load the proposals")
    return [{**r, "spec": _spec(task, r)} for r in rows]


@router.post("/proposals/{proposal_id}/decision")
async def decide(proposal_id: UUID, body: DecisionBody, request: Request, user_id: User, store: Designs) -> dict[str, Any]:
    """Accepting saves the proposal as the design's newest version and makes it the request's specification
    (vendors bid on it from then on); other proposals stay open. Rejecting only closes this one."""
    proposal = (await _get(request, f"task_proposals?id=eq.{proposal_id}&select=*", "load the proposal") or [None])[0]
    task = await _task(request, UUID(proposal["task_id"])) if proposal else None
    if not proposal or not task or task["customer_id"] != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Proposal not found.")
    if proposal["status"] != "pending" or task["status"] != "published":
        raise HTTPException(status.HTTP_409_CONFLICT, "This proposal was already answered or the request is closed.")
    decided = {"status": "accepted" if body.accept else "rejected", "decided_at": datetime.now(timezone.utc).isoformat()}
    if body.accept:
        if not task.get("design_id"):
            raise HTTPException(status.HTTP_409_CONFLICT, "The design of this request was deleted.")
        design = await store.get_design(task["design_id"], user_id)
        version = await store.add_version(
            str(design["id"]), user_id, proposal["data"], glb_url=proposal.get("glb_url"), thumbnail_url=proposal.get("thumbnail_url"),
            note="Vendor's proposed change",
        )
        spec = task.get("spec") or {}
        wanted = _requested(task)
        names = {c.get("id") for c in (((proposal["data"].get("item") or {}).get("components") or {}).get("components") or [])}
        fresh = spec_from_version(design, {**version, "data": proposal["data"]}, [p for p in wanted if p in names] or None if wanted else None)
        # Point the request at the new version first, so tidying old versions keeps it.
        await _write(request, "PATCH", f"freelance_tasks?id=eq.{task['id']}", "update the request", json={"spec": {**spec, **fresh}, "design_version_id": version["id"]})
        await store.prune_versions(str(design["id"]), version["id"])
    saved = (await _write(request, "PATCH", f"task_proposals?id=eq.{proposal_id}", "save your answer", json=decided))[0]
    return {**saved, "spec": _spec(task, saved)}
