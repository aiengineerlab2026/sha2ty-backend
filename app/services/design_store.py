"""designs / design_versions in Supabase (supabase/designs.sql), with the service key: every query filters by owner."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
from fastapi import HTTPException, status

from app.core.config import Settings
from app.services.auth import service_headers

LIST_FIELDS = "id,kind,title,source_model_id,plan_id,current_version_id,version_count,thumbnail_url,created_at,updated_at"
VERSION_FIELDS = "id,design_id,number,schema,glb_url,thumbnail_url,note,restored_from,created_at"


class DesignStore:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client
        self.base = f"{settings.supabase_url.rstrip('/')}/rest/v1"

    def _headers(self) -> dict[str, str]:
        return {**service_headers(self.settings), "Prefer": "return=representation"}

    async def _call(self, method: str, path: str, what: str, **kwargs: Any) -> Any:
        try:
            response = await self.client.request(method, f"{self.base}/{path}", headers=self._headers(), **kwargs)
        except httpx.HTTPError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not {what}.") from exc
        text = response.text
        missing = "designs" in text and ("does not exist" in text or "schema cache" in text or "PGRST205" in text)
        if missing:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Designs are not set up yet: run backend/supabase/designs.sql.")
        if response.status_code == 404 or (response.status_code == 400 and "P0002" in text):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Design not found.")
        if response.status_code >= 400:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not {what}.")
        return response.json() if response.content else None

    async def list_designs(self, owner: str, kind: str | None, plan_id: UUID | None, limit: int, offset: int) -> list[dict[str, Any]]:
        query = f"designs?owner_id=eq.{owner}&deleted_at=is.null&select={LIST_FIELDS}&order=updated_at.desc&limit={limit}&offset={offset}"
        if kind:
            query += f"&kind=eq.{kind}"
        if plan_id:
            query += f"&plan_id=eq.{plan_id}"
        return await self._call("GET", query, "list designs")

    async def get_design(self, design_id: UUID, owner: str) -> dict[str, Any]:
        rows = await self._call("GET", f"designs?id=eq.{design_id}&owner_id=eq.{owner}&deleted_at=is.null&select={LIST_FIELDS}", "load the design")
        if not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Design not found.")
        return rows[0]

    async def versions(self, design_id: UUID) -> list[dict[str, Any]]:
        return await self._call("GET", f"design_versions?design_id=eq.{design_id}&select={VERSION_FIELDS}&order=number.desc", "list versions")

    async def version(self, design_id: UUID, number: int | None = None, version_id: str | None = None) -> dict[str, Any]:
        where = f"id=eq.{version_id}" if version_id else f"number=eq.{int(number or 0)}"
        rows = await self._call("GET", f"design_versions?design_id=eq.{design_id}&{where}&select={VERSION_FIELDS},data", "load the version")
        if not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Version not found.")
        return rows[0]

    async def by_token(self, owner: str, token: UUID) -> dict[str, Any] | None:
        rows = await self._call("GET", f"designs?owner_id=eq.{owner}&client_token=eq.{token}&deleted_at=is.null&select={LIST_FIELDS}", "load the design")
        return rows[0] if rows else None

    async def library(self, owner: str, limit: int, offset: int) -> list[dict[str, Any]]:
        """Customised items with their latest version (data included: the planner replays its edits)."""
        current = f"current:design_versions!designs_current_version_fk({VERSION_FIELDS},data)"
        return await self._call(
            "GET",
            f"designs?owner_id=eq.{owner}&kind=eq.item&deleted_at=is.null&select={LIST_FIELDS},{current}&order=updated_at.desc&limit={limit}&offset={offset}",
            "list your customised models",
        )

    async def insert_design(self, row: dict[str, Any]) -> dict[str, Any]:
        rows = await self._call("POST", "designs", "create the design", json=row)
        return rows[0]

    async def remove_design_row(self, design_id: str, owner: str) -> None:
        """Only to undo a design whose first version could not be written."""
        await self._call("DELETE", f"designs?id=eq.{design_id}&owner_id=eq.{owner}&version_count=eq.0", "clean up the design")

    async def add_version(
        self, design_id: str, owner: str, data: dict[str, Any], *, glb_url: str | None, thumbnail_url: str | None,
        note: str, restored_from: str | None = None, client_token: UUID | None = None,
    ) -> dict[str, Any]:
        """Numbered, made current and owner-checked in one locked database function (idempotent on client_token)."""
        row = await self._call("POST", "rpc/add_design_version", "save the version", json={
            "p_design": design_id, "p_owner": owner, "p_data": data, "p_glb_url": glb_url,
            "p_thumbnail_url": thumbnail_url, "p_note": note, "p_restored_from": restored_from,
            "p_client_token": str(client_token) if client_token else None,
        })
        return row[0] if isinstance(row, list) else row

    async def follow_requests(self, design: dict[str, Any], version: dict[str, Any]) -> int:
        """The design's active requests (open, awarded, in progress) move to its newest version: vendors always see the
        model as the customer last saved it. Each keeps its parts (those still in the model; else the whole model),
        quantity and audience. Returns how many were updated."""
        from app.api.marketplace import OPEN, spec_from_version  # the request spec's single source

        tasks = await self._call("GET", f"freelance_tasks?design_id=eq.{design['id']}&status=in.({','.join(OPEN)})&select=id,spec,design_version_id", "load the requests")
        updated = 0
        for task in tasks or []:
            if task.get("design_version_id") == version["id"]:
                continue
            old = task.get("spec") or {}
            wanted = None if (old.get("design") or {}).get("whole_model", True) else [p["id"] for p in (old.get("design") or {}).get("parts") or [] if p.get("requested")]
            names = {c.get("id") for c in (((version.get("data") or {}).get("item") or {}).get("components") or {}).get("components") or []}
            parts = [p for p in wanted or [] if p in names] or None
            spec = {**old, **spec_from_version(design, version, parts)}
            await self._call("PATCH", f"freelance_tasks?id=eq.{task['id']}", "update the request", json={"design_version_id": version["id"], "spec": spec})
            updated += 1
        return updated

    async def prune_versions(self, design_id: str, keep_id: str) -> None:
        """Only the latest version stays: older ones go, except those a request was made from (the request keeps
        showing exactly what was sent)."""
        tasks = await self._call("GET", f"freelance_tasks?design_id=eq.{design_id}&select=design_version_id", "check requests")
        keep = {keep_id, *(t["design_version_id"] for t in tasks or [] if t.get("design_version_id"))}
        await self._call("DELETE", f"design_versions?design_id=eq.{design_id}&id=not.in.({','.join(sorted(keep))})", "tidy versions")

    async def update_design(self, design_id: UUID, owner: str, fields: dict[str, Any]) -> dict[str, Any]:
        rows = await self._call("PATCH", f"designs?id=eq.{design_id}&owner_id=eq.{owner}&deleted_at=is.null", "update the design", json=fields)
        if not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Design not found.")
        return rows[0]

    async def user_model(self, model_id: UUID, user_id: str) -> dict[str, Any]:
        """One of the user's own models (made from their photo), ready with a 3D file; never anyone else's."""
        rows = await self._call("GET", f"user_models?id=eq.{model_id}&owner_id=eq.{user_id}&select=*", "load your model")
        row = rows[0] if rows else None
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found.")
        if not row.get("glb_url") or row.get("status") in ("generating", "failed"):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "This model is not finished yet.")
        return row

    async def catalog_model(self, model_id: UUID, user_id: str) -> dict[str, Any]:
        """A model the user may start a design from: a ready listing of an approved vendor, or one of their own."""
        try:
            rows = await self._call("GET", f"models?id=eq.{model_id}&select=*,vendors(status,platform)", "load the model")
        except HTTPException as exc:  # before platform_vendor.sql: no platform column
            if exc.status_code != status.HTTP_502_BAD_GATEWAY:
                raise
            rows = await self._call("GET", f"models?id=eq.{model_id}&select=*,vendors(status)", "load the model")
        row = rows[0] if rows else None
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found.")
        own = str(row.get("vendor_id")) == user_id
        vendor_status = (row.get("vendors") or {}).get("status", "approved")  # before vendor_approval.sql: all listed
        if not own and (row.get("status") != "ready" or vendor_status != "approved"):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found.")
        if not row.get("glb_url"):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "This listing has no 3D model to customise.")
        return row
