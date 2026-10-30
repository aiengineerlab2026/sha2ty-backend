from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx
from fastapi import HTTPException, status

from app.core.config import Settings
from app.services.auth import service_headers


class PlanStore:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client
        self.base = f"{settings.supabase_url.rstrip('/')}/rest/v1"

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = service_headers(self.settings)
        headers["Prefer"] = "return=representation"
        if extra:
            headers.update(extra)
        return headers

    async def catalog(self, query: str = "", vendor_id: str | None = None, limit: int = 40, offset: int = 0, model_id: str | None = None) -> dict[str, Any]:
        params = [
            "select=id,title,description,price,price_unit,category,shape,length2_m,width_m,height_m,depth_m,glb_url,source_image_url,vendor_id,status,vendors(id,name,logo_url),model_variants(image_url,sort_order)",
            "status=eq.ready",
            "glb_url=not.is.null",
            "order=title.asc",
            f"limit={limit}",
            f"offset={offset}",
        ]
        if vendor_id:
            params.append(f"vendor_id=eq.{vendor_id}")
        if model_id:
            params.append(f"id=eq.{model_id}")
        if query:
            params.append(f"title=ilike.{quote('%' + query + '%')}")
        response = await self.client.get(f"{self.base}/models?{'&'.join(params)}", headers=self._headers())
        # A newer migration (pricing / categories / shape) not applied yet: drop only the columns the
        # database says are missing and serve the rest (prices stay when only `shape` is missing).
        optional = ["price", "price_unit", "category", "shape", "length2_m"]
        while response.status_code == 400:
            missing = [c for c in optional if f"models.{c}" in response.text or f"'{c}'" in response.text]
            if not missing:
                break
            for c in missing:
                params[0] = params[0].replace(f"{c},", "")
                optional.remove(c)
            response = await self.client.get(f"{self.base}/models?{'&'.join(params)}", headers=self._headers())
        if response.status_code >= 400:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not load the furniture catalog.")
        rows = response.json()
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            vendor = row.get("vendors") or {}
            company_id = vendor.get("id") or row.get("vendor_id") or "unknown"
            groups.setdefault(
                company_id,
                {"id": company_id, "name": vendor.get("name") or "Unknown studio", "logo_url": vendor.get("logo_url"), "models": []},
            )
            variants = sorted(row.get("model_variants") or [], key=lambda item: item.get("sort_order") or 0)
            groups[company_id]["models"].append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "description": row.get("description") or "",
                    "price": _num(row.get("price")),
                    "price_unit": row.get("price_unit") or "item",
                    "category": row.get("category") or "furniture",
                    "shape": row.get("shape") or "straight",
                    "length2_m": _num(row.get("length2_m")),
                    "variants": [v["image_url"] for v in variants if v.get("image_url")],
                    "width_m": _num(row.get("width_m")),
                    "height_m": _num(row.get("height_m")),
                    "depth_m": _num(row.get("depth_m")),
                    "glb_url": row.get("glb_url"),
                    "thumbnail": (variants[0]["image_url"] if variants else None) or row.get("source_image_url"),
                    "company_id": company_id,
                    "company": groups[company_id]["name"],
                    "company_logo": groups[company_id]["logo_url"],
                }
            )
        return {"companies": list(groups.values()), "count": len(rows)}

    async def insert_plan(self, body: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.post(f"{self.base}/floor_plans", headers=self._headers(), json=body)
        if response.status_code >= 400:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, response.text[:300] or "Could not save the plan.")
        data = response.json()
        return data[0] if isinstance(data, list) else data

    async def get_plan(self, plan_id: str, owner_id: str) -> dict[str, Any]:
        response = await self.client.get(
            f"{self.base}/floor_plans?id=eq.{plan_id}&owner_id=eq.{owner_id}&select=*",
            headers=self._headers(),
        )
        rows = response.json() if response.status_code < 400 else []
        if not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Plan not found.")
        return rows[0]

    async def list_plans(self, owner_id: str) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.base}/floor_plans?owner_id=eq.{owner_id}&select=id,name,updated_at,source&order=updated_at.desc",
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not list plans.")
        return response.json()

    async def update_plan(self, plan_id: str, owner_id: str, body: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.patch(
            f"{self.base}/floor_plans?id=eq.{plan_id}&owner_id=eq.{owner_id}",
            headers=self._headers(),
            json=body,
        )
        if response.status_code >= 400:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not update the plan.")
        rows = response.json()
        if not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Plan not found.")
        return rows[0] if isinstance(rows, list) else rows

    async def delete_plan(self, plan_id: str, owner_id: str) -> None:
        response = await self.client.delete(
            f"{self.base}/floor_plans?id=eq.{plan_id}&owner_id=eq.{owner_id}",
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not delete the plan.")


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
