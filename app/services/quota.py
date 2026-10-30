from __future__ import annotations

from typing import Literal

import httpx
from fastapi import HTTPException, status

from app.core.config import Settings
from app.services.auth import service_headers

Provider = Literal["meshy", "astra"]
NOT_SET_UP = "Generation limits are not set up. Run backend/supabase/generation_quota.sql in Supabase."


async def use_generation(vendor_id: str, provider: Provider, delta: int, settings: Settings, client: httpx.AsyncClient) -> int | None:
    """Take (delta=1) or refund (delta=-1) one try. Returns tries used after, or None when the limit is reached."""
    response = await client.post(
        f"{settings.supabase_url.rstrip('/')}/rest/v1/rpc/use_generation",
        headers=service_headers(settings),
        json={"v": vendor_id, "p": provider, "delta": delta, "max_calls": settings.generation_limit},
    )
    if response.status_code >= 400:
        # Fail closed: without the table nobody could be held to the limit.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NOT_SET_UP)
    return response.json()


async def generation_usage(vendor_id: str, settings: Settings, client: httpx.AsyncClient) -> dict[str, int]:
    response = await client.get(
        f"{settings.supabase_url.rstrip('/')}/rest/v1/generation_usage",
        headers=service_headers(settings),
        params={"select": "provider,used", "vendor_id": f"eq.{vendor_id}"},
    )
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NOT_SET_UP)
    used = {"meshy": 0, "astra": 0}
    used.update({row["provider"]: row["used"] for row in response.json()})
    return used
