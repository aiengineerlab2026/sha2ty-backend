"""Which service turns vendors' photos into 3D models (Meshy, GPT Astra or GPT Sol): chosen by the admin, never by the vendor.

Stored in public.app_settings under "model_generator" (supabase/app_settings.sql), read with the service key and
cached briefly like the vision model; MODEL_GENERATOR (default meshy) when unset or unreadable.
"""

from __future__ import annotations

import logging
import time

import httpx

from app.core.config import Settings
from app.domain.generation import Provider
from app.services.auth import service_headers

logger = logging.getLogger(__name__)

LABELS: dict[Provider, str] = {"meshy": "Meshy", "astra": "GPT Astra", "sol": "GPT Sol"}
CACHE_SECONDS = 30.0
_cached: tuple[float, Provider] | None = None


def _rest(settings: Settings) -> str:
    return f"{settings.supabase_url.rstrip('/')}/rest/v1/app_settings"


def configured(settings: Settings, provider: Provider) -> bool:
    key = settings.meshy_api_key if provider == "meshy" else settings.openai_api_key
    return bool(key.get_secret_value())


async def current(settings: Settings, client: httpx.AsyncClient) -> Provider:
    global _cached
    now = time.monotonic()
    if _cached and _cached[0] > now:
        return _cached[1]
    provider: Provider = "meshy"
    try:
        response = await client.get(f"{_rest(settings)}?key=eq.model_generator&select=value", headers=service_headers(settings), timeout=10)
        rows = response.json() if response.status_code < 400 else []
        if rows and rows[0].get("value") in LABELS:
            provider = rows[0]["value"]
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("3D generator setting unreadable (%s); using %s", exc, provider)
    _cached = (now + CACHE_SECONDS, provider)
    return provider


async def set_current(settings: Settings, client: httpx.AsyncClient, provider: Provider) -> Provider:
    global _cached
    response = await client.post(
        _rest(settings), json={"key": "model_generator", "value": provider},
        headers={**service_headers(settings), "Prefer": "resolution=merge-duplicates,return=minimal"}, timeout=10,
    )
    if response.status_code >= 400:
        raise RuntimeError("Could not save the 3D generator. Run supabase/app_settings.sql in Supabase.")
    _cached = (time.monotonic() + CACHE_SECONDS, provider)
    return provider
