"""Which service turns vendors' photos into 3D models (Meshy, GPT Astra or GPT Sol): chosen by the admin, never by the vendor.
Also which GPT combines two photos into one room for Match ("match_model": GPT Astra or GPT Sol, default Sol).

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
MATCH_LABELS: dict[Provider, str] = {"astra": "GPT Astra", "sol": "GPT Sol"}
GENERATOR, MATCH = "model_generator", "match_model"
DEFAULTS: dict[str, Provider] = {GENERATOR: "meshy", MATCH: "sol"}
CACHE_SECONDS = 30.0
_cached: dict[str, tuple[float, Provider]] = {}


def _rest(settings: Settings) -> str:
    return f"{settings.supabase_url.rstrip('/')}/rest/v1/app_settings"


def configured(settings: Settings, provider: Provider) -> bool:
    key = settings.meshy_api_key if provider == "meshy" else settings.openai_api_key
    return bool(key.get_secret_value())


def labels(key: str) -> dict[Provider, str]:
    return MATCH_LABELS if key == MATCH else LABELS


async def current(settings: Settings, client: httpx.AsyncClient, key: str = GENERATOR) -> Provider:
    now = time.monotonic()
    hit = _cached.get(key)
    if hit and hit[0] > now:
        return hit[1]
    provider = DEFAULTS[key]
    try:
        response = await client.get(f"{_rest(settings)}?key=eq.{key}&select=value", headers=service_headers(settings), timeout=10)
        rows = response.json() if response.status_code < 400 else []
        if rows and rows[0].get("value") in labels(key):
            provider = rows[0]["value"]
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("%s setting unreadable (%s); using %s", key, exc, provider)
    _cached[key] = (now + CACHE_SECONDS, provider)
    return provider


async def set_current(settings: Settings, client: httpx.AsyncClient, provider: Provider, key: str = GENERATOR) -> Provider:
    response = await client.post(
        _rest(settings), json={"key": key, "value": provider},
        headers={**service_headers(settings), "Prefer": "resolution=merge-duplicates,return=minimal"}, timeout=10,
    )
    if response.status_code >= 400:
        raise RuntimeError("Could not save the setting. Run supabase/app_settings.sql in Supabase.")
    _cached[key] = (time.monotonic() + CACHE_SECONDS, provider)
    return provider
