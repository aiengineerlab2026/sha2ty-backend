"""The vision model the site uses to read images into JSON: floor plans (plan/vision_analysis.py) and
vendor product photos before 3D generation (product_details.py).

The admin picks it (GET/PUT /api/admin/vision); the choice is kept in public.app_settings and falls back
to PLAN_VISION_MODEL (default gemini-3.5-flash). gemini-* models go to Google's generateContent with GEMINI_API_KEY, the others to
the OpenAI Responses API with OPENAI_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import httpx
from fastapi import HTTPException

from app.core.config import Settings
from app.services.auth import service_headers

logger = logging.getLogger(__name__)

MODELS = {"gemini-3.5-flash": "Gemini 3.5 Flash", "gpt-4o-mini": "GPT-4o mini"}
RETRY_DELAY = 2.0  # seconds, doubled per attempt
RETRIES = 1  # one extra attempt on 429 / 5xx / timeouts


class VisionSettingError(RuntimeError):
    pass


CACHE_SECONDS = 30.0  # the choice is re-read this often (no restart; other API hosts follow within 30 s)
_cached: tuple[float, str] | None = None


def _rest(settings: Settings) -> str:
    return f"{settings.supabase_url.rstrip('/')}/rest/v1/app_settings"


async def current_model(settings: Settings, client: httpx.AsyncClient) -> str:
    """The admin's choice from public.app_settings (supabase/app_settings.sql), else PLAN_VISION_MODEL."""
    global _cached
    now = time.monotonic()
    if _cached and _cached[0] > now:
        return _cached[1]
    model = settings.plan_vision_model
    try:
        response = await client.get(f"{_rest(settings)}?key=eq.vision_model&select=value", headers=service_headers(settings), timeout=10)
        rows = response.json() if response.status_code < 400 else []
        if rows and rows[0].get("value") in MODELS:
            model = rows[0]["value"]
        elif response.status_code >= 400:
            logger.warning("Vision model setting unreadable (%s); using %s. Run supabase/app_settings.sql.", response.status_code, model)
    except (httpx.HTTPError, HTTPException, ValueError) as exc:
        logger.warning("Vision model setting unreadable (%s); using %s", exc, model)
    _cached = (now + CACHE_SECONDS, model)
    return model


async def set_model(settings: Settings, client: httpx.AsyncClient, model: str) -> str:
    global _cached
    if model not in MODELS:
        raise VisionSettingError(f"Unknown vision model: {model}")
    response = await client.post(
        _rest(settings), json={"key": "vision_model", "value": model},
        headers={**service_headers(settings), "Prefer": "resolution=merge-duplicates,return=minimal"}, timeout=10,
    )
    if response.status_code >= 400:
        logger.warning("Could not save the vision model: %s %s", response.status_code, response.text[:200])
        raise VisionSettingError("Could not save the vision model. Run supabase/app_settings.sql in Supabase.")
    _cached = (time.monotonic() + CACHE_SECONDS, model)
    return model


def is_gemini(model: str) -> bool:
    return model.startswith("gemini")


def configured(settings: Settings, model: str) -> bool:
    if not model:
        return False
    if is_gemini(model):
        key = settings.gemini_api_key.get_secret_value().strip()
        return bool(key) and not key.startswith("<")
    return settings.astra_configured


def client_for(state, model: str) -> httpx.AsyncClient:
    """Gemini gets the plain client: the OpenAI client carries the OpenAI key in its headers."""
    return state.auth_http if is_gemini(model) else state.openai_http


async def ask_json(content: list[dict], schema: dict, name: str, settings: Settings, client: httpx.AsyncClient, model: str) -> str | None:
    """`content` is OpenAI-style parts (input_text / input_image data URL); returns the model's JSON text."""
    if is_gemini(model):
        parts = []
        for c in content:
            if c["type"] == "input_text":
                parts.append({"text": c["text"]})
            else:
                mime, _, data = c["image_url"].removeprefix("data:").partition(";base64,")
                parts.append({"inline_data": {"mime_type": mime, "data": data}})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": schema}}
        headers = {"x-goog-api-key": settings.gemini_api_key.get_secret_value()}
    else:
        url, headers = "https://api.openai.com/v1/responses", {}  # the OpenAI client sends its key
        payload = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            # strict=False: Pydantic's schema (optional fields, defaults) is a guide, and Pydantic validates.
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": False}},
        }
    body = await _post(url, payload, headers, settings, client, f"{name} {model}")
    if body is None:
        return None
    try:
        if is_gemini(model):
            return "".join(p.get("text", "") for p in body["candidates"][0]["content"]["parts"] if not p.get("thought"))
        return next(c["text"] for item in body.get("output", []) if item.get("type") == "message" for c in item.get("content", []) if c.get("type") == "output_text")
    except (StopIteration, KeyError, IndexError, TypeError) as exc:
        logger.warning("Vision %s returned no text: %s", name, str(exc)[:200])
        return None


async def _post(url: str, payload: dict, headers: dict, settings: Settings, client: httpx.AsyncClient, what: str) -> dict | None:
    for attempt in range(RETRIES + 1):
        started = time.monotonic()
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=settings.openai_plan_timeout_seconds)
        except httpx.HTTPError as exc:
            logger.warning("Vision %s request failed (attempt %d): %s", what, attempt + 1, exc)
        else:
            data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            usage, gemini = data.get("usage") or {}, data.get("usageMetadata") or {}
            logger.info(
                "Vision %s: %s in %.1fs (input %s, output %s tokens)", what, response.status_code, time.monotonic() - started,
                usage.get("input_tokens", gemini.get("promptTokenCount")), usage.get("output_tokens", gemini.get("candidatesTokenCount")),
            )
            if response.status_code < 400:
                return data
            logger.warning("Vision %s refused: %s", what, response.text[:300])
            if (response.status_code != 429 and response.status_code < 500) or any(c in response.text for c in ("insufficient_quota", "credit_balance_exhausted")):
                return None  # a bad request, or no credits left, does not get better by repeating it
        if attempt < RETRIES:
            await asyncio.sleep(RETRY_DELAY * 2**attempt)
    return None


# ── Answer cache in the database (supabase/vision_cache.sql) ────────────────────────────────────
# Callers also keep their disk cache: it still works when the table is missing or Supabase is down.


async def cache_get(settings: Settings, db: httpx.AsyncClient | None, key: str) -> dict | None:
    if db is None or not settings.supabase_url:
        return None
    try:
        response = await db.get(f"{settings.supabase_url.rstrip('/')}/rest/v1/vision_cache?key=eq.{key}&select=result", headers=service_headers(settings), timeout=10)
        rows = response.json() if response.status_code < 400 else []
    except (httpx.HTTPError, HTTPException, ValueError) as exc:
        logger.warning("Vision cache unreadable: %s", exc)
        return None
    if response.status_code >= 400:
        logger.warning("Vision cache unreadable (%s). Run supabase/vision_cache.sql.", response.status_code)
    return rows[0]["result"] if rows else None


async def cache_put(settings: Settings, db: httpx.AsyncClient | None, key: str, kind: str, model: str, result: dict) -> None:
    if db is None or not settings.supabase_url:
        return
    try:
        response = await db.post(
            f"{settings.supabase_url.rstrip('/')}/rest/v1/vision_cache",
            json={"key": key, "kind": kind, "model": model, "result": result},
            headers={**service_headers(settings), "Prefer": "resolution=merge-duplicates,return=minimal"}, timeout=10,
        )
        if response.status_code >= 400:
            logger.warning("Vision cache not saved (%s): %s", response.status_code, response.text[:200])
    except (httpx.HTTPError, HTTPException) as exc:
        logger.warning("Vision cache not saved: %s", exc)
