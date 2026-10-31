from __future__ import annotations

import httpx
from fastapi import Header, HTTPException, status

from app.core.config import Settings, get_settings


def service_headers(settings: Settings) -> dict[str, str]:
    key = settings.supabase_service_role_key.get_secret_value().strip()
    if not settings.supabase_url or not key:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "SUPABASE_SERVICE_ROLE_KEY is not configured on the server.",
        )
    return {"Authorization": f"Bearer {key}", "apikey": key, "Content-Type": "application/json"}


async def create_confirmed_user(
    email: str,
    password: str,
    metadata: dict[str, str],
    settings: Settings,
    client: httpx.AsyncClient,
) -> str:
    """Create an Auth user with email already confirmed so no mail is sent."""
    try:
        response = await client.post(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/admin/users",
            headers=service_headers(settings),
            json={
                "email": email,
                "password": password,
                "email_confirm": True,
                "user_metadata": metadata,
            },
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not create the account.") from exc

    body = response.json() if response.content else {}
    if response.status_code in {409, 422}:
        message = str(body.get("msg") or body.get("message") or "This email is already registered.")
        if "already" in message.lower() or "registered" in message.lower() or "exists" in message.lower():
            raise HTTPException(status.HTTP_409_CONFLICT, "This email is already registered. Log in instead.")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, message)
    if response.status_code >= 400:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            str(body.get("msg") or body.get("message") or "Could not create the account."),
        )
    user_id = body.get("id")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not create the account.")
    return user_id


async def create_vendor(row: dict[str, object], settings: Settings, client: httpx.AsyncClient) -> None:
    """Insert the vendors row, approved straight away. Without the approval migration, retry the base columns."""
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/vendors"
    try:
        response = await client.post(url, headers=service_headers(settings), json={**row, "status": "approved"})
        if response.status_code == 400 and "column" in response.text:
            response = await client.post(url, headers=service_headers(settings), json={k: row[k] for k in ("id", "name", "location")})
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not create the vendor profile.") from exc
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not create the vendor profile.")


class AuthError(Exception):
    pass


async def user_id_from_bearer(authorization: str | None, settings: Settings, client: httpx.AsyncClient) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in required.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in required.")
    if not settings.supabase_url or not settings.supabase_anon_key.get_secret_value():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "SUPABASE_ANON_KEY is not configured on the server.",
        )
    try:
        response = await client.get(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": settings.supabase_anon_key.get_secret_value(),
            },
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not verify the session.") from exc
    if response.status_code == 401:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired. Sign in again.")
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Could not verify the session.")
    user_id = response.json().get("id")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid session.")
    return user_id


async def require_user(
    authorization: str | None = Header(default=None),
) -> str:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=20) as client:
        return await user_id_from_bearer(authorization, settings, client)
