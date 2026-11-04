"""The platform's own vendor ("Sha2ty") and the admin's way into it (supabase/platform_vendor.sql).

Admins are not Supabase users, so they add catalogue models as the Sha2ty vendor: a system user created here
once (random password, never stored or shown). An admin-only route asks Supabase for a sign-in for that user
(an admin magic link, verified server-side), and the browser then uses the ordinary vendor pages — upload, photo
to 3D, editor — with row-level security keeping every model owned by Sha2ty.
"""

from __future__ import annotations

import secrets
from typing import Any

import httpx
from fastapi import HTTPException, status

from app.core.config import Settings
from app.services.auth import create_confirmed_user, service_headers
from app.services.cdn import Cdn, CdnError

NAME = "Sha2ty"
LOCATION = "Egypt"

# The brand mark (components/shatably/Logo.tsx LogoMark): house outline in navy, the amber hammer, on white.
LOGO_SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="256" height="256">
<rect width="100" height="100" rx="22" fill="#FFFFFF"/>
<path d="M16 50L50 20L84 50V86H16Z" fill="none" stroke="#172033" stroke-width="7" stroke-linejoin="round"/>
<g transform="rotate(-40 50 64)"><rect x="46" y="54" width="8" height="28" rx="4" fill="#172033"/>
<rect x="36" y="44" width="28" height="11" rx="2.5" fill="#F2A33A"/></g></svg>"""


def _rest(settings: Settings) -> str:
    return f"{settings.supabase_url.rstrip('/')}/rest/v1"


async def platform_vendor(settings: Settings, client: httpx.AsyncClient) -> dict[str, Any] | None:
    response = await client.get(f"{_rest(settings)}/vendors?platform=eq.true&select=*", headers=service_headers(settings))
    if response.status_code == 400 and "platform" in response.text:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Run backend/supabase/platform_vendor.sql first.")
    rows = response.json() if response.status_code < 400 else []
    return rows[0] if rows else None


async def is_platform(user_id: str, settings: Settings, client: httpx.AsyncClient) -> bool:
    try:
        response = await client.get(f"{_rest(settings)}/vendors?id=eq.{user_id}&platform=eq.true&select=id", headers=service_headers(settings))
    except httpx.HTTPError:
        return False
    return response.status_code < 400 and bool(response.json())


async def _magic_link(email: str, settings: Settings, client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.post(
        f"{settings.supabase_url.rstrip('/')}/auth/v1/admin/generate_link",
        headers=service_headers(settings),
        json={"type": "magiclink", "email": email},
    )
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not open the Sha2ty catalogue account.")
    return response.json()


async def ensure_platform_vendor(settings: Settings, client: httpx.AsyncClient, cdn: Cdn) -> dict[str, Any]:
    """The Sha2ty vendor row (approved, with its logo), created with its system user the first time."""
    vendor = await platform_vendor(settings, client)
    if vendor is None:
        email = settings.platform_vendor_email.strip().lower()
        try:
            await create_confirmed_user(email, secrets.token_urlsafe(32), {"role": "vendor", "name": NAME}, settings, client)
        except HTTPException as exc:
            if exc.status_code != status.HTTP_409_CONFLICT:  # already created by an earlier, interrupted run
                raise
        user_id = (await _magic_link(email, settings, client)).get("id")
        if not isinstance(user_id, str):
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not open the Sha2ty catalogue account.")
        row = {"id": user_id, "name": NAME, "location": LOCATION, "status": "approved", "platform": True}
        response = await client.post(
            f"{_rest(settings)}/vendors?on_conflict=id",
            headers={**service_headers(settings), "Prefer": "resolution=merge-duplicates,return=representation"},
            json=row,
        )
        if response.status_code >= 400:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not create the Sha2ty vendor.")
        vendor = response.json()[0]
    if not vendor.get("logo_url") or vendor.get("status") != "approved" or vendor.get("name") != NAME:
        fields: dict[str, Any] = {"name": NAME, "status": "approved"}
        if not vendor.get("logo_url"):
            try:
                fields["logo_url"] = cdn.upload(f"{vendor['id']}/brand/sha2ty-logo.svg", LOGO_SVG, "image/svg+xml")
            except CdnError:
                pass  # the vendor works without a logo; the next visit tries again
        response = await client.patch(
            f"{_rest(settings)}/vendors?id=eq.{vendor['id']}",
            headers={**service_headers(settings), "Prefer": "return=representation"},
            json=fields,
        )
        if response.status_code < 400 and response.json():
            vendor = response.json()[0]
    return vendor


async def platform_session(settings: Settings, client: httpx.AsyncClient, cdn: Cdn) -> dict[str, Any]:
    """A Supabase session for the Sha2ty user (access + refresh token), for an admin's browser."""
    vendor = await ensure_platform_vendor(settings, client, cdn)
    link = await _magic_link(settings.platform_vendor_email.strip().lower(), settings, client)
    token_hash = link.get("hashed_token") or (link.get("properties") or {}).get("hashed_token")
    if link.get("id") != vendor["id"] or not token_hash:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The Sha2ty catalogue account does not match its vendor.")
    response = await client.post(
        f"{settings.supabase_url.rstrip('/')}/auth/v1/verify",
        headers=service_headers(settings),
        json={"type": "magiclink", "token_hash": token_hash},
    )
    body = response.json() if response.content else {}
    if response.status_code >= 400 or not body.get("access_token"):
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not sign in to the Sha2ty catalogue account.")
    return {"vendor": vendor, "access_token": body["access_token"], "refresh_token": body["refresh_token"], "expires_in": body.get("expires_in", 3600)}
