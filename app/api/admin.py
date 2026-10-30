"""Admin console: one configured account (ADMIN_USERNAME / ADMIN_PASSWORD) approves vendor signups.

The admin session is a signed, expiring token (no database row). The browser keeps it in an httpOnly
cookie set by the Next.js app, which forwards it as the X-Admin-Token header.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Annotated, Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.services.auth import service_headers

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _secret(settings: Settings) -> bytes:
    key = settings.admin_session_secret.get_secret_value() or settings.supabase_service_role_key.get_secret_value()
    if not key:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ADMIN_SESSION_SECRET is not configured on the server.")
    return key.encode()


def _sign(expires: int, settings: Settings) -> str:
    return hmac.new(_secret(settings), f"admin.{expires}".encode(), hashlib.sha256).hexdigest()


def issue_token(settings: Settings, now: float | None = None) -> tuple[str, int]:
    expires = int((now or time.time()) + settings.admin_session_hours * 3600)
    return f"{expires}.{_sign(expires, settings)}", expires


def token_valid(token: str, settings: Settings, now: float | None = None) -> bool:
    expires, _, signature = token.partition(".")
    if not expires.isdigit() or int(expires) < (now or time.time()):
        return False
    return hmac.compare_digest(signature, _sign(int(expires), settings))


def require_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
    if not x_admin_token or not token_valid(x_admin_token, get_settings()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Admin sign-in required.")


class LoginBody(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(body: LoginBody) -> dict[str, object]:
    settings = get_settings()
    ok_user = hmac.compare_digest(body.username.strip().encode(), settings.admin_username.encode())
    ok_pass = hmac.compare_digest(body.password.encode(), settings.admin_password.get_secret_value().encode())
    if not (ok_user and ok_pass):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username or password.")
    token, expires = issue_token(settings)
    return {"token": token, "expires_at": expires}


def _rest(settings: Settings) -> str:
    return f"{settings.supabase_url.rstrip('/')}/rest/v1/vendors"


@router.get("/vendors", dependencies=[Depends(require_admin)])
async def list_vendors(request: Request) -> dict[str, object]:
    settings = get_settings()
    client: httpx.AsyncClient = request.app.state.auth_http
    response = await client.get(f"{_rest(settings)}?select=*&order=created_at.desc", headers=service_headers(settings))
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not load vendors. Run supabase/vendor_approval.sql.")
    return {"vendors": response.json()}


class StatusBody(BaseModel):
    status: Literal["approved", "rejected", "pending"]


@router.post("/vendors/{vendor_id}/status", dependencies=[Depends(require_admin)])
async def set_status(vendor_id: UUID, body: StatusBody, request: Request) -> dict[str, object]:
    settings = get_settings()
    client: httpx.AsyncClient = request.app.state.auth_http
    response = await client.patch(
        f"{_rest(settings)}?id=eq.{vendor_id}",
        headers={**service_headers(settings), "Prefer": "return=representation"},
        json={"status": body.status},
    )
    rows = response.json() if response.status_code < 400 else None
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND if response.status_code < 400 else status.HTTP_502_BAD_GATEWAY, "Vendor not found.")
    return {"vendor": rows[0]}
