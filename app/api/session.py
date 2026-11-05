"""Browser sessions (moved here from the Next.js /api/auth route: the frontend has no API code).

The Supabase access JWT and refresh token live in httpOnly cookies that JavaScript never sees. `SessionCookies`
(a middleware) turns them into the `Authorization: Bearer` header every route already reads, refreshing the access
token a minute before it expires and writing the rotated cookies back; the admin cookie becomes `X-Admin-Token`.
The browser reaches this API on its own origin (frontend next.config rewrites /api/* here), so the cookies are
first-party and SameSite=Lax.

Routes (/api/auth/…): POST login, logout, refresh, recover, session (reset / app links), password,
admin-login, admin-logout, admin-studio; GET me, realtime, admin-me.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import Settings, get_settings

ACCESS_COOKIE = "sha2ty-access"
REFRESH_COOKIE = "sha2ty-refresh"
ADMIN_COOKIE = "sha2ty-admin"
REFRESH_MAX_AGE = 60 * 60 * 24 * 30  # Supabase rotates the refresh token on every use
SKEW = 60  # refresh a minute early

router = APIRouter(prefix="/api/auth", tags=["auth"])


def jwt_expiry(jwt: str) -> int:
    """Seconds since epoch when a JWT expires (0 when unreadable); the signature is Supabase's to check."""
    try:
        part = jwt.split(".")[1]
        return int(json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))).get("exp", 0))
    except (IndexError, ValueError, TypeError, AttributeError):
        return 0


def _site(host: str) -> str:
    """The registrable part of a host, roughly (its last two labels): sha2ty.net for www.sha2ty.net."""
    # ponytail: two labels; a two-part suffix (co.uk, com.eg) needs the public suffix list.
    return ".".join(host.split(":")[0].lower().split(".")[-2:])


def cookie_args(request: Request) -> dict[str, Any]:
    """httpOnly session cookie attributes. The site calling from another site (www.sha2ty.net → *.run.app) only gets
    its cookies sent with SameSite=None, which needs Secure; Partitioned keeps them (per site) where third-party cookies
    are blocked. Same site (local development): SameSite=Lax."""
    settings = get_settings()
    origin = request.headers.get("origin") or ""
    cross = bool(origin) and _site(origin.split("://", 1)[-1]) != _site(request.headers.get("host", request.url.netloc))
    https = settings.cookie_secure or request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    if cross and https:
        return {"httponly": True, "secure": True, "samesite": "none", "partitioned": True, "path": "/"}
    return {"httponly": True, "secure": https, "samesite": "lax", "path": "/"}


def set_tokens(response: Response, request: Request, tokens: dict[str, Any]) -> None:
    args = cookie_args(request)
    response.set_cookie(ACCESS_COOKIE, tokens["access_token"], max_age=int(tokens.get("expires_in") or 3600), **args)
    response.set_cookie(REFRESH_COOKIE, tokens["refresh_token"], max_age=REFRESH_MAX_AGE, **args)


def clear_tokens(response: Response, request: Request, names: tuple[str, ...] = (ACCESS_COOKIE, REFRESH_COOKIE)) -> None:
    """Deleted with the attributes they were set with (a partitioned cookie is only removed by a partitioned delete)."""
    for name in names:
        response.delete_cookie(name, **cookie_args(request))


async def auth_api(client: httpx.AsyncClient, path: str, method: str = "GET", token: str | None = None, body: dict | None = None) -> httpx.Response:
    """A Supabase Auth REST call with the project key (and the user's token when given)."""
    s = get_settings()
    headers = {"apikey": s.supabase_anon_key.get_secret_value(), "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return await client.request(method, f"{s.supabase_url.rstrip('/')}/auth/v1{path}", headers=headers, json=body)


def _error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = {}
    return str(body.get("error_description") or body.get("msg") or body.get("message") or body.get("error") or f"Request failed ({response.status_code}).")


def fail(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


class SessionCookies(BaseHTTPMiddleware):
    """Cookie session → bearer token (and admin cookie → X-Admin-Token) for every API request. A client's own
    Authorization / X-Admin-Token header (the mobile app, tests) is left as it is."""

    async def dispatch(self, request: Request, call_next):
        headers = list(request.scope["headers"])
        admin = request.cookies.get(ADMIN_COOKIE)
        if admin and not any(k == b"x-admin-token" for k, _ in headers):
            headers.append((b"x-admin-token", admin.encode()))
        # supabase-js always sends the public anon key as its bearer: that is no session, so the cookie's token is used.
        anon = f"Bearer {get_settings().supabase_anon_key.get_secret_value()}".encode()
        headers = [(k, v) for k, v in headers if not (k == b"authorization" and v == anon)]
        refreshed: dict[str, Any] | None = None
        expired = False
        # Signing out must not get fresh cookies written over the ones it clears.
        signing_out = request.url.path in ("/api/auth/logout", "/api/auth/admin-logout")
        if not signing_out and not any(k == b"authorization" for k, _ in headers):
            access = request.cookies.get(ACCESS_COOKIE)
            if not access or jwt_expiry(access) <= time.time() + SKEW:
                access = None
                refresh_token = request.cookies.get(REFRESH_COOKIE)
                if refresh_token:
                    try:
                        res = await auth_api(request.app.state.auth_http, "/token?grant_type=refresh_token", "POST", body={"refresh_token": refresh_token})
                    except httpx.HTTPError:
                        res = None
                    if res is not None and res.status_code < 400:
                        refreshed = res.json()
                        access = refreshed["access_token"]
                    elif res is not None:
                        expired = True  # the refresh token was refused: signed out
            if access:
                headers.append((b"authorization", f"Bearer {access}".encode()))
        request.scope["headers"] = headers
        response = await call_next(request)
        if refreshed:
            set_tokens(response, request, refreshed)
        elif expired:
            clear_tokens(response, request)
        return response


def _bearer(request: Request) -> str | None:
    value = request.headers.get("authorization", "")
    return value.split(" ", 1)[1] if value.lower().startswith("bearer ") else None


@router.get("/me")
async def me(request: Request) -> Response:
    token = _bearer(request)
    if not token:
        return JSONResponse({"user": None})
    res = await auth_api(request.app.state.auth_http, "/user", token=token)
    if res.status_code >= 400:
        out = JSONResponse({"user": None})
        clear_tokens(out, request)
        return out
    return JSONResponse({"user": res.json()}, headers={"Cache-Control": "no-store"})


@router.get("/realtime")
async def realtime(request: Request) -> Response:
    """Live chat: the Realtime socket needs the user's own short-lived token (row-level security per subscriber)."""
    token = _bearer(request)
    return JSONResponse({"token": token}, status_code=200 if token else 401, headers={"Cache-Control": "no-store"})


@router.get("/admin-me")
async def admin_me(request: Request) -> Response:
    return JSONResponse({"admin": bool(request.cookies.get(ADMIN_COOKIE))})


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


@router.post("/login")
async def login(request: Request) -> Response:
    body = await _body(request)
    res = await auth_api(request.app.state.auth_http, "/token?grant_type=password", "POST", body={"email": body.get("email"), "password": body.get("password")})
    if res.status_code >= 400:
        return fail(_error(res), 401 if res.status_code == 400 else res.status_code)
    tokens = res.json()
    out = JSONResponse({"user": tokens.get("user")})
    set_tokens(out, request, tokens)
    return out


@router.post("/logout")
async def logout(request: Request) -> Response:
    token = _bearer(request)
    if token:
        try:
            await auth_api(request.app.state.auth_http, "/logout", "POST", token=token)
        except httpx.HTTPError:
            pass
    out = JSONResponse({"ok": True})
    clear_tokens(out, request, (ACCESS_COOKIE, REFRESH_COOKIE, ADMIN_COOKIE))  # signing out ends an admin's session too
    return out


@router.post("/refresh")
async def refresh(request: Request) -> Response:
    return JSONResponse({"ok": True}) if _bearer(request) else fail("Signed out.", 401)


@router.post("/recover")
async def recover(request: Request) -> Response:
    body = await _body(request)
    redirect = body.get("redirectTo")
    path = "/recover" + (f"?redirect_to={quote(redirect, safe='')}" if isinstance(redirect, str) else "")
    res = await auth_api(request.app.state.auth_http, path, "POST", body={"email": body.get("email")})
    return JSONResponse({"ok": True}) if res.status_code < 400 else fail(_error(res), res.status_code)


@router.post("/session")
async def take_session(request: Request) -> Response:
    """Password-reset links and the mobile app hand over tokens; they become cookies once Supabase accepts them."""
    body = await _body(request)
    access, refresh_token = body.get("access_token"), body.get("refresh_token")
    if not isinstance(access, str) or not isinstance(refresh_token, str):
        return fail("Invalid reset link.")
    res = await auth_api(request.app.state.auth_http, "/user", token=access)
    if res.status_code >= 400:
        return fail("This reset link has expired. Request a new one.", 401)
    out = JSONResponse({"user": res.json()})
    try:
        expires = int(body.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires = 3600
    set_tokens(out, request, {"access_token": access, "refresh_token": refresh_token, "expires_in": expires})
    return out


@router.post("/password")
async def password(request: Request) -> Response:
    token = _bearer(request)
    if not token:
        return fail("Sign in required.", 401)
    body = await _body(request)
    res = await auth_api(request.app.state.auth_http, "/user", "PUT", token=token, body={"password": body.get("password")})
    return JSONResponse({"ok": True}) if res.status_code < 400 else fail(_error(res), res.status_code)


@router.post("/admin-login")
async def admin_login(request: Request) -> Response:
    from app.api.admin import LoginBody, login as admin_token  # the admin console's own check

    body = await _body(request)
    try:
        issued = await admin_token(LoginBody(username=str(body.get("username") or ""), password=str(body.get("password") or "")))
    except Exception as exc:  # noqa: BLE001 - HTTPException carries the message and status
        return fail(str(getattr(exc, "detail", "Could not sign in.")), getattr(exc, "status_code", 401))
    out = JSONResponse({"ok": True})
    out.set_cookie(ADMIN_COOKIE, str(issued["token"]), max_age=max(60, int(issued["expires_at"]) - int(time.time())), **cookie_args(request))
    return out


@router.post("/admin-studio")
async def admin_studio(request: Request) -> Response:
    """Admin → the Sha2ty catalogue vendor: a session for that account in the usual cookies."""
    from app.api.admin import require_admin
    from app.services.platform import platform_session

    try:
        require_admin(request.cookies.get(ADMIN_COOKIE))
    except Exception:  # noqa: BLE001
        return fail("Admin sign-in required.", 401)
    data = await platform_session(get_settings(), request.app.state.auth_http, request.app.state.cdn)
    if not data.get("access_token"):
        return fail("Could not open the Sha2ty catalogue.", 502)
    out = JSONResponse({"vendor": data.get("vendor")})
    set_tokens(out, request, data)
    return out


@router.post("/admin-logout")
async def admin_logout(request: Request) -> Response:
    out = JSONResponse({"ok": True})
    clear_tokens(out, request, (ADMIN_COOKIE,))
    return out
