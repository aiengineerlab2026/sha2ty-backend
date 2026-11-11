"""Admin console: one configured account (ADMIN_USERNAME / ADMIN_PASSWORD) approves vendor signups.

The admin session is a signed, expiring token (no database row). The browser keeps it in an httpOnly
cookie set by the Next.js app, which forwards it as the X-Admin-Token header.
"""

from __future__ import annotations

import hashlib
import logging
import hmac
import time
from typing import Annotated, Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from app.api.proposals import _spec as proposal_spec  # a proposal as the customer and vendor see it
from app.core.config import Settings, get_settings
from app.core.events import bus
from app.domain.events import VendorStatusChanged
from app.domain.vendor import VendorStatus
from app.services import generator_setting, vision
from app.services.auth import service_headers
from app.services.platform import ensure_platform_vendor, is_platform, platform_session

logger = logging.getLogger(__name__)
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
    ok_user = hmac.compare_digest(body.username.strip().lower().encode(), settings.admin_username.strip().lower().encode())  # emails ignore case
    ok_pass = hmac.compare_digest(body.password.encode(), settings.admin_password.get_secret_value().encode())
    if not (ok_user and ok_pass):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong email or password.")
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


async def _users(settings, client: httpx.AsyncClient) -> dict[str, dict[str, str]]:
    """id → name and email of every account (Supabase Auth admin API, service key; up to 1000 per page)."""
    out: dict[str, dict[str, str]] = {}
    for page in range(1, 21):  # ponytail: 20 000 accounts; a search endpoint when there are more
        response = await client.get(f"{settings.supabase_url.rstrip('/')}/auth/v1/admin/users", params={"page": page, "per_page": 1000}, headers=service_headers(settings))
        users = response.json().get("users", []) if response.status_code < 400 else []
        for u in users:
            meta = u.get("user_metadata") or {}
            out[u["id"]] = {"name": (meta.get("full_name") or meta.get("name") or "").strip(), "email": u.get("email") or ""}
        if len(users) < 1000:
            break
    return out


@router.get("/models", dependencies=[Depends(require_admin)])
async def list_models(request: Request) -> dict[str, object]:
    """Every model on the platform with its owner: vendors' listings, Sha2ty's own catalogue, and customers' private
    models made from their photos (user_models; never shown to anyone else)."""
    settings = get_settings()
    client: httpx.AsyncClient = request.app.state.auth_http
    base = settings.supabase_url.rstrip("/") + "/rest/v1"
    # Columns both tables have (models has no thumbnail_url: a listing's picture is its source image).
    fields = "id,title,status,category,width_m,height_m,depth_m,source_image_url,glb_url,created_at"
    listings = await client.get(f"{base}/models?select={fields},vendor_id,vendors(name,email,platform,status)&order=created_at.desc", headers=service_headers(settings))
    if listings.status_code >= 400:
        logger.warning("Admin models list failed: %s %s", listings.status_code, listings.text[:300])
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not load the models.")
    private = await client.get(f"{base}/user_models?select=*&order=created_at.desc", headers=service_headers(settings))
    users = await _users(settings, client)
    rows: list[dict[str, object]] = []
    for m in listings.json():
        vendor = m.pop("vendors", None) or {}
        account = users.get(str(m.get("vendor_id")), {})
        kind = "platform" if vendor.get("platform") else "vendor"
        rows.append({**m, "thumbnail_url": None, "kind": kind, "owner": {"id": m.get("vendor_id"), "name": vendor.get("name") or account.get("name") or "", "email": vendor.get("email") or account.get("email") or "", "status": vendor.get("status")}})
    for m in private.json() if private.status_code < 400 else []:  # before user_models.sql: none
        account = users.get(str(m.get("owner_id")), {})
        rows.append({**{k: m.get(k) for k in fields.split(",")}, "thumbnail_url": m.get("thumbnail_url"), "kind": "customer", "owner": {"id": m.get("owner_id"), "name": account.get("name", ""), "email": account.get("email", ""), "status": None}})
    rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return {"models": rows}


@router.get("/tasks", dependencies=[Depends(require_admin)])
async def list_tasks(request: Request) -> dict[str, object]:
    """The newest implementation requests with their customer and number of offers (each has a PDF report)."""
    settings = get_settings()
    client: httpx.AsyncClient = request.app.state.auth_http
    base = settings.supabase_url.rstrip("/") + "/rest/v1"
    response = await client.get(
        f"{base}/freelance_tasks?select=id,title,status,audience,budget_max,created_at,customer_id,task_bids(count)&order=created_at.desc&limit=100",
        headers=service_headers(settings),
    )
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not load the requests. Run supabase/tasks.sql.")
    users = await _users(settings, client)
    return {"tasks": [
        {**{k: v for k, v in t.items() if k != "task_bids"}, "offers": ((t.get("task_bids") or [{}])[0] or {}).get("count", 0),
         "customer": users.get(str(t.get("customer_id")), {}).get("name") or users.get(str(t.get("customer_id")), {}).get("email") or ""}
        for t in response.json()
    ]}


async def _user(settings, client: httpx.AsyncClient, user_id: str | None) -> dict[str, str]:
    """One account's name, email and phone (Supabase Auth admin API, service key); empty when unknown."""
    if not user_id:
        return {}
    response = await client.get(f"{settings.supabase_url.rstrip('/')}/auth/v1/admin/users/{user_id}", headers=service_headers(settings))
    if response.status_code >= 400:
        return {}
    u = response.json()
    meta = u.get("user_metadata") or {}
    return {"id": user_id, "name": (meta.get("full_name") or meta.get("name") or "").strip(), "email": u.get("email") or "", "phone": u.get("phone") or meta.get("phone") or ""}


@router.get("/tasks/{task_id}", dependencies=[Depends(require_admin)])
async def task_detail(task_id: UUID, request: Request) -> dict[str, object]:
    """One request seen from both sides: what the customer asked for and who they are, and for every vendor who
    took part their offer, proposed changes to the model and their conversation with the customer."""
    settings = get_settings()
    client: httpx.AsyncClient = request.app.state.auth_http
    base = settings.supabase_url.rstrip("/") + "/rest/v1"

    async def rows(path: str, required: bool = False) -> list[dict]:
        response = await client.get(f"{base}/{path}", headers=service_headers(settings))
        if response.status_code >= 400:
            if required:
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not load the request. Run supabase/tasks.sql.")
            return []  # proposals.sql / marketplace.sql not applied yet: that side is simply empty
        return response.json()

    task = (await rows(f"freelance_tasks?id=eq.{task_id}&select=*", required=True) or [None])[0]
    if not task:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found.")
    bids = await rows(f"task_bids?task_id=eq.{task_id}&select=*&order=created_at.asc", required=True)
    proposals = await rows(f"task_proposals?task_id=eq.{task_id}&select=*&order=created_at.desc")
    messages = await rows(f"task_messages?task_id=eq.{task_id}&select=id,vendor_id,sender_id,body,attachment_url,created_at,read_at&order=created_at.asc")
    ids = list(dict.fromkeys(str(x["vendor_id"]) for x in [*bids, *proposals, *messages]))
    if task.get("exclusive_vendor_id") and str(task["exclusive_vendor_id"]) not in ids:
        ids.append(str(task["exclusive_vendor_id"]))  # the only vendor who may bid, even before they did
    vendors = {str(v["id"]): v for v in (await rows(f"vendors?id=in.({','.join(ids)})&select=*") if ids else [])}
    customer = await _user(settings, client, task.get("customer_id"))
    awarded = task.get("awarded_bid_id")
    threads = []
    for vid in ids:
        v = vendors.get(vid, {})
        account = await _user(settings, client, vid)
        bid = next((b for b in bids if str(b["vendor_id"]) == vid), None)
        threads.append({
            "vendor": {"id": vid, "name": v.get("name") or account.get("name") or "", "logo_url": v.get("logo_url"), "location": v.get("location") or "",
                       "status": v.get("status"), "email": account.get("email", ""), "phone": account.get("phone", "")},
            "bid": bid,
            "hired": bool(bid and awarded and str(bid["id"]) == str(awarded)),
            "proposals": [{**p, "spec": proposal_spec(task, p)} for p in proposals if str(p["vendor_id"]) == vid],
            # Who wrote each message: the vendor, or the customer answering them.
            "messages": [{**m, "from": "vendor" if str(m["sender_id"]) == vid else "customer"} for m in messages if str(m["vendor_id"]) == vid],
        })
    # The hired vendor first, then those who offered, then the rest (questions or proposals only).
    threads.sort(key=lambda th: (not th["hired"], th["bid"] is None))
    return {"task": task, "customer": customer, "vendors": threads}


class StatusBody(BaseModel):
    status: VendorStatus


@router.post("/vendors/{vendor_id}/status", dependencies=[Depends(require_admin)])
async def set_status(vendor_id: UUID, body: StatusBody, request: Request) -> dict[str, object]:
    settings = get_settings()
    client: httpx.AsyncClient = request.app.state.auth_http
    if body.status != "approved" and await is_platform(str(vendor_id), settings, client):
        raise HTTPException(status.HTTP_409_CONFLICT, "The Sha2ty catalogue vendor stays approved.")
    response = await client.patch(
        f"{_rest(settings)}?id=eq.{vendor_id}",
        headers={**service_headers(settings), "Prefer": "return=representation"},
        json={"status": body.status},
    )
    rows = response.json() if response.status_code < 400 else None
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND if response.status_code < 400 else status.HTTP_502_BAD_GATEWAY, "Vendor not found.")
    await bus.publish(VendorStatusChanged(str(vendor_id), body.status), request.app.state)
    return {"vendor": rows[0]}


@router.post("/studio-session", dependencies=[Depends(require_admin)])
async def studio_session(request: Request) -> dict[str, object]:
    """Sign the admin's browser in as the Sha2ty vendor, to add and edit the platform's own catalogue models."""
    return await platform_session(get_settings(), request.app.state.auth_http, request.app.state.cdn)


@router.get("/studio", dependencies=[Depends(require_admin)])
async def studio(request: Request) -> dict[str, object]:
    """The Sha2ty vendor (created on first use) and how many models it lists."""
    settings = get_settings()
    vendor = await ensure_platform_vendor(settings, request.app.state.auth_http, request.app.state.cdn)
    response = await request.app.state.auth_http.get(
        f"{settings.supabase_url.rstrip('/')}/rest/v1/models?vendor_id=eq.{vendor['id']}&select=id,status",
        headers=service_headers(settings),
    )
    models = response.json() if response.status_code < 400 else []
    return {"vendor": {k: vendor.get(k) for k in ("id", "name", "logo_url", "status")}, "models": len(models), "ready": sum(m.get("status") == "ready" for m in models)}


class VisionBody(BaseModel):
    model: str


class GeneratorBody(BaseModel):
    provider: Literal["meshy", "astra", "sol"]


@router.get("/generator", dependencies=[Depends(require_admin)])
async def get_generator(request: Request) -> dict[str, object]:
    """The service that turns vendors' photos into 3D models (vendors no longer choose), and which have a key."""
    settings = get_settings()
    return {
        "provider": await generator_setting.current(settings, request.app.state.auth_http),
        "options": [{"id": p, "label": label, "configured": generator_setting.configured(settings, p)} for p, label in generator_setting.LABELS.items()],
    }


@router.put("/generator", dependencies=[Depends(require_admin)])
async def put_generator(body: GeneratorBody, request: Request) -> dict[str, object]:
    settings = get_settings()
    if not generator_setting.configured(settings, body.provider):
        key = "MESHY_API_KEY" if body.provider == "meshy" else "OPENAI_API_KEY"
        raise HTTPException(status.HTTP_409_CONFLICT, f"{generator_setting.LABELS[body.provider]} needs {key} on the server.")
    try:
        provider = await generator_setting.set_current(settings, request.app.state.auth_http, body.provider)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return {"provider": provider}


class MatchModelBody(BaseModel):
    provider: Literal["astra", "sol"]


@router.get("/match-model", dependencies=[Depends(require_admin)])
async def get_match_model(request: Request) -> dict[str, object]:
    """The GPT that combines the model's photo and the other item's photo into one room (Match)."""
    settings = get_settings()
    key = generator_setting.MATCH
    return {
        "provider": await generator_setting.current(settings, request.app.state.auth_http, key),
        "options": [{"id": p, "label": label, "configured": generator_setting.configured(settings, p)} for p, label in generator_setting.MATCH_LABELS.items()],
    }


@router.put("/match-model", dependencies=[Depends(require_admin)])
async def put_match_model(body: MatchModelBody, request: Request) -> dict[str, object]:
    settings = get_settings()
    if not generator_setting.configured(settings, body.provider):
        raise HTTPException(status.HTTP_409_CONFLICT, f"{generator_setting.MATCH_LABELS[body.provider]} needs OPENAI_API_KEY on the server.")
    try:
        provider = await generator_setting.set_current(settings, request.app.state.auth_http, body.provider, generator_setting.MATCH)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return {"provider": provider}


@router.get("/vision", dependencies=[Depends(require_admin)])
async def get_vision(request: Request) -> dict[str, object]:
    """The model that reads floor plans and vendor product photos, and which ones have an API key."""
    settings = get_settings()
    return {
        "model": await vision.current_model(settings, request.app.state.auth_http),
        "options": [{"id": m, "label": label, "configured": vision.configured(settings, m)} for m, label in vision.MODELS.items()],
    }


@router.put("/vision", dependencies=[Depends(require_admin)])
async def put_vision(body: VisionBody, request: Request) -> dict[str, object]:
    settings = get_settings()
    if body.model not in vision.MODELS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Choose one of: {', '.join(vision.MODELS)}.")
    if not vision.configured(settings, body.model):
        key = "GEMINI_API_KEY" if vision.is_gemini(body.model) else "OPENAI_API_KEY"
        raise HTTPException(status.HTTP_409_CONFLICT, f"{vision.MODELS[body.model]} needs {key} on the server.")
    try:
        model = await vision.set_model(settings, request.app.state.auth_http, body.model)
    except vision.VisionSettingError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return {"model": model}
