import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin import router as admin_router
from app.api.admin_paint_colors import router as admin_paint_colors_router
from app.api.admin_woods import router as admin_woods_router
from app.api.designs import router as designs_router
from app.api.files import router as files_router
from app.api.marketplace import router as marketplace_router
from app.api.proposals import router as proposals_router
from app.api.realistic import router as realistic_router
from app.api.reports import router as reports_router
from app.api.match import router as match_router
from app.api.plans import router as plans_router
from app.api.sb_proxy import router as sb_router
from app.api.session import SessionCookies, router as session_router
from app.api.vendor import router as vendor_router
from app.services.cdn import CdnError
from app import handlers  # noqa: F401  subscribes the event handlers
from app.core.config import get_settings
from app.schemas.vendor import HealthResponse
from app.services.astra_service import AstraService
from app.services.cdn import REQUEST_ORIGIN, Cdn
from app.services.meshy_service import MeshyService
from app.services.storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if not settings.meshy_configured:
        logger.warning("MESHY_API_KEY is not set; Meshy generation will return 503.")
    if not settings.astra_configured:
        logger.warning("OPENAI_API_KEY is not set; Astra generation will return 503.")

    storage = Storage(settings.storage_dir)
    storage.init()
    api_client, download_client = MeshyService.create_clients(settings)
    openai_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.meshy_download_timeout_seconds),
        headers={"Authorization": f"Bearer {settings.openai_api_key.get_secret_value()}"},
    )
    auth_http = httpx.AsyncClient(timeout=20)
    app.state.storage = storage
    app.state.meshy = MeshyService(settings, api_client, download_client)
    app.state.astra = AstraService(settings, openai_client)
    app.state.openai_http = openai_client  # also used by Match (image edits)
    app.state.cdn = Cdn(settings)
    app.state.auth_http = auth_http
    app.state.pending_cdn = {}
    try:
        yield
    finally:
        await api_client.aclose()
        await download_client.aclose()
        await openai_client.aclose()
        await auth_http.aclose()


app = FastAPI(title="Shatably Vendor API", version="2.0.0", lifespan=lifespan)


@app.exception_handler(CdnError)
async def storage_failed(_request: Request, exc: CdnError) -> JSONResponse:
    """A failed upload or download is a real error the person must see (never a silent success): its own status and
    message, wherever it happens (a rebuild, a repair, a generation)."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

app.add_middleware(
    CORSMiddleware,
    # ponytail: every origin for now (".*" echoes the caller's origin: a literal "*" is refused by browsers when cookies
    # are sent). Back to settings.cors_origins / cors_origin_regex before launch.
    allow_origin_regex=".*",
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
    allow_headers=["*"],
    allow_credentials=True,  # the web app calls this API directly with its session cookies
    expose_headers=["Retry-After"],
)

@app.middleware("http")
async def public_origin(request, call_next):
    """The origin this request reached, for the file URLs built while serving it (services/cdn.api_origin)."""
    host = request.headers.get("host")
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme).split(",")[0].strip()
    token = REQUEST_ORIGIN.set(f"{'https' if proto == 'https' else 'http'}://{host}" if host else None)
    try:
        return await call_next(request)
    finally:
        REQUEST_ORIGIN.reset(token)


# Cookie sessions → bearer token for every route (the browser's only way in: frontend next.config rewrites /api/*).
app.add_middleware(SessionCookies)

app.include_router(session_router)
app.include_router(sb_router)
app.include_router(vendor_router)
app.include_router(plans_router)
app.include_router(match_router)
app.include_router(admin_router)
app.include_router(admin_woods_router)
app.include_router(admin_paint_colors_router)
app.include_router(designs_router)
app.include_router(files_router)
app.include_router(marketplace_router)
app.include_router(proposals_router)
app.include_router(realistic_router)
app.include_router(reports_router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        meshy_configured=settings.meshy_configured,
        astra_configured=settings.astra_configured,
        supabase_configured=bool(settings.supabase_url and settings.supabase_anon_key.get_secret_value()),
    )
