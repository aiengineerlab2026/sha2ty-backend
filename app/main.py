import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin import router as admin_router
from app.api.match import router as match_router
from app.api.plans import router as plans_router
from app.api.vendor import router as vendor_router
from app import handlers  # noqa: F401  subscribes the event handlers
from app.core.config import get_settings
from app.schemas.vendor import HealthResponse
from app.services.astra_service import AstraService
from app.services.cdn import Cdn
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_origin_regex=get_settings().cors_origin_regex,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)

app.include_router(vendor_router)
app.include_router(plans_router)
app.include_router(match_router)
app.include_router(admin_router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        meshy_configured=settings.meshy_configured,
        astra_configured=settings.astra_configured,
        supabase_configured=bool(settings.supabase_url and settings.supabase_anon_key.get_secret_value()),
    )
