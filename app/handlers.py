"""Application event handlers: side effects of domain events. Imported once by app.main to subscribe."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings
from app.core.events import bus
from app.domain.events import GenerationFailed, GenerationStarted, UserRegistered, VendorStatusChanged
from app.services.auth import create_vendor
from app.services.quota import use_generation

logger = logging.getLogger("app.events")


@bus.subscribe(UserRegistered, VendorStatusChanged, GenerationStarted, GenerationFailed)
async def audit(event: object, _state: Any) -> None:
    logger.info("%s %s", type(event).__name__, event)


@bus.subscribe(UserRegistered)
async def create_vendor_profile(event: UserRegistered, state: Any) -> None:
    # Service role writes the row, so vendors cannot set their own status (supabase/vendor_approval.sql).
    if event.vendor:
        await create_vendor(event.vendor.row(), get_settings(), state.auth_http)


@bus.subscribe(GenerationFailed)
async def refund_try(event: GenerationFailed, state: Any) -> None:
    # ponytail: best-effort refund, not transactional with the failed call.
    await use_generation(event.vendor_id, event.provider, -1, get_settings(), state.auth_http)
