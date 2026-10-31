"""Domain events: facts that already happened. Handlers (app/handlers.py) react to them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.domain.generation import Provider
from app.domain.vendor import Vendor, VendorStatus


@dataclass(frozen=True)
class UserRegistered:
    user_id: str
    kind: Literal["vendor", "customer"]
    vendor: Vendor | None = None


@dataclass(frozen=True)
class VendorStatusChanged:
    vendor_id: str
    status: VendorStatus


@dataclass(frozen=True)
class GenerationStarted:
    vendor_id: str
    provider: Provider
    remaining: int


@dataclass(frozen=True)
class GenerationFailed:
    vendor_id: str
    provider: Provider
