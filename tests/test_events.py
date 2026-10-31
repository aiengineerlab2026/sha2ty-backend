import asyncio

import pytest

from app import handlers
from app.core.events import EventBus
from app.domain.errors import DomainError
from app.domain.events import GenerationFailed, UserRegistered
from app.domain.vendor import Vendor


def test_bus_runs_handlers_in_order_for_their_event_only():
    bus, seen = EventBus(), []

    @bus.subscribe(UserRegistered)
    async def first(event, _state):
        seen.append(("first", event.user_id))

    @bus.subscribe(UserRegistered, GenerationFailed)
    async def second(event, _state):
        seen.append(("second", type(event).__name__))

    asyncio.run(bus.publish(UserRegistered("u1", "customer"), None))
    asyncio.run(bus.publish(GenerationFailed("u1", "meshy"), None))
    assert seen == [("first", "u1"), ("second", "UserRegistered"), ("second", "GenerationFailed")]


def test_vendor_registers_approved_and_needs_name_and_location():
    assert Vendor.register("v1", " Studio ", " Cairo ").row()["status"] == "approved"
    with pytest.raises(DomainError):
        Vendor.register("v1", "", "Cairo")


def test_registered_vendor_creates_profile_and_failed_generation_refunds(monkeypatch):
    calls = []

    async def fake_create(row, _settings, _client):
        calls.append(("create", row["id"], row["status"]))

    async def fake_use(vendor, provider, delta, _settings, _client):
        calls.append(("use", vendor, provider, delta))

    monkeypatch.setattr(handlers, "create_vendor", fake_create)
    monkeypatch.setattr(handlers, "use_generation", fake_use)
    state = type("S", (), {"auth_http": None})()
    asyncio.run(handlers.bus.publish(UserRegistered("v1", "vendor", Vendor.register("v1", "S", "C")), state))
    asyncio.run(handlers.bus.publish(UserRegistered("c1", "customer"), state))
    asyncio.run(handlers.bus.publish(GenerationFailed("v1", "astra"), state))
    assert calls == [("create", "v1", "approved"), ("use", "v1", "astra", -1)]
