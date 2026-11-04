import asyncio

import pytest
from fastapi import HTTPException

from app.api import vendor
from app.core.config import Settings


def test_take_try_returns_remaining_and_blocks_at_limit(monkeypatch):
    used = {"n": 0}

    async def fake_use(_vendor, _provider, delta, settings, _client):
        if used["n"] + delta > settings.generation_limit:
            return None
        used["n"] += delta
        return used["n"]

    monkeypatch.setattr(vendor, "use_generation", fake_use)

    async def not_platform(*_args):
        return False

    monkeypatch.setattr(vendor, "is_platform", not_platform)
    settings = Settings(generation_limit=2)
    request = type("R", (), {"app": type("A", (), {"state": type("S", (), {"auth_http": None})()})()})()

    assert asyncio.run(vendor.take_try(request, "v1", "meshy", settings)) == 1
    assert asyncio.run(vendor.take_try(request, "v1", "meshy", settings)) == 0
    with pytest.raises(HTTPException) as exc:
        asyncio.run(vendor.take_try(request, "v1", "meshy", settings))
    assert exc.value.status_code == 429
    assert "2 Meshy" in exc.value.detail


def test_the_platform_catalogue_has_no_generation_limit(monkeypatch):
    async def fail_use(*_args):
        raise AssertionError("the platform vendor's tries are not counted")

    async def platform(*_args):
        return True

    monkeypatch.setattr(vendor, "use_generation", fail_use)
    monkeypatch.setattr(vendor, "is_platform", platform)
    request = type("R", (), {"app": type("A", (), {"state": type("S", (), {"auth_http": None})()})()})()
    for _ in range(5):
        assert asyncio.run(vendor.take_try(request, "sha2ty", "meshy", Settings(generation_limit=2))) == 2
