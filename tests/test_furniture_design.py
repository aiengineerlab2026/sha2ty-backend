import asyncio
import json
import random
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.services import furniture_design as fd


def answer(spec):
    async def fake(content, schema, name, settings, client, model):
        assert model == settings.openai_sol_model and "Headboard Upholstery" in content[0]["text"]
        return json.dumps(spec)
    return fake


BED = {"title": "سرير مخملي فاخر", "description": "سرير بلون زمردي.", "style": "art deco", "width_m": 9, "depth_m": 2.1, "height_m": 1.2,
       "looks": {"Headboard": {"color": "#114433", "material": "velvet"}}}


def test_a_design_is_kept_inside_the_type_sizes_and_every_part_has_a_look(monkeypatch):
    monkeypatch.setattr(fd.vision, "ask_json", answer(BED))
    spec = asyncio.run(fd.design("bed", Settings(openai_api_key=SecretStr("k")), None, random.Random(1)))
    assert spec.width_m == 2.0 and spec.kind == "bed"
    assert list(spec.looks) == fd.PARTS["bed"] and spec.looks["Mattress"].material == "velvet"


def test_a_design_without_an_arabic_name_is_refused(monkeypatch):
    monkeypatch.setattr(fd.vision, "ask_json", answer({**BED, "title": "Velvet bed"}))
    with pytest.raises(ValueError):
        asyncio.run(fd.design("bed", Settings(openai_api_key=SecretStr("k")), None))


def test_only_the_platform_account_generates(monkeypatch):
    from fastapi import HTTPException
    from app.api import vendor

    async def not_platform(*_a):
        return False
    monkeypatch.setattr(vendor, "is_platform", not_platform)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(auth_http=None)))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(vendor.platform_user(request, "someone"))
    assert caught.value.status_code == 403


def test_photos_name_the_real_materials_and_one_shot_can_be_made_again(monkeypatch):
    seen = []

    async def edit(self, prompt, refs, size=None, quality=None):
        seen.append((prompt, size, quality))
        return b"png"
    monkeypatch.setattr(fd.ImageMatcher, "edit", edit)
    looks = fd.finishes({"Legs": {"color": "#222222", "material": "brass"}})
    images, errors = asyncio.run(fd.photos("bed", "سرير", b"r", Settings(), None, "art deco", looks))
    assert sorted(images) == list(range(len(fd.SHOTS))) and not errors
    assert all(q == "high" for _, _, q in seen) and "brushed brass in #222222" in seen[0][0] and "bedroom" in seen[2][0]
    seen.clear()
    images, _ = asyncio.run(fd.photos("bed", "سرير", b"r", Settings(), None, which=2))
    assert list(images) == [2] and seen[0][1] == "1536x1024"
