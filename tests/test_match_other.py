import asyncio
from types import SimpleNamespace

from app.api import match


class FakeClient:
    def __init__(self, row):
        self.row = row

    async def get(self, url, headers=None, params=None):
        return SimpleNamespace(status_code=200, json=lambda: [self.row])


def request_with(row):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(auth_http=FakeClient(row))))


def test_other_item_reads_size_and_photo_from_catalogue(monkeypatch):
    monkeypatch.setattr(match, "service_headers", lambda _s: {})
    carpet = asyncio.run(match.other_item(request_with({"image_url": "https://cdn/c.png", "width_m": 2, "length_m": 3}), "carpet", "id"))
    assert carpet == {"url": "https://cdn/c.png", "size": (200.0, 300.0, None)}

    texture = asyncio.run(match.other_item(request_with({"kind": "texture", "texture_url": "https://cdn/t.png", "tile_width_m": 0.6}), "wall", "id"))
    assert texture["url"] == "https://cdn/t.png" and texture["tile"] == 60.0

    colour = asyncio.run(match.other_item(request_with({"kind": "color", "color": "#aabbcc"}), "wall", "id"))
    assert colour["color"] == "#aabbcc" and "url" not in colour
