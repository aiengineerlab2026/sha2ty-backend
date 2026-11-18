import asyncio, io
from PIL import Image
from app.core.config import Settings
from app.services import report_photos


def _png(w, h, color):
    out = io.BytesIO(); Image.new("RGB", (w, h), color).save(out, "PNG"); return out.getvalue()


def test_thumbnail_and_parts_become_photos(tmp_path, monkeypatch):
    calls = []

    async def edit(self, prompt, refs, size=None, quality=None):
        calls.append(size)
        return _png(1024, 1024, "red")  # the "photo"

    monkeypatch.setattr(report_photos.ImageMatcher, "edit", edit)
    s = Settings(storage_dir=tmp_path)
    sheet = _png(128, 64, "white")  # 2x1 cells of 64
    meta = {"parts": [{"pic": 1}], "pic_size": 64, "pic_cols": 2}
    thumb, out = asyncio.run(report_photos.realistic(_png(300, 200, "gray"), sheet, meta, s, None))
    assert Image.open(io.BytesIO(thumb)).size == (300, 200)
    assert Image.open(io.BytesIO(thumb)).getpixel((5, 5))[0] > 200  # red photo, original size
    img = Image.open(io.BytesIO(out))
    assert img.getpixel((10, 10)) == (255, 255, 255) and img.getpixel((80, 10))[0] > 200  # only used cell replaced
    assert sorted(calls) == ["1024x1024", "1536x1024"]
    asyncio.run(report_photos.realistic(_png(300, 200, "gray"), None, {}, s, None))
    assert len(calls) == 2  # cached


def test_bedding_is_not_in_the_parts_list():
    from app.services import request_report as rr
    rows = [{"name": n, "main": True, "quantity": 1, "width": 10, "height": 10, "depth": 10} for n in ("Mattress", "Bed_blanket_1", "Left pillow", "Headboard")]
    rows.append({"name": "x", "name_ar": "مخدة 2", "main": True, "quantity": 1})
    assert [r["label"] for r in rr._components({"parts": rows}, "en")] == ["Headboard"]
