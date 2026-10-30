import asyncio
import base64
import io

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.main import app
from app.services import image_match as im

BED = im.Item("BedRoom", 180, 100, 200)
WARDROBE = im.Item("furniture", 120, 200, 60)


def png(w: int = 64, h: int = 48, color=(200, 30, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), color).save(out, "PNG")
    return out.getvalue()


def test_units_and_room_size_fit_both_items_with_clearance():
    assert im.to_cm(2.4, "m") == pytest.approx(240)
    with pytest.raises(im.ImageMatchError):
        im.to_cm(-1, "cm")
    w, d = im.room_size("furniture", "left", BED, WARDROBE)
    assert w >= 180 + 120 + 2 * im.CLEARANCE and d >= 200 + 2 * im.CLEARANCE
    w, d = im.room_size("furniture", "front", BED, WARDROBE)
    assert d >= 200 + 60 + 2 * im.CLEARANCE
    w, d = im.room_size("carpet", None, BED, im.Item("carpet", 240, 300))
    assert w >= 240 and d >= 300


def test_prompt_carries_sizes_units_placement_and_rules():
    prompt = im.build_prompt("furniture", "left", BED, WARDROBE, 0)
    for text in ("180 cm wide", "100 cm tall", "200 cm deep", "120 cm wide", "200 cm tall", "60 cm deep", "centimetres", "LEFT", "Image 1", "Image 2", "no collage", "not cropped"):
        assert text in prompt, text
    room_w, room_d = im.room_size("furniture", "left", BED, WARDROBE)
    assert f"{room_w} cm wide and {room_d} cm deep" in prompt
    # Every variation asks for a different camera.
    cameras = {im.build_prompt("furniture", "left", BED, WARDROBE, i).splitlines()[-1] for i in range(5)}
    assert len(cameras) == 5
    assert "underneath" in im.build_prompt("carpet", None, BED, im.Item("carpet", 240, 300), 0)
    paint = im.build_prompt("wall_paint", "behind", BED, im.Item("wall_paint", 400, 260), 0, color="#2E7D32")
    assert "#2E7D32" in paint and "BEHIND" in paint


def test_reference_images_become_bounded_pngs_and_colours_become_swatches():
    big = io.BytesIO()
    Image.new("RGB", (4000, 2000), (10, 20, 30)).save(big, "JPEG")
    ref = Image.open(io.BytesIO(im.reference_png(big.getvalue())))
    assert ref.format == "PNG" and max(ref.size) == im.REF_SIDE and ref.size[0] / ref.size[1] == pytest.approx(2, rel=0.01)
    assert Image.open(io.BytesIO(im.swatch_png("#D8C9B0"))).getpixel((5, 5)) == (216, 201, 176)
    with pytest.raises(im.ImageMatchError):
        im.swatch_png("beige")
    with pytest.raises(im.ImageMatchError):
        im.reference_png(b"not an image")


def matcher(handler) -> im.ImageMatcher:
    settings = Settings(openai_api_key="sk-test", openai_image_model="gpt-image-test")
    return im.ImageMatcher(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_edit_sends_both_reference_images_to_the_image_edit_endpoint():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        seen["url"] = str(request.url)
        seen["images"] = body.count(b'name="image[]"')
        seen["model"] = b"gpt-image-test" in body
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}]})

    out = asyncio.run(matcher(handler).edit("a room", [png(), png(color=(0, 0, 200))]))
    assert seen == {"url": im.EDITS_URL, "images": 2, "model": True}
    assert Image.open(io.BytesIO(out)).size == (64, 48)


def test_failures_are_reported_and_partial_results_kept():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if b"second" in request.content:
            return httpx.Response(400, json={"error": {"message": "Your request was rejected."}})
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}]})

    images, errors = asyncio.run(matcher(handler).variations(["first", "second", "third"], [png(), png()]))
    assert [i for i, _ in images] == [0, 2] and len(errors) == 1 and "rejected" in errors[0]
    with pytest.raises(im.ImageMatchError) as caught:
        asyncio.run(matcher(handler).variations(["second"], [png(), png()]))
    assert caught.value.status_code == 422


def test_timeouts_become_a_clear_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(im.ImageMatchError) as caught:
        asyncio.run(matcher(handler).edit("a room", [png(), png()]))
    assert caught.value.status_code == 504


def test_api_validates_inputs():
    client = TestClient(app)
    base = {"modelId": "x", "matchType": "carpet", "width": "240", "height": "300", "unit": "cm"}
    assert client.post("/api/match", data={**base, "variationsCount": "6"}).status_code == 422
    assert client.post("/api/match", data={**base, "matchType": "sofa"}).status_code == 422
    assert client.post("/api/match", data={**base, "width": "-1"}).status_code == 422


def test_progress_percent_rises_but_never_claims_done_early():
    assert im.progress([False, False], 0, 75) == 5
    assert im.progress([False], 37.5, 75) == 50
    assert im.progress([False], 999, 75) <= 95  # slow request: held below 100
    assert im.progress([True, False], 0, 75) == 50  # one of two finished
    assert im.progress([True, True], 1, 75) == 95  # 100 only comes with the saved result


def test_progress_stream_ends_with_the_result():
    async def run():
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}]})

        events = [e async for e in matcher(handler).variations_with_progress(["a", "b"], [png(), png()], tick=0.01)]
        return events

    events = asyncio.run(run())
    assert events[0] == ("progress", 5) and events[-1][0] == "result"
    percents = [v for k, v in events if k == "progress"]
    assert percents == sorted(percents)
    images, errors = events[-1][1]
    assert len(images) == 2 and not errors
