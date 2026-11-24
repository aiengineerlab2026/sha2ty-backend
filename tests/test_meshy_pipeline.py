"""Provider-specific preprocessing: Meshy Image to Image → Image to 3D → Auto Split, with no vision-model call; other
providers keep their vision analysis. Every paid API (Meshy, Gemini, OpenAI) is mocked."""

import asyncio
import io
import json
import logging
from types import SimpleNamespace

import anyio
import httpx
import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image, ImageDraw
from pydantic import SecretStr

from app.api import vendor
from app.core import config
from app.core.config import Settings
from app.main import app
from app.services import fidelity
from app.services.generation_prompt import MESHY_CLEAN_PROMPT, MESHY_NEUTRAL_BACKGROUND, meshy_clean_prompt, split_prompt
from app.services.meshy_service import MeshyError, MeshyService, check_image_to_3d
from tests.test_generation_architecture import FakeCdn, textured_glb

SIGNED = "https://assets.meshy.test/clean.png?Expires=1&Signature=secret-sig"


def bed(clutter=False, stretch=1.0, white=False, W=600, H=420) -> bytes:
    """A bed photo: headboard, frame, legs; `clutter` a white duvet and two pillows on it; `white` the room removed."""
    img = Image.new("RGB", (W, H), (255, 255, 255) if white else (120, 120, 120))
    d = ImageDraw.Draw(img)

    def X(x):
        return int(W // 2 + (x - W // 2) * stretch)

    d.rectangle((X(120), 60, X(480), 200), fill=(150, 110, 70), outline=(40, 30, 20), width=6)
    d.rectangle((X(100), 200, X(500), 330), fill=(90, 60, 40), outline=(40, 30, 20), width=6)
    d.rectangle((X(110), 330, X(130), 380), fill=(40, 30, 20))
    d.rectangle((X(470), 330, X(490), 380), fill=(40, 30, 20))
    if clutter:
        d.rectangle((X(105), 170, X(495), 290), fill=(235, 235, 240))
        d.ellipse((X(150), 140, X(260), 200), fill=(200, 60, 60))
        d.ellipse((X(330), 140, X(440), 200), fill=(60, 60, 200))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


PHOTO = bed(clutter=True)
CLEANED = bed(white=True)
BOX = [round(float(v), 4) for v in fidelity._foreground_box(CLEANED)[0]]  # the bed in PHOTO, as a photo reading gives it


def see_through(cleaned: bytes = CLEANED, alpha: int = 60) -> bytes:
    """`cleaned` as background removal that ate the furniture: the room transparent, the bed itself semi-transparent
    (light doors against a light wall, Meshy gpt-image-2-5-sunburst on a wardrobe)."""
    img = Image.open(io.BytesIO(cleaned)).convert("RGBA")
    a = np.asarray(img).copy()
    white = (a[..., :3] > 242).all(axis=-1)
    a[..., 3] = np.where(white, 0, alpha)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


class Meshy:
    """Meshy with every call mocked. *_states: what successive polls see (the last one repeats)."""

    def __init__(self, cleaned=CLEANED, image_states=("IN_PROGRESS", "SUCCEEDED"), mesh_states=("IN_PROGRESS", "SUCCEEDED"),
                 split_states=("IN_PROGRESS", "SUCCEEDED"), image_urls=1, refuse_image=None, slow_download=0.0):
        self.calls: list = []
        self.cleaned, self.image_urls, self.refuse_image, self.slow = cleaned, image_urls, refuse_image, slow_download
        self.states = {"image": list(image_states), "mesh": list(mesh_states), "split": list(split_states)}
        self.glb = textured_glb((90, 60, 40))

    def _next(self, kind):
        return self.states[kind].pop(0) if len(self.states[kind]) > 1 else self.states[kind][0]

    async def create_image_to_image_task(self, prompt, **kwargs):
        if self.refuse_image:
            raise self.refuse_image
        self.calls.append(("image-to-image", {"prompt": prompt, **kwargs}))
        return "img-1"

    async def get_image_to_image_task(self, _id):
        state = self._next("image")
        if isinstance(state, MeshyError):
            raise state
        return {"status": state, "progress": 100 if state == "SUCCEEDED" else 40, "image_urls": [SIGNED] * self.image_urls if state == "SUCCEEDED" else [],
                "task_error": {"message": "image refused"} if state == "FAILED" else None}

    async def download_bytes(self, url):
        self.calls.append(("download", url))
        await asyncio.sleep(self.slow)
        return self.cleaned

    async def create_image_to_3d_task(self, _options, **kwargs):
        self.calls.append(("image-to-3d", kwargs))
        return "mesh-1"

    async def get_image_to_3d_task(self, _id):
        state = self._next("mesh")
        return {"status": state, "progress": 100 if state == "SUCCEEDED" else 50, "model_urls": {"glb": "https://assets.meshy.test/m.glb"} if state == "SUCCEEDED" else {},
                "task_error": {"message": "mesh failed"} if state == "FAILED" else None}, None

    async def create_split_task(self, input_task_id, parts, keep_texture):
        self.calls.append(("split", {"input_task_id": input_task_id, "parts": parts}))
        return "split-1"

    async def get_split_task(self, _id):
        state = self._next("split")
        return {"status": state, "progress": 50, "part_count": 1, "prompt_ignored": False,
                "model_urls": {"glb": "https://assets.meshy.test/split.glb"} if state == "SUCCEEDED" else {}}

    async def download_file(self, _url, dest):
        self.calls.append(("download-glb", _url))
        dest.write_bytes(self.glb)
        return len(self.glb)

    def made(self, kind):
        return [kw for k, kw in self.calls if k == kind]


class Recorder:
    def __init__(self):
        self.vision, self.refunds, self.tries = [], 0, 0


def run(tmp_path, monkeypatch, meshy: Meshy, provider="meshy", category="furniture", kind="double bed", polls=12, astra=None, on_poll=None, **settings):
    """POST /generate, then poll until the task ends. Every vision entry point records (and Meshy's must stay empty)."""
    rec = Recorder()

    def recording(name, value=None):
        async def call(*_a, **_k):
            rec.vision.append(name)
            return value
        return call

    async def take(*_a):
        rec.tries += 1
        return 29

    async def publish(event, _state):
        if type(event).__name__ == "GenerationFailed":
            rec.refunds += 1

    async def chosen(*_a, **_k):
        return provider

    for name in ("ask_json", "clean_image"):
        monkeypatch.setattr(vendor.vision, name, recording(name))
    monkeypatch.setattr(vendor.vision, "current_model", recording("current_model", "gemini-3.1-pro-preview"))
    monkeypatch.setattr(vendor, "read_furniture_spec", recording("read_furniture_spec"))
    monkeypatch.setattr(vendor, "take_try", take)
    monkeypatch.setattr(vendor.bus, "publish", publish)
    monkeypatch.setattr(vendor.generator_setting, "current", chosen)
    cdn = FakeCdn()
    s = Settings(meshy_api_key="msy-test", openai_api_key="sk-test", storage_dir=tmp_path, supabase_url="", **settings)
    monkeypatch.setattr(vendor, "get_settings", lambda: s)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: meshy, vendor.get_cdn: lambda: cdn,
                                     config.get_settings: lambda: s, **({vendor.get_astra: lambda: astra} if astra else {})})
    seen: list[dict] = []
    try:
        with TestClient(app) as client:
            res = client.post("/api/vendor/generate", files={"image": ("bed.png", PHOTO, "image/png")}, data={"category": category, "kind": kind, "lang": "en"})
            first = res.json()
            if res.status_code == 200 and "glb_url" not in first:
                for _ in range(polls):
                    seen.append(client.get(f"/api/vendor/tasks/{first['task_id']}").json())
                    if on_poll:  # a test acting mid-task (a decision, a retry); what it returns is the next answer
                        seen[-1] = on_poll(client, first["task_id"], seen[-1]) or seen[-1]
                    if seen[-1]["status"] in ("SUCCEEDED", "FAILED", "CANCELED"):
                        break
                again = client.get(f"/api/vendor/tasks/{first['task_id']}").json() if seen else None
            else:
                again = None
    finally:
        app.dependency_overrides.clear()
    return SimpleNamespace(status=res.status_code, first=first, polls=seen, done=seen[-1] if seen else first, again=again, rec=rec, cdn=cdn, meshy=meshy)


def phases(r) -> list[str]:
    out: list[str] = []
    for p in r.polls:
        if not out or out[-1] != p.get("phase"):
            out.append(p.get("phase"))
    return out


# ---- 1, 2, 3, 9, 10, 12: the default Meshy pipeline ----


def test_meshy_cleans_with_image_to_image_then_meshes_that_task_with_no_vision_call(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, Meshy())
    assert r.status == 200 and r.first["details"] is None  # no photo reading: the vendor's own fields stand
    assert r.rec.vision == []  # 2. no Gemini / OpenAI call, no admin vision-model lookup, anywhere
    assert [k for k, _ in r.meshy.calls if k != "download-glb"] == ["image-to-image", "download", "image-to-3d", "split"]
    clean = r.meshy.made("image-to-image")[0]  # 1. Meshy Image to Image, with the furniture prompt
    assert clean["prompt"].startswith(MESHY_CLEAN_PROMPT) and "The main furniture item is a double bed." in clean["prompt"]
    assert clean["aspect_ratio"] == "3:2" and clean["image"] == PHOTO  # 600×420 (1.43): GPT Image's nearest ratio
    mesh = r.meshy.made("image-to-3d")[0]  # 3. its output is the Image to 3D input, by task id
    assert mesh["input_task_id"] == "img-1" and mesh["image"] is None and mesh["image_url"] is None
    split = r.meshy.made("split")[0]  # 9. Auto Split from the furniture type, no analysis
    assert split["input_task_id"] == "mesh-1" and split["parts"].startswith("headboard")
    # 12. progress through every phase, never backwards, ending at the final GLB
    assert phases(r) == ["image_preprocessing", "image_validation", "mesh_generation", "auto_split", "postprocessing", "completed"]
    progress = [p["progress"] for p in r.polls]
    assert progress == sorted(progress) and progress[-1] == 100
    done = r.done
    assert done["status"] == "SUCCEEDED" and done["glb_url"] == "https://cdn.test/vendor-1/" + done["glb_url"].rsplit("/", 1)[1]
    assert done["clean_image_url"].endswith(".clean.png") and not done["warnings"]
    doc = done["components"]
    check = doc["cleaning"]["check"]
    # no photo reading: no box to compare layout in, so the absolute checks ran and the layout was not judged (not faked)
    assert doc["cleaning"]["used"] == "cleaned" and check["mode"] == "removal" and check["status"] == "passed"
    assert check["checked"] == ["alpha", "colours"] and "layout and edges not compared" in check["notes"][0]  # said, not hidden
    assert doc["meshyInput"]["inputTaskId"] == "img-1" and doc["meshyInput"]["texture"] == "input_image"
    # 10. no spec: no structural model, said plainly; parts named from geometry
    assert not doc["structural"]["built"] and "No structural specification" in doc["structural"]["uncertain"][0] and "carpenter" not in doc
    assert "named from the model's own geometry" in doc["limits"]
    # Auto Split came back as one mesh: the model is kept and said not to be modular (never claimed modular)
    assert doc["split"]["accepted"] is False and "not modular" in doc["limits"] and done["source_glb_url"] is None
    assert r.rec.tries == 1 and r.rec.refunds == 0
    # 11. a later poll gets the stored answer: no Meshy call, download or processing again
    calls = len(r.meshy.calls)
    assert r.again["glb_url"] == done["glb_url"] and len(r.meshy.calls) == calls


def test_the_cleaning_prompt_is_configurable_without_a_vision_model():
    p = meshy_clean_prompt("wardrobe", None, hints={"wardrobe": "Keep the crown moulding."})
    assert p.startswith(MESHY_CLEAN_PROMPT) and "The main furniture item is a wardrobe." in p and p.endswith("Keep the crown moulding.")
    assert meshy_clean_prompt("furniture", None, override="Only remove the cat.", neutral_background=False) == "Only remove the cat."
    assert meshy_clean_prompt("furniture", None, override="Only remove the cat.").endswith(MESHY_NEUTRAL_BACKGROUND)
    assert meshy_clean_prompt("furniture", None, neutral_background=False) == MESHY_CLEAN_PROMPT  # nothing typed: nothing assumed


# ---- 4, 5: the fallback policy ----


def test_a_failed_image_to_image_task_meshes_the_original_and_says_so(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, Meshy(image_states=("FAILED",)), meshy_auto_split=False)
    mesh = r.meshy.made("image-to-3d")[0]
    assert mesh["image"] == PHOTO and mesh["input_task_id"] is None and not r.meshy.made("download")
    done = r.done
    assert done["status"] == "SUCCEEDED" and done["clean_image_url"] is None
    assert any("Image cleaning failed (image refused)" in w and "original photo was used" in w for w in done["warnings"])
    doc = done["components"]
    assert doc["cleaning"]["used"] == "original" and doc["cleaning"]["cleaned"] is False and doc["cleaning"]["fallback"]["what"] == "failed"
    assert any("Image cleaning failed" in w for w in doc["validation"]["warnings"])  # shown with the model's check
    assert [s["status"] for s in done["steps"]] == ["SKIPPED", "SUCCEEDED"] and r.rec.refunds == 0


def test_with_the_fallback_off_a_failed_cleaning_stops_and_refunds(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, Meshy(image_states=("FAILED",)), meshy_use_original_on_bad_cleaning=False)
    assert r.done["status"] == "FAILED" and r.done["phase"] == "failed" and "image refused" in r.done["error"]
    assert not r.meshy.made("image-to-3d") and r.rec.refunds == 1
    assert r.again["status"] == "FAILED" and r.rec.refunds == 1  # polled again: refunded once


def test_a_refused_image_to_image_request_meshes_the_original(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, Meshy(refuse_image=MeshyError(429, "Too many Meshy tasks")), meshy_auto_split=False)
    assert r.meshy.made("image-to-3d")[0]["image"] == PHOTO
    assert r.done["status"] == "SUCCEEDED" and "Image cleaning failed (Too many Meshy tasks)" in r.done["warnings"][0]


def test_a_see_through_cleaned_image_is_rejected_and_the_original_meshed(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, Meshy(cleaned=see_through()), meshy_auto_split=False)
    mesh = r.meshy.made("image-to-3d")[0]
    assert mesh["image"] == PHOTO and mesh["input_task_id"] is None  # the rejected task is never the input
    doc = r.done["components"]
    assert doc["cleaning"]["rejected"] and doc["cleaning"]["used"] == "original" and doc["cleaning"]["check"]["ok"] is False
    assert "see-through" in doc["cleaning"]["check"]["issues"][0]
    assert doc["cleaning"]["image_url"].endswith(".clean.png")  # the rejected image kept for debugging, not as the used one
    assert any("rejected" in w for w in r.done["warnings"]) and r.done["clean_image_url"] is None


class WireMeshy(Meshy):
    """The fake Meshy, but Image to 3D goes through the real MeshyService request builder to a mock transport: the
    test sees the exact HTTP body Meshy would receive."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sent: list = []
        self.real = _service(self.sent)

    async def create_image_to_3d_task(self, options, **kwargs):
        self.calls.append(("image-to-3d", kwargs))
        await self.real.create_image_to_3d_task(options, **kwargs)
        return "mesh-1"


def test_after_a_successful_cleaning_the_outgoing_image_to_3d_body_references_that_task_and_not_the_photo(tmp_path, monkeypatch, caplog):
    meshy = WireMeshy()
    with caplog.at_level(logging.INFO, logger=vendor.logger.name):
        r = run(tmp_path, monkeypatch, meshy, meshy_auto_split=False)
    assert r.done["status"] == "SUCCEEDED" and not r.done["warnings"]
    (request,) = [q for q in meshy.sent if q.url.path.endswith("/image-to-3d")]
    body = json.loads(request.content)
    image_task = r.meshy.made("image-to-image") and "img-1"  # the fake's Image to Image task id
    assert body["input_task_id"] == image_task == r.done["components"]["cleaning"]["imageTaskId"]
    assert "image_url" not in body and "texture_image_url" not in body and "texture_prompt" not in body
    assert b"base64" not in request.content  # the original photo is not in the request in any form
    assert "image_source=input_task_id image_to_image_task=img-1 input_task_id=img-1" in caplog.text
    assert "base64," not in caplog.text and "Signature" not in caplog.text


def test_the_cleaned_handoff_refuses_any_task_but_the_one_that_produced_the_image():
    pending = {"mesh": {"textured": True, "texture_prompt": None, "target_polycount": None}, "image_task_id": "img-1"}
    with pytest.raises(RuntimeError):
        anyio.run(lambda: vendor._start_mesh(None, Meshy(), "t", pending, "cleaned", image=CLEANED, input_task_id="img-OTHER"))
    with pytest.raises(RuntimeError):
        anyio.run(lambda: vendor._start_mesh(None, Meshy(), "t", pending, "cleaned", image=CLEANED))


def test_more_than_one_cleaned_image_is_not_used_as_the_input_task(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, Meshy(image_urls=3), meshy_auto_split=False)
    assert r.meshy.made("image-to-3d")[0]["image"] == PHOTO and "returned 3 images" in r.done["warnings"][0]


# ---- 6: cleaning off ----


def test_cleaning_off_goes_straight_to_image_to_3d(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, Meshy(), meshy_clean_image=False, meshy_auto_split=False)
    assert [k for k, _ in r.meshy.calls if k != "download-glb"] == ["image-to-3d"] and r.meshy.made("image-to-3d")[0]["image"] == PHOTO
    assert r.rec.vision == [] and phases(r) == ["mesh_generation", "postprocessing", "completed"]
    assert r.done["components"]["cleaning"]["used"] == "original" and not r.done["warnings"]


# ---- 7, 8: other providers keep the vision workflow and never touch Image to Image ----


def test_astra_keeps_its_vision_analysis_and_never_calls_meshy(tmp_path, monkeypatch):
    class Astra:
        def __init__(self):
            self.prompts = []

        async def generate_glb(self, _raw, _mime, prompt, _model=None):
            self.prompts.append(prompt)
            return textured_glb((90, 60, 40))

    class NoMeshy:
        def __getattr__(self, name):
            raise AssertionError(f"Meshy.{name} called for Astra")

    astra = Astra()
    r = run(tmp_path, monkeypatch, NoMeshy(), provider="astra", astra=astra)
    assert r.status == 200 and r.first["glb_url"] and len(astra.prompts) == 1
    assert "read_furniture_spec" in r.rec.vision  # 7. the existing photo reading still runs for Astra


# ---- 11: concurrent polls ----


def test_concurrent_polls_create_one_image_to_3d_task(monkeypatch):
    monkeypatch.setattr(vendor, "get_settings", lambda: Settings(meshy_api_key="msy-test", supabase_url=""))
    meshy = Meshy(slow_download=0.05, mesh_states=("IN_PROGRESS",))

    async def go():
        pending = {"stage": "validation", "clean_url": SIGNED, "image_task_id": "img-1", "photo": PHOTO, "mime": "image/png", "user_id": "v",
                   "model_id": "m", "warnings": [], "mesh": {"textured": True, "texture_prompt": "t", "target_polycount": 60000}}
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(pending_cdn={"t1": pending})))
        poll = lambda: vendor.get_task("t1", request, meshy, FakeCdn(), "v", BackgroundTasks())  # noqa: E731
        return await asyncio.gather(poll(), poll(), poll())

    out = anyio.run(go)
    assert len(meshy.made("image-to-3d")) == 1 and len(meshy.made("download")) == 1
    assert all(o.status == "IN_PROGRESS" and o.phase == "mesh_generation" for o in out)


def test_a_rate_limited_poll_waits_instead_of_failing(monkeypatch):
    monkeypatch.setattr(vendor, "get_settings", lambda: Settings(meshy_api_key="msy-test", supabase_url=""))
    meshy = Meshy(image_states=(MeshyError(429, "Meshy rate limit exceeded.", retry_after=7),))

    async def go():
        pending = {"stage": "image", "image_task_id": "img-1", "photo": PHOTO, "user_id": "v", "warnings": [], "clean_started": 0.0,
                   "mesh": {"textured": True, "texture_prompt": "t", "target_polycount": 1}}
        import time
        pending["clean_started"] = time.monotonic()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(pending_cdn={"t1": pending})))
        return await vendor.conditioned_image(request, meshy, None, "t1", pending)

    out = anyio.run(go)
    assert out.status == "IN_PROGRESS" and out.retry_after_seconds == 7 and not meshy.made("image-to-3d")


def test_an_image_task_past_its_timeout_falls_back_to_the_original(monkeypatch):
    monkeypatch.setattr(vendor, "get_settings", lambda: Settings(meshy_api_key="msy-test", supabase_url="", meshy_clean_timeout_seconds=1))
    meshy = Meshy(image_states=("IN_PROGRESS",))

    async def go():
        pending = {"stage": "image", "image_task_id": "img-1", "photo": PHOTO, "user_id": "v", "warnings": [], "clean_started": -1e9,
                   "mesh": {"textured": True, "texture_prompt": "t", "target_polycount": 1}}
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(pending_cdn={"t1": pending})))
        return await vendor.conditioned_image(request, meshy, None, "t1", pending)

    out = anyio.run(go)
    assert meshy.made("image-to-3d")[0]["image"] == PHOTO and "timed out" in out.warnings[0]


def test_a_failed_image_to_3d_task_is_refunded_once(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, Meshy(mesh_states=("FAILED",)), meshy_clean_image=False)
    assert r.done["status"] == "FAILED" and r.done["error"] == "mesh failed" and r.again["status"] == "FAILED"
    assert r.rec.tries == 1 and r.rec.refunds == 1


# ---- 13: invalid Meshy parameters never reach Meshy ----


def _service(sent: list, **settings) -> MeshyService:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(202, json={"result": "task-1"})

    s = Settings(meshy_api_key=SecretStr("msy-secret-key"), **settings)
    api = httpx.AsyncClient(base_url=s.meshy_base_url, transport=httpx.MockTransport(handler))
    return MeshyService(s, api, api)


@pytest.mark.parametrize("settings", [
    {"meshy_texture_model": "meshy-n1", "meshy_texture_resolution": "4k"},
    {"meshy_texture_model": "meshy-6-lite", "meshy_texture_resolution": "8k"},
])
def test_invalid_image_to_3d_settings_are_refused_before_submission(settings):
    sent: list = []
    service = _service(sent, **settings)
    from app.schemas.vendor import GenerationOptions

    with pytest.raises(MeshyError) as err:
        anyio.run(lambda: service.create_image_to_3d_task(GenerationOptions(should_texture=True, enable_pbr=True), image=PHOTO))
    assert err.value.status_code == 400 and "Invalid Meshy Image to 3D settings" in err.value.message and sent == []


def test_payload_rules_follow_the_docs():
    base = {"ai_model": "meshy-6", "image_url": "x", "should_texture": True}
    assert any("4k needs" in e for e in check_image_to_3d({**base, "geometry_resolution": "4k"}))
    assert any("texture_prompt and texture_image_url" in e for e in check_image_to_3d({**base, "texture_prompt": "a", "texture_image_url": "b"}))
    assert any("meshy-n1 does not take enable_pbr" in e for e in check_image_to_3d({**base, "texture_model": "meshy-n1", "enable_pbr": True}))
    assert check_image_to_3d({**base, "ai_model": "meshy-7.1", "geometry_resolution": "4k", "texture_resolution": "4k"}) == []


def test_quality_settings_reach_the_payload_valid():
    sent: list = []
    service = _service(sent, meshy_geometry_resolution="2k", meshy_texture_resolution="4k", meshy_texture_model="meshy-7")
    from app.schemas.vendor import GenerationOptions

    anyio.run(lambda: service.create_image_to_3d_task(GenerationOptions(should_texture=True, enable_pbr=True), input_task_id="img-1"))
    body = json.loads(sent[0].content)
    assert body["input_task_id"] == "img-1" and "image_url" not in body and "texture_image_url" not in body and "texture_prompt" not in body
    assert body["ai_model"] == "meshy-7.1" and body["geometry_resolution"] == "2k" and body["texture_resolution"] == "4k"
    assert body["texture_model"] == "meshy-7" and body["enable_pbr"] is True and body["should_remesh"] is False and body["image_enhancement"] is False


def test_the_default_image_to_image_model_is_gpt_image_2_5_sunburst():
    sent: list = []
    anyio.run(lambda: _service(sent).create_image_to_image_task("clean", image=PHOTO, aspect_ratio="3:2"))
    body = json.loads(sent[0].content)
    assert body["ai_model"] == "gpt-image-2-5-sunburst" and body["aspect_ratio"] == "3:2"
    sent.clear()  # nano-banana rejects 3:2: it is left out rather than sent and refused
    anyio.run(lambda: _service(sent, meshy_image_model="nano-banana").create_image_to_image_task("clean", image=PHOTO, aspect_ratio="3:2"))
    assert "aspect_ratio" not in json.loads(sent[0].content)


def test_an_unknown_image_to_image_model_is_refused_before_submission():
    sent: list = []
    with pytest.raises(MeshyError) as err:
        anyio.run(lambda: _service(sent, meshy_image_model="dall-e-9").create_image_to_image_task("clean", image=PHOTO))
    assert err.value.status_code == 400 and sent == []


# ---- 14: logs carry no image data, key or signed URL ----


def test_logs_redact_images_keys_and_signed_urls(caplog):
    sent: list = []
    service = _service(sent)
    with caplog.at_level(logging.INFO):
        anyio.run(lambda: service.create_image_to_image_task("clean", image=PHOTO, aspect_ratio="4:3"))
        anyio.run(lambda: service.create_image_to_image_task("clean", image_url=SIGNED))

        def refuse(_request):
            raise httpx.ConnectError("connection refused for " + SIGNED)

        downloads = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
        assert anyio.run(lambda: MeshyService(Settings(meshy_api_key="msy-secret-key"), downloads, downloads).download_bytes(SIGNED)) is None
    logs = caplog.text
    assert "data URI" in logs and "base64," not in logs  # the photo is described, never logged
    assert "Signature" not in logs and "secret-sig" not in logs and "msy-secret-key" not in logs
    assert json.loads(sent[0].content)["aspect_ratio"] == "4:3"


# ---- fidelity: removal is not damage ----


def test_removing_bedding_passes_removal_mode_but_failed_the_strict_check():
    legit = fidelity.compare_images(PHOTO, CLEANED, BOX, removal=True)
    assert legit["ok"] is True and legit["mode"] == "removal"
    assert fidelity.compare_images(PHOTO, CLEANED, BOX)["ok"] is False  # the old check rejected legitimate cleanup


def test_without_a_reading_a_reframed_cleaning_is_not_rejected_by_comparing_the_wrong_region():
    """Meshy centres and rescales the product: the cleaned image's box is another region of the original. That
    compared a good cleaning with a patch of wall (edge match ~0) and meshed the original instead."""
    img = Image.open(io.BytesIO(CLEANED))
    reframed = Image.new("RGB", (600, 420), (255, 255, 255))
    reframed.paste(img.crop((60, 40, 540, 400)).resize((300, 225)), (220, 120))
    buf = io.BytesIO()
    reframed.save(buf, "PNG")
    check = fidelity.compare_images(PHOTO, buf.getvalue(), None, removal=True)
    assert check["status"] == "passed" and not check["issues"] and "layout" not in check["checked"]


@pytest.mark.parametrize(("cleaned", "box", "why"), [
    (bed(white=True, stretch=1.4), BOX, "proportions"),
    (bed(white=True, stretch=1.25), BOX, "proportions"),
    (bed(white=True, W=200, H=140), None, "too small"),
    (see_through(), None, "see-through"),
])
def test_destructive_cleaning_is_rejected(cleaned, box, why):
    check = fidelity.compare_images(PHOTO, cleaned, box, removal=True)
    assert check["ok"] is False and any(why in i for i in check["issues"])


def test_an_empty_or_reframed_cleaned_image_is_not_passed_as_checked():
    empty = io.BytesIO()
    Image.new("RGB", (600, 420), (255, 255, 255)).save(empty, "PNG")
    assert fidelity.compare_images(PHOTO, empty.getvalue(), None, removal=True)["ok"] is False
    square = bed(white=True, W=500, H=500)
    framed = fidelity.compare_images(PHOTO, square, BOX, removal=True)  # colours checked; the layout cannot be
    assert framed["status"] == "passed" and "layout" not in framed["checked"] and "framed differently" in framed["notes"][0]
    unread = fidelity.compare_images(b"not an image", CLEANED, None, removal=True)  # nothing to compare with
    assert unread["status"] == "inconclusive" and unread["ok"] is None
