"""Meshy visual fidelity and modularity: what Meshy is given, what it textures from, when a cleaned photo or an
Auto Split result is refused, and what the checks say about a real Meshy wardrobe. No paid call is made."""

import asyncio
import glob
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from app.api import vendor
from app.core import config
from app.core.config import Settings
from app.main import app
from app.schemas.vendor import GenerationOptions
from app.services import fidelity, furniture_spec, mesh_analysis
from app.services.meshy_service import image_to_3d_payload, texture_conditioning

ROOT = Path(__file__).resolve().parents[1]
WARDROBE_GLB = ROOT / "storage/models/05c9faa43f5842f3b8e1c02a7defeefc.glb"  # a real Meshy result (local only)
# The photo reading of that same wardrobe (other cached readings are other photos).
WARDROBE_SPEC = sorted(glob.glob(str(ROOT / "storage/furniture-spec/aea18cd13a167b149dbfd11c744d34a65977b74d8dac931153999e370a411f83.json")))


def poll(client, task_id: str) -> dict:
    """Polls a Meshy task until it ends (the processing poll comes one poll after Meshy reports success)."""
    for _ in range(6):
        done = client.get(f"/api/vendor/tasks/{task_id}").json()
        if done.get("status") in ("SUCCEEDED", "FAILED", "CANCELED"):
            break
    return done


# ---- 1. one texture guide, and which one Meshy uses ----


@pytest.mark.parametrize(("source", "fields", "used"), [
    ("image", {"texture_image_url"}, "texture_image_url"),
    ("input", set(), "input_image"),
    ("prompt", {"texture_prompt"}, "texture_prompt"),
])

def test_meshy_gets_exactly_one_texture_guide(source, fields, used):
    payload = image_to_3d_payload(GenerationOptions(should_texture=True, enable_pbr=True), {"image_url": "data:image/png;base64,AA"}, "Oak wardrobe.", 30000, source)
    assert {"texture_prompt", "texture_image_url"} & set(payload) == fields  # never both: texture_prompt would win
    assert texture_conditioning(payload) == used
    assert payload["enable_pbr"] is True and payload["should_remesh"] is False and payload["image_enhancement"] is False
    if source == "image":
        assert payload["texture_image_url"] == payload["image_url"]  # the very image being meshed


def test_the_default_textures_from_the_photo():
    assert Settings().meshy_texture_source == "image" and Settings().meshy_auto_split is True


# ---- 2. a cleaned photo is checked before Meshy sees it ----


def product_photo(width=200, height=140, colours=((181, 140, 96), (40, 36, 30)), background=(120, 120, 120), box=(40, 20, 160, 120)) -> bytes:
    """A wardrobe-like product (two colour bands and a dark frame) in a room."""
    img = Image.new("RGB", (width, height), background)
    x0, y0, x1, y1 = box
    px = img.load()
    for x in range(x0, x1):
        for y in range(y0, y1):
            edge = x - x0 < 6 or x1 - x < 6 or y - y0 < 6 or y1 - y < 6
            px[x, y] = colours[1] if edge else colours[0] if x < (x0 + x1) // 2 else (210, 205, 190)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def cleaned_like(photo: bytes, bbox, stretch=1.0) -> bytes:
    """The product cut out on white, optionally stretched sideways (what a bad cleaning does)."""
    img = Image.open(io.BytesIO(photo)).convert("RGB")
    w, h = img.size
    prod = img.crop((int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)))
    prod = prod.resize((int(prod.width * stretch), prod.height))
    out = Image.new("RGB", (prod.width + 40, prod.height + 40), (255, 255, 255))
    out.paste(prod, (20, 20))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


BBOX = [0.2, 0.1428, 0.8, 0.857]


def cleaned_in_place(photo: bytes, bbox, stretch=1.0) -> bytes:
    """What Meshy Image to Image returns: the same framing (aspect_ratio matched), the room replaced by white, the
    product where it was; optionally stretched sideways about its centre (what a bad cleaning does)."""
    img = Image.open(io.BytesIO(photo)).convert("RGB")
    w, h = img.size
    box = (int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h))
    prod = img.crop(box)
    prod = prod.resize((int(prod.width * stretch), prod.height))
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(prod, ((box[0] + box[2]) // 2 - prod.width // 2, box[1]))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def big_photo() -> bytes:
    return product_photo(width=600, height=420, box=(120, 60, 480, 360))  # the BBOX product, at a usable size


class GateMeshy:
    def __init__(self, cleaned: bytes | None):
        self.calls, self.cleaned = [], cleaned

    async def download_bytes(self, _url):
        return self.cleaned

    async def create_image_to_3d_task(self, _options, **kwargs):
        self.calls.append(kwargs)
        return "mesh-1"


def gate(cleaned: bytes | None, photo: bytes, monkeypatch, fallback=True):
    """The validation poll: the Image to Image task succeeded with one image (clean_url)."""
    async def quiet(*_a, **_k):
        return None

    monkeypatch.setattr(vendor.bus, "publish", quiet)
    monkeypatch.setattr(vendor, "get_settings", lambda: Settings(meshy_api_key="msy-test", meshy_use_original_on_bad_cleaning=fallback))

    async def go():
        meshy = GateMeshy(cleaned)
        pending = {"stage": "validation", "clean_url": "https://assets.meshy.test/c.png?sig=x", "image_task_id": "img-1", "photo": photo,
                   "mime": "image/png", "user_id": "v", "warnings": [], "details": {"bbox": BBOX},
                   "mesh": {"textured": True, "texture_prompt": "t", "target_polycount": 30000}}
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(pending_cdn={"t1": pending})))
        out = await vendor.conditioned_image(request, meshy, None, "t1", pending)
        return out, pending, meshy

    return asyncio.run(go())


def test_a_faithful_cleaned_photo_is_meshed_from_its_task_and_recorded(monkeypatch):
    photo = big_photo()
    out, pending, meshy = gate(cleaned_in_place(photo, BBOX), photo, monkeypatch)
    assert out.status == "IN_PROGRESS" and out.phase == "mesh_generation" and pending["cleaning_check"]["ok"]
    # The cleaned image goes by its Image to Image task (no re-upload); Meshy textures from that input image.
    assert meshy.calls[0]["input_task_id"] == "img-1" and meshy.calls[0]["image"] is None and meshy.calls[0]["image_url"] is None
    assert pending["meshy_input"]["source"] == "cleaned" and pending["meshy_input"]["texture"] == "input_image"
    assert pending["meshy_input"]["sha256"] and pending["reference"] and not out.warnings


def test_a_cleaned_photo_that_changed_the_proportions_is_rejected_and_the_original_meshed(monkeypatch):
    photo = big_photo()
    out, pending, meshy = gate(cleaned_in_place(photo, BBOX, stretch=1.4), photo, monkeypatch)
    check = pending["cleaning_check"]
    assert check["ok"] is False and check["rejected"] and any("proportions" in i for i in check["issues"])
    assert meshy.calls[0]["image"] == photo and meshy.calls[0]["input_task_id"] is None  # not the distorted one
    assert pending["meshy_input"]["source"] == "original" and "reference" not in pending
    assert out.warnings and "rejected" in out.warnings[0] and "original photo was used" in out.warnings[0]
    assert out.clean_image_url is None and out.steps[0].status == "SKIPPED"


def test_a_bad_cleaning_stops_when_the_fallback_is_off(monkeypatch):
    photo = big_photo()
    out, _pending, meshy = gate(cleaned_in_place(photo, BBOX, stretch=1.4), photo, monkeypatch, fallback=False)
    assert out.status == "FAILED" and out.phase == "failed" and "changed the furniture" in out.error and not meshy.calls


def test_an_unreadable_cleaned_image_is_not_meshed(monkeypatch):
    photo = big_photo()
    out, pending, meshy = gate(b"<html>not an image</html>", photo, monkeypatch)
    assert meshy.calls[0]["image"] == photo and "not a PNG, JPG or WEBP" in out.warnings[0]


def test_cleaning_that_redraws_the_furniture_is_caught_by_layout_not_only_colour():
    photo = product_photo()
    # Same colours, same size, but the light and dark halves swapped: a different piece of furniture.
    swapped = product_photo(colours=((210, 205, 190), (40, 36, 30)))
    img = Image.open(io.BytesIO(cleaned_like(swapped, BBOX))).transpose(Image.FLIP_LEFT_RIGHT)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    check = fidelity.compare_images(photo, buf.getvalue(), BBOX)
    assert check["metrics"]["layout"] > 0 and check["metrics"]["structure"] is not None


# ---- 3. Auto Split is used only when it is really the same model in parts ----


BOXES = [((0, 0, 0), (2, 2, 0.6)), ((0.04, 0.6, 0.6), (1, 1.9, 0.62)), ((1, 0.6, 0.6), (1.96, 1.9, 0.62))]


def boxes():
    return [trimesh.creation.box(bounds=[lo, hi]) for lo, hi in BOXES]


def fused() -> bytes:
    return trimesh.util.concatenate(boxes()).export(file_type="glb")


def separate() -> bytes:
    return trimesh.Scene(boxes()).export(file_type="glb")


OK = {"status": "SUCCEEDED", "taskId": "s1", "partCount": 3, "promptIgnored": False, "prompt": "body, left door, right door"}


def test_a_genuine_split_is_accepted_with_its_mesh_counts():
    report = fidelity.split_check(fused(), separate(), OK)
    assert report["accepted"] and report["meshesBefore"] == 1 and report["meshesAfter"] == 3 and report["partCount"] == 3


@pytest.mark.parametrize(("split", "task", "reason"), [
    (lambda: fused(), OK, "one fused mesh"),
    (lambda: separate(), {**OK, "promptIgnored": True}, "prompt_ignored"),
    (lambda: None, {"status": "FAILED", "error": "textured_split_refused", "prompt": "a, b"}, "textured_split_refused"),
    (lambda: trimesh.Scene([b.apply_transform(np.diag([1.3, 1, 1, 1])) for b in boxes()]).export(file_type="glb"), OK, "proportions"),  # stretched wider
    (lambda: b"glTF", OK, "cannot be read"),
])
def test_a_split_that_is_not_the_same_model_in_parts_is_refused_with_its_reason(split, task, reason):
    report = fidelity.split_check(fused(), split(), task)
    assert not report["accepted"] and any(reason in r for r in report["issues"]), report["issues"]
    assert report["status"] == task["status"] and report["promptIgnored"] == task.get("promptIgnored")


def test_a_fused_meshy_mesh_is_never_cut_into_guessed_parts(monkeypatch):
    monkeypatch.setattr(vendor, "segment_glb", lambda *_a: (_ for _ in ()).throw(AssertionError("no geometric guessing for Meshy")))
    glb, source, doc = vendor.meshy_parts(fused(), "wardrobe", None)
    assert glb == fused() or len(glb) == len(fused())
    assert source is None and doc["method"] == "none" and not doc["separated"] and "not modular" in doc["limits"]


def test_the_split_names_are_the_readings_real_components():
    spec = furniture_spec.parse(json.dumps({
        "object_type": "wardrobe", "width_m": 2, "height_m": 2, "depth_m": 0.6, "dimension_source": "inferred", "loose_objects": [], "design_features": [],
        "structure": {"objectType": "wardrobe", "components": [
            {"id": "body", "name": "Wardrobe body", "box": [0, 0, 1, 1], "role": "structural"},
            {"id": "door_l", "name": "Left door", "box": [0, 0.1, 0.5, 0.9], "role": "functional"},
            {"id": "door_r", "name": "Right door", "box": [0.5, 0.1, 1, 0.9], "role": "functional"},
            {"id": "back", "name": "Back panel", "box": None, "visibility": "inferred"},
            {"id": "vase", "name": "Vase", "box": [0.4, 0, 0.5, 0.05], "role": "decorative"},
        ]},
    }), "en")
    assert mesh_analysis.split_prompt(spec.analysis) == "wardrobe body, left door, right door"  # no hidden or loose parts


# ---- 4. the Meshy task end to end: split refused → the raw model, honestly one mesh, structural beside it ----


class TaskMeshy:
    def __init__(self, raw: bytes, split: bytes | None, split_task: dict):
        self.raw, self.split, self.split_task, self.made = raw, split, split_task, []

    async def create_image_to_3d_task(self, _options, **kwargs):
        self.made.append(kwargs)
        return "mesh-1"

    async def get_image_to_3d_task(self, _id):
        return {"status": "SUCCEEDED", "progress": 100, "model_urls": {"glb": "https://meshy.test/m.glb"}}, None

    async def create_split_task(self, _id, prompt, keep_texture):
        assert keep_texture is True  # textures kept on the parts
        self.made.append({"split": prompt})
        return "split-1"

    async def get_split_task(self, _id):
        return {**self.split_task, "model_urls": {"glb": "https://meshy.test/s.glb"} if self.split else {}}

    async def download_file(self, url, dest):
        dest.write_bytes(self.raw if url.endswith("/m.glb") else self.split)


class Cdn:
    def __init__(self):
        self.uploads = {}

    def upload(self, path, data, _type):
        self.uploads[path] = data
        return f"https://cdn.test/{path}"

    def public_url(self, path):
        return f"https://cdn.test/{path}"


SPEC = {
    "object_type": "wardrobe", "category": "wardrobe", "name": "W", "description": "W", "bbox": BBOX, "proportions": [1.0, 0.3, 1.0],
    "width_m": 2.0, "height_m": 2.0, "depth_m": 0.6, "dimension_source": "inferred", "confidence": 0.8, "loose_objects": [], "design_features": [],
    "structure": {"objectType": "wardrobe", "bodyId": "body", "components": [
        {"id": "body", "name": "Wardrobe body", "box": [0, 0, 1, 1], "role": "structural", "materialClass": "laminate", "color": "#B5895F"},
        {"id": "door_l", "name": "Left door", "box": [0.02, 0.05, 0.5, 0.95], "role": "functional", "materialClass": "painted_wood", "color": "#D2CDBE"},
        {"id": "door_r", "name": "Right door", "box": [0.5, 0.05, 0.98, 0.95], "role": "functional", "materialClass": "painted_wood", "color": "#D2CDBE"},
    ]},
}


def run_task(tmp_path, monkeypatch, split: bytes | None, split_task: dict):
    async def ask(*_a, **_k):
        return json.dumps(SPEC)

    async def one_try(*_a):
        return 4

    monkeypatch.setattr(vendor, "read_furniture_spec", furniture_spec.read)
    monkeypatch.setattr(vendor.vision, "ask_json", ask)
    monkeypatch.setattr(vendor, "take_try", one_try)
    meshy, cdn = TaskMeshy(fused(), split, split_task), Cdn()
    s = Settings(meshy_api_key="msy-test", meshy_clean_image=False, meshy_vision_analysis=True, gemini_api_key="gem-test", storage_dir=tmp_path)
    monkeypatch.setattr(vendor, "get_settings", lambda: s)
    app.dependency_overrides.update({vendor.current_user: lambda: "vendor-1", vendor.get_meshy: lambda: meshy, vendor.get_cdn: lambda: cdn, config.get_settings: lambda: s})
    try:
        with TestClient(app) as client:
            first = client.post("/api/vendor/generate", files={"image": ("p.png", product_photo(), "image/png")}, data={"category": "wardrobe", "lang": "en"}).json()
            done = poll(client, first["task_id"])
    finally:
        app.dependency_overrides.clear()
    return done, meshy, cdn


def test_a_refused_split_keeps_the_raw_model_and_says_it_is_one_mesh(tmp_path, monkeypatch):
    done, meshy, cdn = run_task(tmp_path, monkeypatch, fused(), {"status": "SUCCEEDED", "part_count": 1, "prompt_ignored": False})
    doc = done["components"]
    assert meshy.made[0]["image"] == product_photo() and meshy.made[1] == {"split": "wardrobe body, left door, right door"}
    assert doc["method"] == "none" and not doc["separated"] and "not modular" in doc["limits"] and "Auto Split was not used" in doc["limits"]
    assert doc["split"]["accepted"] is False and doc["split"]["meshesAfter"] == 1 and doc["split"]["status"] == "SUCCEEDED"
    assert done["source_glb_url"] is None  # no "original" beside an unsplit model: it is the original
    assert doc["meshyInput"]["source"] == "original" and doc["meshyInput"]["texture"] == "texture_image_url"
    assert cdn.uploads[doc["meshyInput"]["url"].removeprefix("https://cdn.test/")] == product_photo()  # the exact image
    assert doc["carpenter"]["url"] and doc["structural"]["built"]  # components are edited in the structural model
    assert "split" in {s["stage"] for s in doc["fidelity"]["stages"]}


def test_an_accepted_split_is_the_visual_model_and_the_unsplit_one_is_kept(tmp_path, monkeypatch):
    done, _meshy, _cdn = run_task(tmp_path, monkeypatch, separate(), {"status": "SUCCEEDED", "part_count": 3, "prompt_ignored": False})
    doc = done["components"]
    assert doc["method"] == "meshy_auto_split" and doc["separated"] and len(doc["components"]) == 3
    assert doc["split"]["accepted"] and done["source_glb_url"].endswith(".source.glb")
    stages = {s["stage"]: s for s in doc["fidelity"]["stages"]}
    assert stages["processing"]["metrics"]["meshes"] == [1, 3]


def test_a_failed_split_reports_meshys_status_and_error(tmp_path, monkeypatch):
    done, _meshy, _cdn = run_task(tmp_path, monkeypatch, None, {"status": "FAILED", "task_error": {"message": "textured_split_refused"}})
    split = done["components"]["split"]
    assert split["status"] == "FAILED" and split["error"] == "textured_split_refused" and not split["accepted"]


# ---- 5. proportions and axes from a camera-aligned render ----


def test_a_model_wider_than_the_photo_or_lying_wrong_is_named():
    expected = [1.0, 0.207, 0.931]  # width : depth : height (the wardrobe reading)
    ok, _ = fidelity.proportions(np.array([1.9, 1.77, 0.39]), expected)
    assert ok == []
    wide, _ = fidelity.proportions(np.array([2.6, 1.77, 0.39]), expected)
    assert any("wider than the photo" in i for i in wide)
    swapped, metrics = fidelity.proportions(np.array([0.39, 1.77, 1.9]), expected)
    assert any("width and depth swapped" in i for i in swapped) and metrics["axisFit"]["width and depth swapped"] < metrics["axisFit"]["as is"]


def test_the_front_render_is_the_same_camera_for_every_model():
    rgb, mask, span = fidelity.front_render(separate(), 64)
    assert mask.mean() > 0.9 and rgb.shape[1] == 64 and np.allclose(span, [2, 2, 0.62], atol=1e-3)


# ---- 6. the real Meshy wardrobe (experiment D on the stored raw GLB; skipped where the file is not present) ----

real = pytest.mark.skipif(not WARDROBE_GLB.exists() or not WARDROBE_SPEC, reason="the stored Meshy wardrobe GLB is local only")


@real
def test_real_meshy_wardrobe_is_one_fused_mesh_with_its_pbr_maps():
    raw = WARDROBE_GLB.read_bytes()
    assert fidelity.mesh_count(raw) == 1
    _, _, doc = vendor.meshy_parts(raw, "wardrobe", None)
    assert doc["method"] == "none" and not doc["separated"]


@real
def test_real_meshy_wardrobe_proportions_and_axes_against_its_reading():
    spec = json.loads(Path(WARDROBE_SPEC[0]).read_text())
    _, _, span = fidelity.front_render(WARDROBE_GLB.read_bytes())
    issues, metrics = fidelity.proportions(span, spec["details"]["proportions"])
    assert min(metrics["axisFit"], key=metrics["axisFit"].get) == "as is"  # upright, facing +z
    assert abs(metrics["modelWHD"][0] / metrics["photoWHD"][0] - 1) < 0.12 and issues == []  # ~8 % wider: within tolerance


@real
def test_real_meshy_wardrobe_processing_keeps_its_textures_and_shape():
    from app.services import geometry_quality
    from app.services.glb_size import fit_glb

    raw = WARDROBE_GLB.read_bytes()
    _, _, doc = vendor.meshy_parts(raw, "wardrobe", None)
    processed, _, _ = geometry_quality.process(raw, doc, None, False)
    processed, _ = fit_glb(processed)
    stages = {s["stage"]: s for s in fidelity.compare_glbs(raw, processed)}
    assert stages["processing"]["ok"] and stages["processing"]["metrics"]["meshes"] == [1, 1]
    assert stages["materials"]["ok"] and {"baseColorTexture", "metallicRoughnessTexture", "normalTexture"} <= set(stages["materials"]["metrics"]["maps"])
