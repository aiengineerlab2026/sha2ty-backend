"""Colour fidelity of Meshy's cleaned image, the inconclusive-cleaning decision, and modular output: Auto Split
accepted when it really is the same model in parts, kept through processing, retried without a new generation.
Every paid Meshy call is mocked (tests/test_meshy_pipeline.Meshy)."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from app.api import vendor
from app.services import fidelity
from app.services.generation_prompt import TYPE_NAMES, split_prompt
from tests.test_meshy_pipeline import BOX, CLEANED, PHOTO, Meshy, _service, bed, run, see_through

# ---- GLBs: one fused mesh (Image to 3D) and its parts (Auto Split, re-triangulated along the cuts) ----

STRIPES = [(150, 110, 70), (220, 160, 170), (240, 235, 225)]  # wood body, pink door, white door
PARTS = [((1.6, 2.0, 0.5), (0, 1.0, 0), 0), ((0.78, 1.9, 0.02), (-0.4, 1.0, 0.26), 1), ((0.78, 1.9, 0.02), (0.4, 1.0, 0.26), 2)]


def _material():
    img = Image.new("RGB", (48, 16))
    for i, c in enumerate(STRIPES):
        img.paste(c, (i * 16, 0, i * 16 + 16, 16))
    return trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=0.8)


def _part(extents, centre, stripe, material, subdivide=0):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(centre)
    for _ in range(subdivide):
        m = m.subdivide()
    uv = np.column_stack([np.full(len(m.vertices), (stripe + 0.5) / 3), np.full(len(m.vertices), 0.5)])
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return m


def raw_glb() -> bytes:
    material = _material()
    return trimesh.Scene(trimesh.util.concatenate([_part(e, c, s, material) for e, c, s in PARTS])).export(file_type="glb")


def split_glb(move=None) -> bytes:
    """The parts as separate meshes, each subdivided twice (16× the triangles, as a real cut re-triangulates)."""
    material, scene = _material(), trimesh.Scene()
    for i, (e, c, s) in enumerate(PARTS):
        c = np.add(c, move) if move is not None and i == 1 else c
        scene.add_geometry(_part(e, c, s, material, subdivide=2), node_name=f"part_{i}", geom_name=f"part_{i}")
    return scene.export(file_type="glb")


RAW, SPLIT = raw_glb(), split_glb()


class PartsMeshy(Meshy):
    """Meshy whose Image to 3D file is one fused mesh and whose Auto Split file is its three parts."""

    def __init__(self, split=SPLIT, fail_first_split=False, **kwargs):
        super().__init__(**kwargs)
        self.glb, self.split_file, self.fail_first_split = RAW, split, fail_first_split

    async def get_split_task(self, _id):
        task = await super().get_split_task(_id)
        if self.fail_first_split and len(self.made("split")) == 1:  # the first Auto Split task fails, a retry's works
            task = {**task, "status": "FAILED", "task_error": {"message": "split failed"}, "model_urls": {}}
        return task

    async def download_file(self, url, dest):
        self.calls.append(("download-glb", url))
        dest.write_bytes(self.split_file if "split" in url else self.glb)
        return dest.stat().st_size


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---- 1–5: image fidelity ----


def test_original_colours_pass_on_a_neutral_backdrop_and_on_transparency():
    gray = Image.open(io.BytesIO(CLEANED)).convert("RGB")
    a = np.asarray(gray).copy()
    a[(a > 242).all(axis=-1)] = (128, 128, 128)  # Meshy's neutral backdrop (remove_background off)
    transparent = Image.open(io.BytesIO(see_through(alpha=255)))  # the room removed, the furniture fully opaque
    for cleaned in (png(Image.fromarray(a)), png(transparent)):
        check = fidelity.compare_images(PHOTO, cleaned, BOX, removal=True)
        assert check["status"] == "passed" and check["checked"] == ["alpha", "colours", "layout"], check


def test_recolouring_the_furniture_is_detected_even_without_a_photo_reading():
    a = np.asarray(Image.open(io.BytesIO(CLEANED)).convert("RGB")).copy()
    wood = (np.abs(a.astype(int) - (90, 60, 40)).sum(-1) < 30) | (np.abs(a.astype(int) - (150, 110, 70)).sum(-1) < 30)
    a[wood] = (30, 90, 200)  # light wood painted blue: a colour that is nowhere in the photo
    check = fidelity.compare_images(PHOTO, png(Image.fromarray(a)), None, removal=True)
    assert check["status"] == "failed" and any("recoloured" in i for i in check["issues"])


def test_doors_turned_black_are_detected():
    a = np.asarray(Image.open(io.BytesIO(CLEANED)).convert("RGB")).copy()
    a[(np.abs(a.astype(int) - (150, 110, 70)).sum(-1) < 30)] = (12, 12, 12)  # the headboard (the "door") now black
    # The photo has no near-black of that kind on a light room: caught from colours alone.
    light_room = Image.open(io.BytesIO(PHOTO)).convert("RGB")
    lr = np.asarray(light_room).copy()
    lr[(np.abs(lr.astype(int) - (40, 30, 20)).sum(-1) < 30)] = (90, 60, 40)  # outlines as wood, no black anywhere
    check = fidelity.compare_images(png(Image.fromarray(lr)), png(Image.fromarray(a)), None, removal=True)
    assert check["status"] == "failed" and any("darker" in i or "recoloured" in i for i in check["issues"])
    # A photo that does have near-black (dark outlines, legs): colours alone cannot tell; the aligned layout check does.
    check = fidelity.compare_images(PHOTO, png(Image.fromarray(a)), BOX, removal=True)
    assert check["status"] == "failed" and any("layout" in i or "edges" in i for i in check["issues"]), check


def test_background_removal_that_deletes_a_door_is_detected_as_a_hole():
    """Measured on Meshy remove_background: whole wardrobe doors came back RGBA (0,0,0,0) inside the frame."""
    img = Image.open(io.BytesIO(see_through(alpha=255))).convert("RGBA")
    a = np.asarray(img).copy()
    a[75:185, 135:465] = 0  # the headboard's inside deleted, its outline kept
    check = fidelity.compare_images(PHOTO, png(Image.fromarray(a)), None, removal=True)
    assert check["status"] == "failed" and any("holes" in i for i in check["issues"])
    assert check["metrics"]["holes"] > fidelity.REMOVAL_HOLES


def test_gaps_open_on_one_side_are_not_holes():
    """Under a table top or between chair legs is transparent but not enclosed: never a hole."""
    img = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    a = np.asarray(img).copy()
    a[60:90, 40:360] = (150, 110, 70, 255)  # the top
    a[90:360, 40:70] = a[90:360, 330:360] = (150, 110, 70, 255)  # two legs
    _, holes = fidelity._alpha_damage(a.astype(float), fidelity._product_mask(a.astype(float)))
    assert holes == 0.0


def test_an_inconclusive_check_waits_for_the_user_and_the_cleaned_image_goes_by_its_task(tmp_path, monkeypatch):
    monkeypatch.setattr(vendor.fidelity, "compare_images", lambda *_a: {"stage": "cleaning", "mode": "removal", "status": "inconclusive",
                                                                       "ok": None, "issues": [], "notes": ["not compared"]})
    seen: list[str] = []

    def approve(client, task_id, answer):
        seen.append(answer["phase"])
        if answer["phase"] == "awaiting_approval":
            assert answer["review_image_url"].endswith(".clean.png") and answer["cleaning_check"]["status"] == "inconclusive"
            assert answer["clean_image_url"] is None  # not "used" until the user says so
            return client.post(f"/api/vendor/tasks/{task_id}/cleaning", json={"use": "cleaned"}).json()
        return None

    r = run(tmp_path, monkeypatch, Meshy(), meshy_auto_split=False, on_poll=approve)
    assert "awaiting_approval" in seen and r.done["status"] == "SUCCEEDED"
    (mesh,) = r.meshy.made("image-to-3d")
    assert mesh["input_task_id"] == "img-1" and mesh["image"] is None
    assert r.done["components"]["cleaning"]["check"]["decision"] == "cleaned" and any("approved" in w for w in r.done["warnings"])


def test_rejecting_an_inconclusive_cleaned_image_meshes_the_original_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(vendor.fidelity, "compare_images", lambda *_a: {"stage": "cleaning", "mode": "removal", "status": "inconclusive",
                                                                       "ok": None, "issues": [], "notes": ["not compared"]})

    def reject(client, task_id, answer):
        if answer["phase"] == "awaiting_approval":
            return client.post(f"/api/vendor/tasks/{task_id}/cleaning", json={"use": "original"}).json()
        return None

    # even with the "fail on bad cleaning" policy: the user's own choice of the original is honoured
    r = run(tmp_path, monkeypatch, Meshy(), meshy_auto_split=False, meshy_use_original_on_bad_cleaning=False, on_poll=reject)
    (mesh,) = r.meshy.made("image-to-3d")
    assert mesh["image"] == PHOTO and mesh["input_task_id"] is None
    assert r.done["components"]["cleaning"]["used"] == "original" and any("declined" in w for w in r.done["warnings"])


@pytest.mark.parametrize("policy", ["use_original", "use_cleaned"])
def test_the_inconclusive_policy_never_passes_silently(tmp_path, monkeypatch, policy):
    monkeypatch.setattr(vendor.fidelity, "compare_images", lambda *_a: {"stage": "cleaning", "mode": "removal", "status": "inconclusive",
                                                                       "ok": None, "issues": [], "notes": ["not compared"]})
    r = run(tmp_path, monkeypatch, Meshy(), meshy_auto_split=False, meshy_inconclusive_cleaning=policy)
    (mesh,) = r.meshy.made("image-to-3d")
    assert (mesh["input_task_id"] == "img-1") == (policy == "use_cleaned")
    assert r.done["warnings"] and r.done["cleaning_check"]["status"] == "inconclusive"


# ---- 7: modularity guidance only where Meshy reads it ----


@pytest.mark.parametrize("category", [*TYPE_NAMES, "sofa", "bed", "chair"])
def test_every_category_gets_auto_split_part_names_meshy_accepts(category):
    names = split_prompt(category, None)
    assert names and len(names) <= 600 and 1 <= len(names.split(",")) <= 10  # docs: 1 to 10 part names, 600 chars


def test_no_geometry_or_modularity_prompt_is_invented_for_image_to_3d():
    """Image to 3D has no geometry prompt (only texture_prompt, which guides the texture): nothing else is sent."""
    sent: list = []
    import anyio
    from app.services.meshy_service import GenerationOptions

    anyio.run(lambda: _service(sent).create_image_to_3d_task(GenerationOptions(should_texture=True, enable_pbr=True), input_task_id="img-1"))
    body = json.loads(sent[0].content)
    assert not {"prompt", "object_prompt", "parts", "modular"} & set(body) and "texture_prompt" not in body
    sent.clear()
    anyio.run(lambda: _service(sent).create_split_task("mesh-1", split_prompt("wardrobe"), True))
    body = json.loads(sent[0].content)
    assert body["mode"] == "by_parts" and body["input_task_id"] == "mesh-1" and body["keep_texture"] is True
    assert "each door" in body["prompt"] and len(body["prompt"].split(",")) <= 10


# ---- 8–12: Auto Split ----


def test_a_real_split_is_accepted_though_its_triangle_count_changed():
    report = fidelity.split_check(RAW, SPLIT, {"status": "SUCCEEDED", "partCount": 3})
    assert report["accepted"] and report["meshesBefore"] == 1 and report["meshesAfter"] == 3
    assert report["triangles"][1] > report["triangles"][0] * 10 and report["shape"]["silhouette"] > 0.99


@pytest.mark.parametrize(("split", "why"), [
    (split_glb(move=(0.8, 0, 0.001)), "colours changed"),  # a door moved onto the other: same size, wrong surface
    (split_glb(move=(-1.2, 0.6, 0)), "proportions"),  # a door pushed outside the body
])
def test_a_split_that_distorts_the_furniture_is_rejected(split, why):
    report = fidelity.split_check(RAW, split, {"status": "SUCCEEDED"})
    assert not report["accepted"] and any(why in i for i in report["issues"]), report["issues"]


def test_split_runs_and_its_parts_survive_processing(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, PartsMeshy(), category="wardrobe", kind="")
    assert r.done["status"] == "SUCCEEDED"
    (split,) = r.meshy.made("split")
    assert split["input_task_id"] == "mesh-1" and "each door" in split["parts"]
    doc = r.done["components"]
    assert doc["split"]["accepted"] and doc["split"]["meshesAfter"] == 3 and doc["method"] == "meshy_auto_split"
    final = next(v for k, v in r.cdn.uploads.items() if k.endswith(".glb") and not k.endswith((".raw.glb", ".source.glb", ".carpenter.glb")))
    assert fidelity.mesh_count(final) == 3  # grouping, naming, checks, animation and size fitting kept every part
    names = [n.get("name") for n in json.loads(final[20:20 + int.from_bytes(final[12:16], "little")])["nodes"]]
    assert len({n for n in names if n}) >= 3  # each part its own named, selectable node


def test_a_failed_split_is_reported_and_never_called_modular(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, PartsMeshy(split_states=("FAILED",)), category="wardrobe", kind="")
    doc = r.done["components"]
    assert r.done["status"] == "SUCCEEDED" and not doc["split"]["accepted"]
    assert "did not finish (FAILED" in doc["split"]["issues"][0] and doc["method"] == "none"
    assert "not modular" in doc["limits"] and "Auto Split was not used" in doc["limits"]


def test_retrying_auto_split_reuses_the_image_to_3d_task(tmp_path, monkeypatch):
    retried: list[dict] = []

    def retry_once(client, task_id, answer):
        if answer["status"] == "SUCCEEDED" and not retried:
            res = client.post(f"/api/vendor/tasks/{task_id}/retry-split")
            assert res.status_code == 200, res.text
            retried.append(res.json())
            return res.json()
        return None

    r = run(tmp_path, monkeypatch, PartsMeshy(fail_first_split=True), category="wardrobe", kind="", polls=20, on_poll=retry_once)
    assert retried[0]["phase"] == "auto_split"
    assert len(r.meshy.made("image-to-image")) == 1 and len(r.meshy.made("image-to-3d")) == 1  # nothing regenerated
    assert [s["input_task_id"] for s in r.meshy.made("split")] == ["mesh-1", "mesh-1"]
    assert r.done["status"] == "SUCCEEDED" and r.done["components"]["split"]["accepted"] and r.rec.tries == 1


def test_auto_split_retries_are_bounded_and_refused_once_modular(tmp_path, monkeypatch):
    answers: list[int] = []

    def retry_again(client, task_id, answer):
        if answer["status"] == "SUCCEEDED":
            res = client.post(f"/api/vendor/tasks/{task_id}/retry-split")
            answers.append(res.status_code)
            return res.json() if res.status_code == 200 else None
        return None

    run(tmp_path, monkeypatch, PartsMeshy(split_states=("FAILED",)), category="wardrobe", kind="", polls=30, on_poll=retry_again)
    assert answers[:vendor.SPLIT_RETRIES] == [200] * vendor.SPLIT_RETRIES and answers[vendor.SPLIT_RETRIES] == 429
    modular: list[int] = []
    run(tmp_path, monkeypatch, PartsMeshy(), category="wardrobe", kind="",
        on_poll=lambda c, t, a: modular.append(c.post(f"/api/vendor/tasks/{t}/retry-split").status_code) if a["status"] == "SUCCEEDED" else None)
    assert modular[0] == 409
