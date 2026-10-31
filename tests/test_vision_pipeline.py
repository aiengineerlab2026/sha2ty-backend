"""Regression: vision-LLM analysis → OpenCV refinement → 2D layout → 3D/GLB.

The plan (tests/fixtures/open_plan_apartment.png, drawn by make_open_plan.py) has the layout that used
to collapse into one region: balcony left, open living + kitchen, bathroom upper-right, hallway
middle-right, bedroom bottom-right, thin walls, furniture, no dimensions. Model answers are fixtures
(hand-written + one recorded from the live model), so no test pays for an API call. RUN_LIVE_LLM=1
runs the live end-to-end test too.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path

import httpx
import pytest
import trimesh
from shapely.geometry import LineString, Point, Polygon

from app.core.config import Settings, get_settings
from app.schemas.plan_analysis import LlmPlan, PlanAnalysis
from app.services.plan import image_processor, vision_analysis
from app.services.plan.dimension_ocr import OcrBox
from app.services.plan.geometry_builder import build_room_glb
from app.services.plan.vision_analysis import analyze_plan, validate_and_correct

FIXTURES = Path(__file__).parent / "fixtures"
IMAGE = FIXTURES / "open_plan_apartment.png"
ANSWERS = ["open_plan_apartment.analysis.json", "open_plan_apartment.recorded-analysis.json"]


def no_ocr(_image):
    return [], []


def convert(answer: str, monkeypatch) -> dict:
    monkeypatch.setattr(image_processor, "run_ocr", no_ocr)  # names must come from the model
    analysis = PlanAnalysis.model_validate_json((FIXTURES / answer).read_text())
    return image_processor.plan_from_image(IMAGE.read_bytes(), analysis=analysis)


def centre(room: dict) -> Point:
    return Polygon(room["polygon"]).centroid


@pytest.fixture(params=ANSWERS)
def plan(request, monkeypatch):
    return convert(request.param, monkeypatch)


def test_all_functional_spaces_are_built_where_the_drawing_has_them(plan):
    assert plan["geometry_source"] == "vision"
    assert plan["validation"]["ok"], plan["validation"]["errors"]
    rooms = {r["kind"]: r for r in plan["rooms"]}
    assert {"living", "kitchen", "bedroom", "bathroom", "hall", "patio"} <= rooms.keys()
    assert not any("closet" in r["name"].lower() for r in plan["rooms"])  # none on the drawing
    xs = [x for r in plan["rooms"] for x, _ in r["polygon"]]
    zs = [z for r in plan["rooms"] for _, z in r["polygon"]]
    mid_x, mid_z = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    bed, bath, hall, balcony = (centre(rooms[k]) for k in ("bedroom", "bathroom", "hall", "patio"))
    assert bed.x > mid_x and bed.y > mid_z  # bottom-right (plan z grows downwards)
    assert bath.x > mid_x and bath.y < mid_z  # upper-right
    assert bath.y < hall.y < bed.y and hall.x > mid_x  # hallway between them, middle-right
    assert balcony.x < min(centre(rooms[k]).x for k in ("living", "kitchen"))  # far left
    assert not Polygon(plan["exterior"]).contains(balcony)  # outside the apartment walls
    assert Polygon(rooms["hall"]["polygon"]).area > 1.0  # the narrow hallway is kept (m²)


def test_open_plan_and_connectivity(plan):
    by_id = {r["id"]: r["kind"] for r in plan["rooms"]}
    via = {frozenset((by_id[e["a"]], by_id[e["b"]])): e["via"] for e in plan["adjacency"]}
    assert via[frozenset(("living", "kitchen"))] == "opening"  # no wall between them
    assert via[frozenset(("hall", "bedroom"))] == "door"
    assert via[frozenset(("hall", "bathroom"))] == "door"
    assert frozenset(("living", "patio")) in via


def test_doors_are_openings_and_furniture_is_not_wall(plan):
    full = [w for w in plan["walls"] if not w.get("elevation")]  # floor-standing wall bodies
    for door in plan["doors"]:
        mid = LineString([door["start"], door["end"]]).interpolate(0.5, normalized=True)
        assert all(LineString([w["start"], w["end"]]).distance(mid) > w["thickness"] / 2 for w in full), "a door is filled with wall"
    bed = next(f for f in plan["fixtures"] if f["type"] == "bed")
    w, d, _ = bed["size"]
    assert 1.0 < min(w, d) < 2.2 and 1.4 < max(w, d) < 2.6  # bed-sized at the estimated scale (approximate)
    walls = [LineString([w["start"], w["end"]]) for w in full]
    footprint = Polygon(plan["exterior"]).buffer(-0.3)
    for f in plan["fixtures"]:
        if footprint.contains(Point(f["position"])):
            assert not any(line.within(Point(f["position"]).buffer(min(f["size"][:2]) / 2)) for line in walls)


def test_glb_has_named_meshes_for_every_space(plan):
    scene = trimesh.load(io.BytesIO(build_room_glb(plan)), file_type="glb")
    names = set(scene.graph.nodes_geometry)
    for slug in ("living_room", "kitchen", "bedroom", "bathroom", "hall", "balcony"):
        assert f"room_{slug}_1_floor" in names
    assert any(n.startswith("wall_bedroom_1_") for n in names)
    assert "furniture_bedroom_1_bed" in names
    assert any(n.startswith("door_") for n in names) and any(n.startswith("window_") for n in names)
    width, _, depth = scene.extents
    assert 7 < width < 14 and 4 < depth < 8  # a whole flat, not one rectangle


def test_scale_is_estimated_and_labelled(plan):
    scale = plan["scale"]
    assert scale["estimated"] and scale["confidence"] < 0.8
    assert "Dimensions automatically estimated from your floor plan. Measurements are approximate." in plan["warnings"]
    assert plan["analysis"]["scale"]["pixels_per_meter"] > 0  # written back for later editing


def test_opencv_alone_reports_failure_instead_of_a_fake_room(monkeypatch):
    monkeypatch.setattr(image_processor, "run_ocr", no_ocr)
    payload = image_processor.plan_from_image(IMAGE.read_bytes())
    assert not payload["validation"]["ok"]  # diagnosed, not presented as a finished apartment
    assert not any(r["name"] == "Closet" for r in payload["rooms"])


def test_good_opencv_plans_keep_opencv_geometry(monkeypatch):
    rows = json.loads((FIXTURES / "three_bedroom_apartment.ocr.json").read_text())
    monkeypatch.setattr(image_processor, "run_ocr", lambda _i: ([OcrBox(**r) for r in rows], []))
    analysis = PlanAnalysis.model_validate_json((FIXTURES / "three_bedroom_apartment.analysis.json").read_text())
    payload = image_processor.plan_from_image((FIXTURES / "three_bedroom_apartment.png").read_bytes(), analysis=analysis)
    assert payload["geometry_source"] == "opencv" and payload["opencv_agreement"] >= 0.7
    assert abs(payload["scale"]["px_per_ft"] / 16.2 - 1) < 0.04


# ── vision_analysis: validation, retry, cache ───────────────────────────────────────────────


def test_correction_clamps_repairs_and_reports():
    raw = {
        "rooms": [
            {"id": "a", "name": "Living", "type": "living", "polygon": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]], "connects_to": ["b", "ghost"]},
            {"id": "b", "name": "Kitchen", "type": "kitchen", "polygon": [[0.3, 0], [1.2, 0], [1.2, 0.5], [0.3, 0.5]]},
            {"id": "c", "name": "Bad", "type": "other", "polygon": [[0.1, 0.1], [0.2, 0.1]]},
        ],
        "walls": [{"id": "w", "start": [0, 0], "end": [0, 0.001]}],
    }
    plan, issues = validate_and_correct(LlmPlan.model_validate(raw))
    assert [r.id for r in plan.rooms] == ["a", "b"] and plan.rooms[0].connects_to == ["b"]
    assert max(x for x, _ in plan.rooms[1].polygon) == 1.0 and not plan.walls
    assert any("overlap" in i for i in issues) and any("clamped" in i for i in issues) and any("room c" in i for i in issues)


def reply(plan: dict) -> httpx.Response:
    return httpx.Response(200, json={"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(plan)}]}], "usage": {"input_tokens": 1, "output_tokens": 2}})


def run_analysis(handler, tmp_path) -> tuple[PlanAnalysis | None, int]:
    calls = {"n": 0}

    def counted(request):
        calls["n"] += 1
        return handler(request, calls["n"])

    settings = Settings(openai_api_key="sk-test", openai_plan_model="test-model", storage_dir=tmp_path)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(counted)) as client:
            return await analyze_plan(b"image-bytes", "image/png", (100, 50), settings, client)

    return asyncio.run(go()), calls


def test_targeted_retry_then_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_analysis, "RETRY_DELAY", 0)
    answer = json.loads((FIXTURES / "open_plan_apartment.analysis.json").read_text())
    good = {k: answer[k] for k in LlmPlan.model_fields}
    seen = []

    def handler(request, n):
        seen.append(json.loads(request.content)["input"][0]["content"])
        return reply({"rooms": []}) if n == 1 else reply(good)

    analysis, calls = run_analysis(handler, tmp_path)
    assert calls["n"] == 2 and len(analysis.rooms) == 6  # empty answer → one retry that names the problem
    assert "previous answer had these problems" in seen[1][-1]["text"]
    again, calls = run_analysis(handler, tmp_path)
    assert calls["n"] == 0 and again.cached and len(again.rooms) == 6  # same image + model: no new call


def test_failures_fall_back_to_opencv(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_analysis, "RETRY_DELAY", 0)
    analysis, calls = run_analysis(lambda _r, _n: httpx.Response(503), tmp_path)
    assert analysis is None and calls["n"] == 4  # (1 + 1 retry) × (first ask + targeted retry)


@pytest.mark.skipif(os.environ.get("RUN_LIVE_LLM") != "1", reason="set RUN_LIVE_LLM=1 to call the real vision model")
def test_live_model_reads_the_regression_plan(tmp_path, monkeypatch):
    settings = get_settings().model_copy(update={"storage_dir": tmp_path})

    async def go():
        key = settings.openai_api_key.get_secret_value()
        async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}) as client:
            return await analyze_plan(IMAGE.read_bytes(), "image/png", (1000, 600), settings, client)

    analysis = asyncio.run(go())
    assert analysis is not None
    monkeypatch.setattr(image_processor, "run_ocr", no_ocr)
    payload = image_processor.plan_from_image(IMAGE.read_bytes(), analysis=analysis)
    assert payload["validation"]["ok"] and {"living", "kitchen", "bedroom", "bathroom", "hall", "patio"} <= {r["kind"] for r in payload["rooms"]}


def test_upload_streams_progress_then_the_full_plan(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import plans
    from app.main import app

    analysis = PlanAnalysis.model_validate_json((FIXTURES / "open_plan_apartment.analysis.json").read_text())

    async def fake_analyze(*_args):
        return analysis

    monkeypatch.setattr(plans, "analyze_plan", fake_analyze)
    monkeypatch.setattr(image_processor, "run_ocr", no_ocr)
    with TestClient(app) as client:
        response = client.post("/api/plans/from-image", files={"image": ("plan.png", IMAGE.read_bytes(), "image/png")}, data={"height": "2.7"})
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    stages = [line["stage"] for line in lines]
    assert stages[0] == "Analyzing" and stages[-1] == "Done"
    assert {"Extracting rooms", "Estimating dimensions", "Generating 3D"} <= set(stages)
    percents = [line["progress"] for line in lines]
    assert percents == sorted(percents) and percents[-1] == 100
    plan = lines[-1]["plan"]
    assert plan["validation"]["ok"] and plan["height"] == 2.7 and len(plan["rooms"]) == 6
