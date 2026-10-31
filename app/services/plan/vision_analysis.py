"""Vision-LLM analysis of a floor plan → validated architectural JSON (app/schemas/plan_analysis.py).

The model reads the whole layout (rooms, walls, doors, windows, furniture, scale references) in
normalised image coordinates. It is semantic guidance only: OpenCV measures and refines geometry,
scale_estimator weighs the references. One call per distinct image (cached on disk by image + model
+ prompt/schema version); a failure returns None and the OpenCV pipeline runs alone.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from pathlib import Path

import httpx
from pydantic import ValidationError
from shapely.geometry import LineString, Point, Polygon

from app.core.config import Settings
from app.schemas.plan_analysis import ImageSize, LlmPlan, PlanAnalysis

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "plan-analysis-v2"  # bump when PROMPT or the schema changes: it invalidates the cache
RETRY_DELAY = 2.0  # seconds, doubled per attempt
RETRIES = 1  # one extra attempt on 429 / 5xx / timeouts (each can take minutes)
SNAP = 0.012  # normalised distance under which wall ends are the same corner
PROMPT = """You are an architect reading a 2D floor plan image. Describe the WHOLE layout as JSON.

Coordinates: normalised to the image, origin top-left, x to the right, y downwards, 0..1. Never pixels or metres.

rooms: every enclosed or open space (living, kitchen, dining, bedroom, bathroom, hall, balcony, closet, ...).
  An open living/kitchen area without a separating wall is still two rooms in one space: list both and put
  each in the other's connects_to. A balcony/terrace lies OUTSIDE the apartment's exterior wall.
  If a size is written on the plan, copy it to dimension_text, convert to metres and set dimension_source
  "annotated". Otherwise give your estimate with "inferred", or null with "unknown".
walls: structural walls only (thick lines), as straight segments with both ends. Never furniture or
  cabinet outlines. Mark exterior walls. A shared wall appears once, with both rooms listed. Include a
  balcony's or terrace's outer edge (railing) as exterior walls. Never invent rooms (e.g. a closet) that
  are not drawn; list every room that is drawn, even narrow hallways.
doors: the opening between its jambs, swing, connected rooms ("outside" for the entrance).
windows: the segment along its wall.
furniture: beds, sofas, tables, chairs, wardrobes, cabinets/countertops, appliances, sinks, toilets,
  showers, bathtubs... with a tight bbox, orientation in degrees and an estimated size.
scale_references: objects whose real size is known, as a segment ALONG the measured side: written
  dimension lines (real_m from the text, "annotated"), interior doors (jamb to jamb, ~0.8 m), double beds
  (long side 2.0 m), bathtubs (long side ~1.6 m), kitchen countertop depth (0.6 m). Prefer several.
warnings: anything unclear. Return only what you can see; do not invent rooms."""


def cache_key(image: bytes, model: str) -> str:
    return hashlib.sha256(image + model.encode() + SCHEMA_VERSION.encode() + PROMPT.encode()).hexdigest()


async def analyze_plan(image: bytes, mime: str, size: tuple[int, int], settings: Settings, client: httpx.AsyncClient) -> PlanAnalysis | None:
    model = settings.openai_plan_model
    if not model or not settings.astra_configured:
        return None
    key = cache_key(image, model)
    path = Path(settings.storage_dir) / "plan-analysis" / f"{key}.json"
    if path.exists():
        try:
            return PlanAnalysis.model_validate_json(path.read_text()).model_copy(update={"cached": True})
        except ValidationError:
            path.unlink(missing_ok=True)

    content = [{"type": "input_text", "text": PROMPT}, {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"}]
    plan, issues = await _ask(content, settings, client)
    if plan is None or _serious(plan, issues):
        # Targeted retry: the same image, told exactly what was wrong.
        problems = "; ".join(issues) or "the answer was not valid JSON for the schema"
        retry = [*content, {"type": "input_text", "text": f"Your previous answer had these problems: {problems}. Return the complete JSON again with them fixed."}]
        again, again_issues = await _ask(retry, settings, client)
        if again is not None and (plan is None or len(again.rooms) >= len(plan.rooms)):
            plan, issues = again, again_issues
    if plan is None:
        return None
    analysis = PlanAnalysis(**plan.model_dump(), image=ImageSize(width=size[0], height=size[1]), model=model, schema_version=SCHEMA_VERSION, issues=issues)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(analysis.model_dump_json())
    except OSError as exc:
        logger.warning("Could not cache the plan analysis: %s", exc)
    return analysis


async def _ask(content: list[dict], settings: Settings, client: httpx.AsyncClient) -> tuple[LlmPlan | None, list[str]]:
    payload = {
        "model": settings.openai_plan_model,
        "input": [{"role": "user", "content": content}],
        # strict=False: Pydantic's schema (optional fields, defaults) is a guide, and Pydantic validates.
        "text": {"format": {"type": "json_schema", "name": "floor_plan", "schema": LlmPlan.model_json_schema(), "strict": False}},
    }
    body = await _post(payload, settings, client)
    if body is None:
        return None, []
    try:
        text = next(c["text"] for item in body.get("output", []) if item.get("type") == "message" for c in item.get("content", []) if c.get("type") == "output_text")
        return validate_and_correct(LlmPlan.model_validate(json.loads(text)))
    except (StopIteration, ValueError, ValidationError) as exc:
        logger.warning("Plan analysis returned invalid JSON: %s", str(exc)[:300])
        return None, [f"invalid JSON: {str(exc)[:200]}"]


async def _post(payload: dict, settings: Settings, client: httpx.AsyncClient) -> dict | None:
    for attempt in range(RETRIES + 1):
        started = time.monotonic()
        try:
            response = await client.post("https://api.openai.com/v1/responses", json=payload, timeout=settings.openai_plan_timeout_seconds)
        except httpx.HTTPError as exc:
            logger.warning("Plan analysis request failed (attempt %d): %s", attempt + 1, exc)
        else:
            usage = (response.json().get("usage") if response.headers.get("content-type", "").startswith("application/json") else None) or {}
            logger.info(
                "Plan analysis %s in %.1fs (input %s, output %s tokens)",
                response.status_code, time.monotonic() - started, usage.get("input_tokens"), usage.get("output_tokens"),
            )
            if response.status_code < 400:
                return response.json()
            logger.warning("Plan analysis refused: %s", response.text[:300])
            if (response.status_code != 429 and response.status_code < 500) or any(c in response.text for c in ("insufficient_quota", "credit_balance_exhausted")):
                return None  # a bad request, or no credits left, does not get better by repeating it
        if attempt < RETRIES:
            await asyncio.sleep(RETRY_DELAY * 2**attempt)
    return None


def validate_and_correct(plan: LlmPlan) -> tuple[LlmPlan, list[str]]:
    """Deterministic fixes first (clamp, repair, snap, associate); what can't be fixed is reported."""
    issues: list[str] = []
    clamped = 0

    def clamp(p: list[float]) -> list[float]:
        nonlocal clamped
        q = [min(max(float(v), 0.0), 1.0) for v in p]
        clamped += q != list(p)
        return q

    rooms = []
    for room in plan.rooms:
        poly = Polygon([clamp(p) for p in room.polygon]) if len(room.polygon) >= 3 else None
        if poly is None or poly.area < 1e-4:
            issues.append(f"room {room.id} has no usable polygon")
            continue
        if not poly.is_valid:
            fixed = poly.buffer(0)
            poly = max(getattr(fixed, "geoms", [fixed]), key=lambda g: g.area)
        rooms.append(room.model_copy(update={"polygon": [list(c) for c in list(poly.exterior.coords)[:-1]]}))
    for i, a in enumerate(rooms):
        pa = Polygon(a.polygon)
        for b in rooms[i + 1 :]:
            pb = Polygon(b.polygon)
            if pa.intersection(pb).area > 0.2 * min(pa.area, pb.area):
                issues.append(f"rooms {a.id} and {b.id} overlap")
    ids = {r.id for r in rooms}
    rooms = [r.model_copy(update={"connects_to": [c for c in r.connects_to if c in ids]}) for r in rooms]

    walls = [w.model_copy(update={"start": clamp(w.start), "end": clamp(w.end)}) for w in plan.walls]
    walls = [w for w in walls if LineString([w.start, w.end]).length >= 0.005]
    walls = _snap_walls(walls)
    dangling = sum(
        1 for w in walls for end in (w.start, w.end)
        if not any(o is not w and min(Point(end).distance(Point(o.start)), Point(end).distance(Point(o.end)), LineString([o.start, o.end]).distance(Point(end))) <= SNAP for o in walls)
    )
    if walls and dangling > 0.3 * 2 * len(walls):
        issues.append(f"{dangling} wall ends do not meet another wall")

    lines = {w.id: LineString([w.start, w.end]) for w in walls}

    def nearest_wall(a: list[float], b: list[float]) -> str | None:
        # The segment, not its midpoint: walls often stop at the jambs, so the opening only touches them.
        span = LineString([a, b])
        best = min(lines.items(), key=lambda kv: kv[1].distance(span), default=None)
        return best[0] if best and best[1].distance(span) <= 0.02 else None

    doors = []
    for d in plan.doors:
        d = d.model_copy(update={"start": clamp(d.start), "end": clamp(d.end)})
        wall_id = d.wall_id if d.wall_id in lines else nearest_wall(d.start, d.end)
        if lines and wall_id is None:
            issues.append(f"door {d.id} is not on a wall")
        doors.append(d.model_copy(update={"wall_id": wall_id, "connects": [c for c in d.connects if c in ids or c == "outside"]}))
    windows = []
    for w in plan.windows:
        w = w.model_copy(update={"start": clamp(w.start), "end": clamp(w.end)})
        windows.append(w.model_copy(update={"wall_id": w.wall_id if w.wall_id in lines else nearest_wall(w.start, w.end)}))

    # Connectivity: every room but a balcony/closet should be reachable through a door or an open side.
    linked = {c for d in doors for c in d.connects} | {c for r in rooms for c in r.connects_to} | {r.id for r in rooms if r.connects_to}
    for r in rooms:
        if r.id not in linked and r.type not in ("balcony", "closet"):
            issues.append(f"room {r.id} is not connected to any other room")

    # Scale consistency: an annotated size must have roughly the polygon's proportions.
    for r in rooms:
        if r.dimension_source == "annotated" and r.estimated_width_m and r.estimated_length_m:
            xs, ys = [p[0] for p in r.polygon], [p[1] for p in r.polygon]
            box = (max(xs) - min(xs)) / max(max(ys) - min(ys), 1e-6)
            sizes = r.estimated_width_m / r.estimated_length_m
            # ponytail: normalised aspect ignores the image aspect ratio; only gross disagreements are flagged.
            if min(abs(box / sizes - 1), abs(box * sizes - 1)) > 1.0:
                issues.append(f"room {r.id}: written size {r.dimension_text or ''} does not match its shape")

    furniture = [f.model_copy(update={"bbox": [*clamp(f.bbox[:2]), *clamp(f.bbox[2:])]}) for f in plan.furniture]
    furniture = [f for f in furniture if f.bbox[2] > f.bbox[0] and f.bbox[3] > f.bbox[1]]
    refs = [s.model_copy(update={"start": clamp(s.start), "end": clamp(s.end)}) for s in plan.scale_references]
    refs = [s for s in refs if LineString([s.start, s.end]).length > 0.003]
    if clamped:
        issues.append(f"{clamped} coordinates were outside the image and were clamped")
    fixed = LlmPlan(rooms=rooms, walls=walls, doors=doors, windows=windows, furniture=furniture, scale_references=refs, warnings=plan.warnings)
    return fixed, list(dict.fromkeys(issues))


def _serious(plan: LlmPlan, issues: list[str]) -> bool:
    return not plan.rooms or sum("no usable polygon" in i for i in issues) > len(plan.rooms) / 2


def _snap_walls(walls: list) -> list:
    """Merge wall ends closer than SNAP into one corner (their average)."""
    ends = [list(p) for w in walls for p in (w.start, w.end)]
    for i, p in enumerate(ends):
        group = [q for q in ends if abs(q[0] - p[0]) <= SNAP and abs(q[1] - p[1]) <= SNAP]
        if len(group) > 1:
            ends[i] = [sum(q[0] for q in group) / len(group), sum(q[1] for q in group) / len(group)]
    return [w.model_copy(update={"start": ends[2 * i], "end": ends[2 * i + 1]}) for i, w in enumerate(walls)]
