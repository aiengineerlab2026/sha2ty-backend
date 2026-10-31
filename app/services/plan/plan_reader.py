"""GPT reads the floor plan first: room names and the measurements written on it (or its best estimate).

The reading never draws geometry (positions from a language model are not reliable); it only feeds
scale references, which scale_estimator weighs against doors and furniture. Any failure returns None
and the OpenCV pipeline carries on alone.
"""

from __future__ import annotations

import base64
import json
import logging

import httpx
import numpy as np

from app.core.config import Settings
from app.services.plan.dimension_ocr import parse_room_label

logger = logging.getLogger(__name__)

PROMPT = (
    "You are reading an architectural 2D floor plan. List every room you can see with its name and its size in metres. "
    "If the size is written on the plan (e.g. 13'x12', 4.0 x 3.6, 400x360 cm), convert it to metres and set printed=true. "
    "If nothing is written, estimate a realistic size from doors, furniture and proportions and set printed=false. "
    "width_m is the room's horizontal extent on the drawing, length_m its vertical extent. "
    "Also give the overall outside size of the apartment when it is written (else null)."
)

_ROOM = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "width_m", "length_m", "printed"],
    "properties": {"name": {"type": "string"}, "width_m": {"type": "number"}, "length_m": {"type": "number"}, "printed": {"type": "boolean"}},
}
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rooms", "overall", "unit_on_plan"],
    "properties": {
        "rooms": {"type": "array", "items": _ROOM},
        "overall": {
            "anyOf": [
                {"type": "null"},
                {"type": "object", "additionalProperties": False, "required": ["width_m", "length_m"], "properties": {"width_m": {"type": "number"}, "length_m": {"type": "number"}}},
            ]
        },
        "unit_on_plan": {"type": "string", "enum": ["metric", "imperial", "none"]},
    },
}


async def read_plan(image: bytes, mime: str, settings: Settings, client: httpx.AsyncClient) -> dict | None:
    if not settings.openai_plan_model or not settings.astra_configured:
        return None
    payload = {
        "model": settings.openai_plan_model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"},
                ],
            }
        ],
        "text": {"format": {"type": "json_schema", "name": "floor_plan_reading", "schema": SCHEMA, "strict": True}},
    }
    try:
        response = await client.post("https://api.openai.com/v1/responses", json=payload, timeout=settings.openai_plan_timeout_seconds)
        response.raise_for_status()
        text = next(
            c["text"] for item in response.json().get("output", []) if item.get("type") == "message" for c in item.get("content", []) if c.get("type") == "output_text"
        )
        reading = json.loads(text)
    except (httpx.HTTPError, StopIteration, ValueError, KeyError) as exc:
        logger.warning("GPT plan reading failed, converting with OpenCV only: %s", exc)
        return None
    reading["rooms"] = [r for r in reading.get("rooms", []) if 0.5 <= r["width_m"] <= 40 and 0.5 <= r["length_m"] <= 40]
    return reading


def reading_references(reading: dict | None, zones: list[dict], footprint: list[list[float]]) -> list[dict]:
    """One metres-per-pixel candidate per GPT room matched to a detected zone (same kind, best shape), plus the overall size."""
    if not reading:
        return []
    refs, used = [], set()
    for room in reading.get("rooms", []):
        named = parse_room_label(room["name"])
        kind = named[1] if named else None
        best = None
        for z in zones:
            m = z.get("measure_px") or {}
            if z["id"] in used or not m.get("width") or not m.get("depth") or (kind and z.get("kind") != kind):
                continue
            # Either orientation; the one where both axes agree is the match.
            for a, b in ((room["width_m"], room["length_m"]), (room["length_m"], room["width_m"])):
                mx, my = a / m["width"], b / m["depth"]
                spread = abs(mx - my) / max(mx, my)
                if spread < 0.2 and (best is None or spread < best[0]):
                    best = (spread, z["id"], (mx + my) / 2)
        if best and kind:  # ponytail: unnamed rooms aren't matched; shape alone is too ambiguous
            used.add(best[1])
            printed = room["printed"]
            refs.append({"source": "gpt_printed" if printed else "gpt", "label": f"GPT: {room['name']} {room['width_m']:.2f} × {room['length_m']:.2f} m", "mpp": best[2], "weight": 2.0 if printed else 0.8})
    overall = reading.get("overall")
    if overall and footprint:
        pts = np.array(footprint, dtype=float)
        w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        options = [(overall["width_m"] / w, overall["length_m"] / h), (overall["length_m"] / w, overall["width_m"] / h)]
        mx, my = min(options, key=lambda o: abs(o[0] - o[1]) / max(o))
        if abs(mx - my) / max(mx, my) < 0.15:
            refs.append({"source": "gpt_printed", "label": f"GPT: overall {overall['width_m']:.2f} × {overall['length_m']:.2f} m", "mpp": (mx + my) / 2, "weight": 2.0})
    return refs
