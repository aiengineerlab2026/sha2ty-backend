"""Optional vision review of a rebuilt model (settings.hybrid_vision_review, off by default).

When the deterministic correction loop (hybrid.build_corrected) leaves discrepancies with the photo, the admin's
configured vision model is shown the photo and the model rendered from the photo's estimated camera, and asked for
STRUCTURED corrections only (the same operations the loop uses: remove an invented trim, a front's extent, a plinth
height, a mirror frame border, a colour). Each is validated (known operation, known component, values in range) and
applied through hybrid.refine, which keeps the result only when it matches the photo better and breaks nothing.
At most one call per generation, only when something is left to explain; never a new 3D model.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

from fastapi import Request
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {"corrections": {"type": "array", "items": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["remove", "front_extent", "plinth_height", "frame_border", "recolour"]},
        "component": {"type": "string", "description": "A component id from the list"},
        "kind": {"type": "string"},
        "rect": {"type": "array", "items": {"type": "number"}, "description": "front_extent: [left, top, right, bottom] as fractions of the straightened front"},
        "height": {"type": "number", "description": "plinth_height: a fraction of the front's height"},
        "border": {"type": "number", "description": "frame_border: metres"},
        "color": {"type": "string", "description": "recolour: #RRGGBB as the photo shows the finish"},
        "reason": {"type": "string"}}, "required": ["op", "component", "reason"]}}},
    "required": ["corrections"],
}

PROMPT = """The first image is a photo of a piece of furniture; the second is a 3D model rebuilt from it, rendered from
about the same camera. The model's parts: {parts}
Differences an automatic check could not settle: {issues}
Give only corrections the PHOTO clearly supports, as structured operations on the listed parts:
- remove: a trim (crown_trim or plinth) the model has but the photo does not show.
- front_extent: a door or drawer whose top / bottom / sides differ from the photo (fractions of the front face).
- plinth_height, frame_border (a mirror door's frame width in metres), recolour (#RRGGBB of the finish).
Never invent parts or decorations. An empty list when the model already matches."""


def validate(raw: str | None, ids: set[str]) -> list[dict[str, Any]]:
    """The model's answer reduced to well-formed corrections on known parts (anything else is dropped)."""
    try:
        items = (json.loads(raw or "{}") or {}).get("corrections") or []
    except (ValueError, AttributeError):
        return []
    out = []
    for c in items[:12]:
        if not isinstance(c, dict) or c.get("component") not in ids:
            continue
        op = c.get("op")
        ok = (op == "remove" and c.get("kind") in ("crown_trim", "plinth")) \
            or (op == "front_extent" and isinstance(c.get("rect"), list) and len(c["rect"]) == 4 and all(isinstance(v, (int, float)) and -0.05 <= v <= 1.05 for v in c["rect"])
                and c["rect"][0] < c["rect"][2] and c["rect"][1] < c["rect"][3]) \
            or (op == "plinth_height" and isinstance(c.get("height"), (int, float)) and 0 < c["height"] < 0.3) \
            or (op == "frame_border" and isinstance(c.get("border"), (int, float)) and 0 <= c["border"] <= 0.15) \
            or (op == "recolour" and isinstance(c.get("color"), str) and len(c["color"]) == 7 and c["color"].startswith("#"))
        if ok:
            out.append({k: v for k, v in c.items() if k in ("op", "component", "kind", "rect", "height", "border", "color", "reason")})
    return out


async def maybe_review(request: Request, settings, photo: bytes, mime: str, spec_doc: dict[str, Any], category: str | None, kind: str | None,
                       mode: str, size, glb: bytes, doc: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """The reviewed model, or the one given (unchanged) when the review is off, not needed or not an improvement."""
    from app.services import hybrid, vision, visual_check

    issues = (doc.get("visualCheck") or {}).get("issues") or []
    if not settings.hybrid_vision_review or not issues:
        doc["visionReview"] = {"called": False, "reason": "off" if not settings.hybrid_vision_review else "nothing left to explain"}
        return glb, doc
    model = await vision.current_model(settings, request.app.state.auth_http)
    if not vision.configured(settings, model):
        doc["visionReview"] = {"called": False, "reason": "no vision model configured"}
        return glb, doc
    analysis = doc.get("analysis") or spec_doc["analysis"]
    shot = await run_in_threadpool(lambda: visual_check.views(glb, analysis, doc["sizeM"])["reference"])
    buf = io.BytesIO()
    shot.save(buf, "PNG")
    parts = ", ".join(f"{c['id']} ({c.get('kind')})" for c in doc.get("components") or [])
    content = [{"type": "input_text", "text": PROMPT.format(parts=parts, issues=" ".join(issues))},
               {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(photo).decode('ascii')}"},
               {"type": "input_image", "image_url": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"}]
    try:
        raw = await vision.ask_json(content, SCHEMA, "visual_review", settings, vision.client_for(request.app.state, model), model)
    except Exception as exc:  # optional: the model stands as it is
        logger.warning("Visual review failed: %s", str(exc)[:200])
        doc["visionReview"] = {"called": True, "error": str(exc)[:200]}
        return glb, doc
    ids = {c["id"] for c in analysis.get("components") or []} | {c["id"] for c in doc.get("components") or []}
    corrections = validate(raw, ids)
    current = (glb, doc, doc.get("visualCheck") or {})
    (glb2, doc2, vc2), _, entry = await run_in_threadpool(hybrid.refine, photo, {**spec_doc, "analysis": analysis}, category, kind, mode, size,
                                                         current, corrections, "vision review")
    out = doc2 if entry["accepted"] else doc
    out["visualCheck"] = vc2 if entry["accepted"] else current[2]
    out["visualCorrection"] = {**(doc.get("visualCorrection") or {}), "rounds": [*((doc.get("visualCorrection") or {}).get("rounds") or []), entry]}
    out["visionReview"] = {"called": True, "model": model, "proposed": len(corrections), "accepted": entry["accepted"], "reason": entry.get("reason")}
    return (glb2 if entry["accepted"] else glb), out
