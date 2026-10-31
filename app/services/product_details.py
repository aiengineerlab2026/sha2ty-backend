"""Workflow B: read a vendor product photo into ProductDetails JSON with the admin-selected vision model,
before Meshy / Astra turn the photo into 3D. The details guide the generation prompt (object, parts,
materials, colours, estimated size); the geometry still comes from the photo. A failure returns None
and generation goes ahead without them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path

import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.product_details import ProductDetails
from app.services import vision
from app.services.generation_prompt import object_description

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "product-details-v1"  # bump with PROMPT or the schema: invalidates the cache
SIZE_RANGE_M = (0.02, 6.0)  # outside this, an estimated size is noise
PROMPT = """You are a furniture and product modeller looking at a vendor's product photo. The vendor says it is {what}.
Describe the main object as JSON so it can be modelled in 3D.
- object_type: what it is, specifically. matches_category: false if the photo shows something else.
- bbox: the main object's boundary; components: every visible part with its bbox, material and colour.
  Boxes are normalised to the image (x_min, y_min, x_max, y_max; 0..1, origin top-left).
- orientation of the camera to the object; proportions width : depth : height with the largest = 1.
- width_m / depth_m / height_m: your estimate in metres from typical sizes ("inferred"), or sizes written on
  the image ("annotated"); null and "unknown" if you cannot tell. Never present a guess as exact.
- materials and colours exactly as seen; hidden: what the photo cannot show (back, underside).
- warnings: anything unclear (several objects, cropped, low quality). Only describe what you can see."""


def _key(image: bytes, model: str, what: str) -> str:
    return hashlib.sha256(image + model.encode() + what.encode() + SCHEMA_VERSION.encode() + PROMPT.encode()).hexdigest()


async def extract_product_details(
    image: bytes, mime: str, category: str | None, title: str | None, settings: Settings, client: httpx.AsyncClient, model: str,
    db: httpx.AsyncClient | None = None,
) -> ProductDetails | None:
    """`client` talks to the model provider; `db` (no provider key in its headers) to the cache table."""
    if not vision.configured(settings, model):
        return None
    what = object_description(category, title)
    key = _key(image, model, what)
    stored = await vision.cache_get(settings, db, key)
    if stored:
        try:
            logger.info("Product details from the cache (%s): no model call", model)
            return ProductDetails.model_validate(stored)
        except ValidationError:
            pass  # an older shape: ask again and overwrite
    path = Path(settings.storage_dir) / "product-details" / f"{key}.json"
    if path.exists():
        try:
            return ProductDetails.model_validate_json(path.read_text())
        except ValidationError:
            path.unlink(missing_ok=True)
    content = [
        {"type": "input_text", "text": PROMPT.format(what=what)},
        {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"},
    ]
    text = await vision.ask_json(content, ProductDetails.model_json_schema(), "product_details", settings, client, model)
    if text is None:
        return None
    try:
        details = clean(ProductDetails.model_validate(json.loads(text)))
    except (ValueError, ValidationError) as exc:
        logger.warning("Product details returned invalid JSON: %s", str(exc)[:300])
        return None
    details.model = model
    await vision.cache_put(settings, db, key, "product", model, details.model_dump(mode="json"))
    logger.info("Product details (%s): %s, %s × %s × %s m (%s), %d parts", model, details.object_type,
                details.width_m, details.depth_m, details.height_m, details.dimension_source, len(details.components))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(details.model_dump_json())
    except OSError as exc:
        logger.warning("Could not cache product details: %s", exc)
    return details


def clean(details: ProductDetails) -> ProductDetails:
    """Deterministic fixes: boxes clamped and ordered, implausible sizes dropped, proportions normalised."""

    def box(b):
        if b is None:
            return None
        x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in b)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return [x0, y0, x1, y1] if x1 - x0 > 0.005 and y1 - y0 > 0.005 else None

    details.bbox = box(details.bbox)
    for c in details.components:
        c.bbox = box(c.bbox)
    for name in ("width_m", "depth_m", "height_m"):
        v = getattr(details, name)
        if v is not None and not SIZE_RANGE_M[0] <= v <= SIZE_RANGE_M[1]:
            setattr(details, name, None)
            details.warnings.append(f"{name} {v} m is implausible and was dropped.")
    if not any((details.width_m, details.depth_m, details.height_m)):
        details.dimension_source = "unknown"
    if details.proportions and max(details.proportions) > 0:
        top = max(details.proportions)
        details.proportions = [round(max(p, 0.0) / top, 3) for p in details.proportions]
    else:
        details.proportions = None
    return details
