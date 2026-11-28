"""The one vision call per photo: a structured furniture specification, validated and cached.

Before, generation read the same photo twice (product_details: the listing and size; mesh_analysis: the
components), then asked the vision model again to group and to name the generated meshes. Now one call returns
everything both models need, and everything after it is deterministic:

  photo ─ vision model (1 call, cached) ─▶ FurnitureSpec
                                            ├─ details     (ProductDetails: listing, category, estimated size)
                                            ├─ analysis    (MeshAnalysis: components, boxes, materials, motion)
                                            ├─ loose_objects / needs_cleaning (whether the photo must be cleaned)
                                            └─ design_features
  details + analysis ─▶ Meshy / Astra prompt (visual model) and services/structural.py (structural model)

Sizes are estimates unless written on the image (`dimension_source` "annotated"); a single photo never gives
exact measurements, and the spec says so.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.config import Settings
from app.schemas.product_details import Component as DetailComponent
from app.schemas.product_details import ProductDetails
from app.services import mesh_analysis, vision
from app.services.generation_prompt import object_description
from app.services.product_details import LISTING, Lang, clean

logger = logging.getLogger(__name__)

SPEC_VERSION = "furniture-spec-v5"  # bump with PROMPT or the schema: invalidates the cache

PROMPT = """You are a furniture modeller preparing a vendor's product photo for two 3D models: a visual one that must
look exactly like the photo, and a structural one a carpenter can build from. The vendor says it is {what}.
Return ONE JSON object.

The product:
- object_type: what it is, specifically. matches_category: false if the photo shows something else.
- category: its catalogue type: furniture (sofas, beds, chairs, storage), table, wardrobe, door, tv, lighting
  (floor / table lamps), ceiling_light, air_conditioner or wall_art.
- name and description are the shop listing. {listing} No prices, brands or sizes you cannot see.
  Every other field stays in English.
- bbox: the main object's boundary in the image (x_min, y_min, x_max, y_max; 0..1, origin top-left).
- orientation of the camera; proportions width : depth : height with the largest = 1.
- width_m / depth_m / height_m: your ESTIMATE in metres from typical sizes ("inferred"), or sizes written on the
  image ("annotated"); null and "unknown" if you cannot tell. Never present a guess as exact.
- materials, colors: the whole product's. hidden: what the photo cannot show. confidence: 0..1 overall.
- design_features: what makes this design itself (curved headboard, mirrored doors, fluted fronts, tapered legs,
  brass handles, tufted upholstery…), so the 3D model keeps them.

What is not the product:
- loose_objects: every object placed on, in or against it that is NOT part of it and can be lifted off (books,
  phones, vases, plants, lamps, decorations, clothes, boxes, bedding, loose pillows, mattresses, the contents of
  shelves and drawers…). Empty when there is nothing. Never list a permanent part (built-in upholstery, a fixed
  mirror, handles, glass panels).

structure: the meaningful PHYSICAL components the models should have (doors, drawers, shelves, legs, handles,
side panels, top, base, back, frame, headboard, upholstery… whatever this object really has), plus the main body.
- Only what this photo shows. A part you cannot see but that must exist (a hidden back panel, a fourth leg) is
  "inferred". Never add a component that is neither visible nor structurally certain. No loose objects here.
- One entry per physical part: two doors are door_left and door_right, not "doors". No surface details (grain,
  stitching, shadows); ornaments are role "decorative".
- box: the part's rectangle as fractions of the WHOLE OBJECT's box in the photo (not the image), x0 y0 x1 y1.
- dimensions: its size as fractions (0..1] of the whole object's width, height and depth; null where not visible.
- color: its real colour as #RRGGBB (ignore highlights and shadows). materialClass: solid_wood, wood_veneer,
  painted_wood, mdf, laminate, glass, mirror, metal, fabric, leather, stone, plastic or other; material and
  texture in words (e.g. "oak", "wood grain"); finish (matte, satin, gloss, brushed…).
- confidence: 0..1 per part: low when its size or presence is a guess.
- decoration: none, fluted (vertical ribs), grooved, raised_panel, recessed_panel, arched, carved, upholstered or
  other. complexity: "complex" only for a part whose SHAPE is organic or sculpted (floral carving, sculpted leg,
  tufted or shaped upholstery); flat panels, doors, shelves, boxes, regular grooves and flutes are "regular" even when
  decorated. Never call the whole object complex because one part is.
- relations: how parts connect (door_left attached_to body, handle_1 attached_to door_left, shelf_1 inside body…).
- kind: what each part physically is (side_panel, top_panel, crown_trim, plinth, back_panel, door, sliding_door,
  drawer_front, door_frame, mirror, glass, moulding, handle, shelf, leg, tabletop, headboard…). A door that is a frame
  holding a mirror or glass is a LOGICAL assembly: list it with logical true (no solid of its own), then its parts
  with assembly = that door's id: its door_frame (geometry "frame"), its mirror or glass (geometry "pane"). Never list
  the same physical part twice under two names.
- quad: the corners of the part's visible FRONT face in the IMAGE (fractions of the image, not of the object), in
  the order top-left, top-right, bottom-right, bottom-left, following the face's real edges (in perspective they are
  not a rectangle). sideQuad: a visible side panel's outer face the same way.
- mouldings: every rectangular moulding on a part's front face, one entry each (two panels of trim on a door are two
  entries), as fractions of that face, raised or recessed, with the trim's width.
- motion: sliding doors (two overlapping doors on tracks, no hinges) are kind sliding_door with motion "slide"
  (direction left / right); hinged doors "rotate" with the hinge side. If the photo does not show which, give the
  motion with confidence below 0.4 and say why: never assume a door turns because it is a door.
- frontQuad: the whole object's front face corners in the image; viewYaw: the camera's side angle.
- exclusions: every region on the furniture that is not its material (the room reflected in a mirror, a cast shadow,
  an object standing in front), as image quads.
- motion: only for parts that really move in use (hinged doors, drawers, sliding doors, flaps, lids); everything
  else is type "none". Never invent a movable part. Doors hinge on the side away from the handle.
- interior: only for storage pieces whose inside is hidden: simple, typical items with a count.
- At most {limit} components, the main body first.
- warnings: anything unclear (several objects, cropped, low quality)."""


def schema() -> dict[str, Any]:
    """ProductDetails' fields (without its image-box component list: `structure` lists the parts once) plus the
    component analysis, the loose objects and the design features."""
    out = copy.deepcopy(ProductDetails.model_json_schema())
    props = out["properties"]
    for drop in ("components", "model"):
        props.pop(drop, None)
    props["structure"] = mesh_analysis.SCHEMA
    props["loose_objects"] = {"type": "array", "items": {"type": "string"}, "description": "Objects on or in it that are not part of it"}
    props["design_features"] = {"type": "array", "items": {"type": "string"}}
    out["required"] = sorted({*out.get("required", []), "structure", "loose_objects", "design_features", "width_m", "depth_m", "height_m", "dimension_source", "confidence"})
    return out


class FurnitureSpec(BaseModel):
    version: str = SPEC_VERSION
    details: ProductDetails
    analysis: mesh_analysis.MeshAnalysis | None = None
    loose_objects: list[str] = Field(default_factory=list)
    design_features: list[str] = Field(default_factory=list)

    @property
    def needs_cleaning(self) -> bool:
        """Only a photo with something on or in the product is cleaned; a clean product photo goes to 3D as it is."""
        return bool(self.loose_objects)


def parse(text: str | None, lang: Lang = "ar") -> FurnitureSpec | None:
    if not text:
        return None
    try:
        raw = json.loads(text)
        structure = raw.pop("structure", None)
        loose = [str(x).strip()[:60] for x in raw.pop("loose_objects", None) or [] if str(x).strip()][:20]
        features = [str(x).strip()[:80] for x in raw.pop("design_features", None) or [] if str(x).strip()][:12]
        details = clean(ProductDetails.model_validate(raw), lang)
    except (ValueError, TypeError, AttributeError, ValidationError) as exc:
        logger.warning("Furniture spec returned invalid JSON: %s", str(exc)[:300])
        return None
    analysis = mesh_analysis.parse(json.dumps(structure)) if isinstance(structure, dict) else None
    if analysis:
        # The listing's "visible parts" come from the one component list (no second list to keep in step).
        details.components = [DetailComponent(name=c.name, material=c.material or None, color=c.color, confidence=c.confidence)
                              for c in analysis.components if c.role != "decorative"][:12]
    else:
        details.warnings.append("The components could not be read: the structural model is not built from this photo.")
    return FurnitureSpec(details=details, analysis=analysis, loose_objects=loose, design_features=features)


def _key(image: bytes, model: str, what: str, lang: str) -> str:
    return hashlib.sha256(image + model.encode() + what.encode() + lang.encode() + SPEC_VERSION.encode() + PROMPT.encode()).hexdigest()


async def read(
    image: bytes, mime: str, category: str | None, title: str | None, settings: Settings, client: httpx.AsyncClient, model: str,
    db: httpx.AsyncClient | None = None, kind: str | None = None, lang: Lang = "ar",
) -> FurnitureSpec | None:
    """The photo's spec: from the cache (database, then local file) when this photo, model, type and language were read
    before, else one vision call. None when no vision model is configured or it gave nothing usable."""
    if not vision.configured(settings, model):
        return None
    what = object_description(category, title, kind)
    key = _key(image, model, what, lang)
    path = Path(settings.storage_dir) / "furniture-spec" / f"{key}.json"
    stored = await vision.cache_get(settings, db, key)
    if not stored and path.exists():
        try:
            stored = json.loads(path.read_text())
        except (OSError, ValueError):
            stored = None
    if stored:
        try:
            spec = FurnitureSpec.model_validate(stored)
            logger.info("Furniture spec from the cache (%s): no vision call", model)
            return spec
        except ValidationError:
            pass  # an older shape: read again and overwrite
    content = [
        {"type": "input_text", "text": PROMPT.format(what=what, listing=LISTING[lang], limit=mesh_analysis.MAX_COMPONENTS)
         + (f"\nThe vendor says: {kind.strip()}" if kind and kind.strip() else "")},
        {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"},
    ]
    spec = parse(await vision.ask_json(content, schema(), "furniture_spec", settings, client, model), lang)
    if spec is None:
        return None
    spec.details.model = model
    dumped = spec.model_dump(mode="json")
    await vision.cache_put(settings, db, key, "furniture_spec", model, dumped)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dumped))
    except OSError as exc:
        logger.warning("Could not cache the furniture spec: %s", exc)
    d = spec.details
    logger.info("Furniture spec (%s): %s, %s × %s × %s m (%s), %d components, loose objects %s", model, d.object_type,
                d.width_m, d.depth_m, d.height_m, d.dimension_source, len(spec.analysis.components) if spec.analysis else 0, spec.loose_objects)
    return spec
