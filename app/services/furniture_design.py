"""Admin "Generate furniture": GPT Sol designs a random, attractive piece of a given type as a parametric spec
(sizes, shapes, a material per part, an Arabic listing); the browser builds the modular GLB from it
(frontend lib/vendor/furnitureBuilder.ts), so every part is its own named mesh. GPT Sol then turns a render
of that model into product photos (image_match.ImageMatcher). Nothing is saved here.
"""

from __future__ import annotations

import asyncio
import io
import json
import random
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from app.core.config import Settings
from app.services import vision
from app.services.image_match import ImageMatcher, ImageMatchError
from app.services.product_details import in_language

Kind = Literal["bed", "sofa", "chair", "dining_table"]
Material = Literal["fabric", "velvet", "leather", "boucle", "wood", "metal", "brass", "marble", "lacquer"]

# The modules of each type, in the GLB's mesh order (the builder makes them in this order).
PARTS: dict[str, list[str]] = {
    "bed": ["Headboard", "Mattress", "Legs", "Sides", "Headboard Upholstery"],
    "dining_table": ["Table Head", "Legs"],
    "chair": ["Chair Base", "Legs", "Chair Back"],
    "sofa": ["Sofa Base", "Sofa Legs", "Sofa Back", "Sofa Back Upholstery"],
}
# Typical overall sizes (metres): width, depth, height ranges the design must stay inside.
SIZES = {
    "bed": ((1.5, 2.0), (2.05, 2.3), (0.95, 1.4)),
    "sofa": ((1.8, 2.6), (0.85, 1.0), (0.75, 0.95)),
    "chair": ((0.44, 0.58), (0.5, 0.6), (0.8, 1.0)),
    "dining_table": ((1.4, 2.2), (0.85, 1.05), (0.74, 0.77)),
}
NAMES = {"bed": "bed", "sofa": "sofa", "chair": "dining chair", "dining_table": "dining table"}
# Modern, contemporary looks only (the admin's reference: an upholstered bed with a tall channel-tufted headboard
# and a box base in greige fabric). One is picked at random per design.
STYLES = [
    "modern upholstered in warm greige linen, tall channel-tufted headboard, clean lines",
    "contemporary soft taupe bouclé with rounded, generous forms",
    "modern minimal in light oatmeal fabric with slim pale-oak details",
    "modern luxe in mushroom velvet with thin brushed-brass accents",
    "warm modern in sand-beige fabric with walnut details",
    "Japandi modern in stone-grey linen with natural ash wood",
    "quiet-luxury modern in ivory bouclé and travertine tones",
    "modern in charcoal-grey textured fabric with matte black metal",
]


class PartLook(BaseModel):
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    material: Material


class FurnitureSpec(BaseModel):
    kind: Kind
    title: str = Field(max_length=120, description="Arabic catalogue name, 2-6 words")
    description: str = Field(max_length=600, description="Two or three Arabic sentences: style, materials, colours")
    style: str = Field(max_length=200)
    width_m: float = Field(gt=0)
    depth_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    leg_shape: Literal["square", "round", "tapered", "splayed"] = "tapered"
    leg_height_m: float = 0.15
    leg_thickness_m: float = 0.045
    # Bed: headboard above the mattress; chair and sofa: back above the seat.
    back_shape: Literal["straight", "rounded", "curved"] = "straight"
    back_height_m: float = 0.5
    seat_height_m: float = 0.45
    top_shape: Literal["rectangle", "rounded", "oval"] = "rectangle"
    top_thickness_m: float = 0.04
    arms: Literal["square", "rounded", "none"] = "square"
    cushions: int = 3
    tufted: bool = False
    channels: bool = False  # vertical channel tufting on upholstered backs and headboards
    bed_base: Literal["frame", "box"] = "box"  # beds: upholstered box base, or a raised frame on legs
    looks: dict[str, PartLook]
    # ponytail: shape values are not range-checked here; the builder clamps each to what it can build.


PROMPT = """You are a furniture designer for an Egyptian online furniture store. Design ONE new, attractive,
sellable, MODERN {name}, like a premium contemporary furniture brand would sell today. Style direction for this
one: {style}. Clean lines, generous upholstery, a calm, sophisticated, mostly neutral palette (greige, taupe,
oatmeal, sand, mushroom, ivory, warm or charcoal grey) with at most one tasteful accent; no ornate, rustic,
antique or kitsch details, no loud colours.
{kind_hint}
- Sizes in metres: width {w[0]}–{w[1]}, depth {d[0]}–{d[1]}, height {h[0]}–{h[1]}.
- looks: one entry for EACH of these parts, keyed exactly: {parts}. Each has a hex colour and a material.
  Upholstered parts use fabric, velvet, leather or boucle; legs and frames wood, metal, brass or lacquer;
  a table top wood, marble or lacquer.
- Shapes: leg_shape, leg_height_m, leg_thickness_m; back_shape and back_height_m ({back}); seat_height_m
  (chairs and sofas); top_shape and top_thickness_m (tables); arms and cushions (sofas: seat cushions 2–4);
  tufted (button-tufted upholstery); channels (vertical channel tufting, the modern choice); bed_base (beds).
- title and description are the shop listing in Modern Standard Arabic (Arabic script): title 2–6 words,
  description two or three sentences about the style, materials and colours. No prices or brands.
- style: a short English summary of the design."""

KIND_HINTS = {
    "bed": "Prefer a fully upholstered bed: a tall, wide headboard (height 1.2–1.4 m) in vertical channel tufting or a "
           "plain padded panel, an upholstered box base (bed_base \"box\") in the same fabric as the headboard, legs "
           "hidden or very short, crisp white or ivory mattress. Headboard, Headboard Upholstery and Sides share one fabric.",
    "sofa": "Low, deep, modern: track or softly rounded arms, plump seat and back cushions, slim or hidden legs.",
    "chair": "A modern upholstered dining chair: padded seat and back (channels or plain), slim tapered or metal legs.",
    "dining_table": "A modern dining table: a clean rectangular, rounded or oval top in wood, marble or lacquer, slim legs.",
}

BACK = {"bed": "headboard height above the mattress, 0.5–0.9", "sofa": "back above the seat, 0.3–0.5",
        "chair": "back above the seat, 0.35–0.55", "dining_table": "not used"}


def clamp(spec: FurnitureSpec) -> FurnitureSpec:
    """Sizes back inside the type's ranges, and a look for every part (a model may skip one)."""
    (w0, w1), (d0, d1), (h0, h1) = SIZES[spec.kind]
    spec.width_m, spec.depth_m, spec.height_m = (min(max(v, lo), hi) for v, lo, hi in ((spec.width_m, w0, w1), (spec.depth_m, d0, d1), (spec.height_m, h0, h1)))
    fallback = next(iter(spec.looks.values()), PartLook(color="#8a7f72", material="fabric"))
    spec.looks = {p: spec.looks.get(p, fallback) for p in PARTS[spec.kind]}
    return spec


async def design(kind: Kind, settings: Settings, client, rng: random.Random | None = None) -> FurnitureSpec:
    style = (rng or random).choice(STYLES)
    w, d, h = SIZES[kind]
    prompt = PROMPT.format(name=NAMES[kind], style=style, kind_hint=KIND_HINTS[kind], w=w, d=d, h=h, parts=", ".join(PARTS[kind]), back=BACK[kind])
    schema = FurnitureSpec.model_json_schema()
    text = await vision.ask_json([{"type": "input_text", "text": prompt}], schema, "furniture_design", settings, client, settings.openai_sol_model)
    if not text:
        raise ValueError("GPT Sol did not answer. Try again.")
    try:
        spec = FurnitureSpec.model_validate({**json.loads(text), "kind": kind})
    except (ValueError, ValidationError) as exc:
        raise ValueError("GPT Sol returned a design that could not be read. Try again.") from exc
    if not in_language(spec.title, "ar"):
        raise ValueError("GPT Sol did not write the name in Arabic. Try again.")
    return clamp(spec)


# The product photos, in listing order (the first is the main photo and thumbnail): what each shows, and its size.
SHOTS: list[tuple[str, str]] = [
    ("Hero catalogue photo. Three-quarter view from the front-left at about seat height, the whole piece in frame "
     "with comfortable margins. Seamless warm off-white studio backdrop with a very soft gradient; a large softbox key "
     "light from the left, gentle fill, a subtle rim light separating it from the background, and a soft, natural "
     "contact shadow under it. Full-frame camera, 50 mm lens, f/8: everything sharp.", "1024x1024"),
    ("Straight-on front view, the camera centred at mid-height, on the same seamless warm off-white studio backdrop "
     "with soft even lighting and a soft floor shadow, the whole piece in frame: a clean e-commerce photo that shows "
     "its true proportions.", "1024x1024"),
    ("Interior-design magazine photo: the piece as the hero of a beautiful, believable, lived-in {room} decorated in "
     "the same style ({style}). Soft natural daylight from a large window, realistic global illumination and "
     "shadows, a few tasteful accessories (a rug, a plant, a lamp, books or a throw) that never hide the piece, "
     "35 mm lens at eye level, slight depth of field. The piece is fully visible and stays the main subject.", "1536x1024"),
    ("Close-up detail photo of its materials and craftsmanship: the texture of the upholstery, the seams, the grain "
     "or finish of the legs and frame, where the parts meet. Macro lens, shallow depth of field, soft raking light "
     "that brings out the texture.", "1024x1024"),
]
ROOMS = {"bed": "bedroom", "sofa": "living room", "chair": "dining room", "dining_table": "dining room"}
MATERIALS = {
    "fabric": "woven linen-blend fabric with a visible weave", "velvet": "plush velvet with a soft sheen", "leather": "smooth top-grain leather with subtle grain",
    "boucle": "nubby bouclé fabric", "wood": "solid wood with natural grain", "metal": "powder-coated metal", "brass": "brushed brass",
    "marble": "polished natural marble with veining", "lacquer": "smooth glossy lacquer",
}


def finishes(spec_looks: dict[str, dict]) -> str:
    """The parts' materials in words: the render only shows flat colours."""
    return "; ".join(f"{part}: {MATERIALS.get(look.get('material', ''), look.get('material', ''))} in {look.get('color', '')}" for part, look in spec_looks.items())


DRESSING = {
    "bed": "Dress the bed like a premium bedding photo: crisp white sheets and duvet, layered white, ivory and grey "
           "pillows with one textured cushion, a soft knitted throw draped across the foot of the bed.",
    "sofa": "Style it with two or three soft cushions in coordinating neutral tones and a light throw on one arm.",
    "chair": "",
    "dining_table": "",
}
ROOM_DETAILS = {
    "bed": "matching nightstands with ceramic table lamps on each side, two framed botanical prints above the headboard, "
           "a soft neutral rug, warm white walls",
    "sofa": "a low coffee table, a floor lamp, a soft rug and framed art, warm white walls",
    "chair": "a modern dining table and a pendant light",
    "dining_table": "modern dining chairs around it, a pendant light above and a simple centrepiece",
}


def photo_prompt(kind: Kind, title: str, style: str, materials: str, shot: str) -> str:
    return "\n".join([
        f"Image 1 is a plain 3D render of a {NAMES[kind]} with flat, untextured surfaces.",  # no name: it must never appear in the photo
        "Turn it into a real, high-end photograph of exactly this piece, indistinguishable from a professional "
        "furniture-brand photo shoot: the same silhouette, proportions, parts and colours, but with real materials, "
        "realistic textures (fabric weave, wood grain, stitching, soft natural creases in upholstery), physically "
        "correct light, reflections and shadows.",
        f"Its materials: {materials}." if materials else "",
        f"Style: {style}." if style else "",
        "Do not redesign it, add or remove parts, or change its colours. No people, no text, logos, labels or watermarks; "
        "a single photo, not a collage.",
        DRESSING[kind],
        shot.format(room=f"{ROOMS[kind]} with {ROOM_DETAILS[kind]}", style=style or "its own style"),
    ])


def compressed(image: bytes, limit: int) -> bytes:
    """A JPEG of at most `limit` bytes: lower quality first, then smaller, until it fits."""
    picture = Image.open(io.BytesIO(image)).convert("RGB")
    while True:
        for quality in (85, 75, 65, 55, 45):
            out = io.BytesIO()
            picture.save(out, "JPEG", quality=quality, optimize=True, progressive=True)
            if out.tell() <= limit:
                return out.getvalue()
        picture = picture.resize((max(1, int(picture.width * 0.8)), max(1, int(picture.height * 0.8))), Image.LANCZOS)


async def photos(kind: Kind, title: str, render: bytes, settings: Settings, client, style: str = "", materials: str = "", which: int | None = None) -> tuple[dict[int, bytes], list[str]]:
    """Product photos from the model's render (all of SHOTS, or only `which`), at high quality, made at once;
    {shot index: image} for the ones made and the reasons for those that failed."""
    matcher = ImageMatcher(settings, client, settings.openai_sol_model)
    shots = list(enumerate(SHOTS)) if which is None else [(which, SHOTS[which])]
    results = await asyncio.gather(
        *(matcher.edit(photo_prompt(kind, title, style, materials, shot), [render], size=size, quality=settings.openai_product_photo_quality) for _, (shot, size) in shots),
        return_exceptions=True,
    )
    images, errors = {}, []
    for (i, _), result in zip(shots, results):
        if isinstance(result, ImageMatchError):
            errors.append(str(result))
        elif isinstance(result, BaseException):
            raise result
        else:
            # Each photo gets an equal share of the limit, so the whole set fits it whichever are made again.
            images[i] = compressed(result, settings.product_photos_max_bytes // len(SHOTS))
    if not images and errors:
        raise ImageMatchError(errors[0])
    return images, errors
