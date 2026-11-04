"""What to extract from a vendor photo, by catalogue category, for Meshy and Astra.

Both prompts name the one object to model and tell the generator to ignore everything around it
(a sofa photo with a lamp, a rug and a plant yields only the sofa). A table keeps the chairs set at it
and nothing else. Colours and materials are the photo's own (no recolouring or stylising).
"""

from __future__ import annotations

import re

# Category → what the object is (kept in sync with frontend lib/plan/categories.ts).
OBJECTS: dict[str, str] = {
    "furniture": "a single piece of furniture (such as a sofa, bed, armchair, chair or storage unit)",
    "table": "a single table (such as a dining, coffee or side table, or a desk) with the chairs set at it, if the photo shows them",
    "wardrobe": "a single wardrobe or closet (cabinet with doors, including its handles and legs or plinth)",
    "door": "a single door: the door leaf with its frame, handle and hinges, standing upright as fitted in a doorway",
    "wall_art": "a single framed picture, painting, print or mirror for hanging on a wall, including its frame",
    "tv": "a TV or audio-visual unit (screen, TV stand or sound system), including its stand or base if shown",
    "lighting": "a floor or table lamp, including its shade, stem and base",
    "ceiling_light": "a ceiling light (pendant, chandelier or flush mount), including its canopy and cord or rod",
    "air_conditioner": "an air conditioner unit (wall-mounted split or standing unit)",
}

MESHY_TEXTURE_PROMPT_MAX = 800  # Meshy's documented texture_prompt limit (docs.meshy.ai: image-to-3d)

# The parts each furniture type is asked to be built from. Astra writes the GLB itself, so it can honour this;
# Meshy's image-to-3D API has no part option (texture_prompt only guides texturing), so for Meshy the GLB is
# inspected afterwards and, when it came back as one mesh, split from its geometry (glb_parts.py).
_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sofa", ("sofa", "couch", "settee")),
    ("bed", ("bed", "bunk")),
    ("wardrobe", ("wardrobe", "closet", "armoire")),
    ("table", ("table", "desk")),
    ("chair", ("chair", "armchair", "stool")),
    ("door", ("door",)),
)
_PARTS = {
    "sofa": "frame, seat cushions, back pillows, armrests, legs",
    "bed": "frame, mattress, headboard, pillows",
    "wardrobe": "body, doors, drawers, handles",
    "table": "tabletop, legs, frame",
    "chair": "seat, back, legs, arms",
    "door": "leaf, frame, handle",
}


def object_description(category: str | None, title: str | None = None, kind: str | None = None) -> str:
    """`kind`: the furniture type the vendor typed (e.g. "corner sofa", "bunk bed"); it replaces the generic
    furniture description, so the vision model, Meshy and Astra know exactly what to measure and model."""
    custom = " ".join((kind or "").split())[:60]
    category = (category or "").strip()
    what = f"a single {custom} (a piece of furniture)" if category in ("", "furniture") and custom else OBJECTS.get(category, OBJECTS["furniture"])
    name = (title or "").strip()
    return f"{what}{f' — the product “{name[:80]}”' if name else ''}"


def focus_rule(what: str, category: str | None = None) -> str:
    """Only the product: everything around it in the photo is scenery, not part of the model."""
    around = "the background, floor, walls, rugs, people, plants, decoration and any other furniture or objects"
    if (category or "").strip() == "table":
        return (
            f"Model only {what}: the table and the chairs at it, nothing else. "
            f"Ignore {around}, and anything standing on the table (tableware, vases, books)."
        )
    return f"Model only {what}: one object of this type. Ignore {around} in the photo, even where they touch it."


def colour_rule(what: str, category: str | None = None) -> str:
    return (
        f"{focus_rule(what, category)} "
        "Reproduce the exact colours, patterns and materials visible in the photo: do not recolour, "
        "stylise, brighten or add colours that are not in the photo."
    )


def _what(category: str | None, title: str | None, details, kind: str | None = None) -> str:
    """The vision model's reading of the photo names the object more precisely than the category."""
    what = object_description(category, title, kind)
    return f"{what}: a {details.object_type}" if details and details.matches_category and details.object_type else what


def details_text(details) -> str:
    """Visible parts, materials and colours from the vision model (services/product_details.py)."""
    if not details:
        return ""
    bits = []
    if details.components:
        parts = dict.fromkeys(" ".join(x for x in (c.color, c.material, c.name) if x) for c in details.components)  # each part once
        bits.append("Visible parts: " + ", ".join(list(parts)[:8]) + ".")
    if details.materials:
        bits.append("Materials: " + ", ".join(details.materials[:5]) + ".")
    if details.colors:
        bits.append("Colours: " + ", ".join(details.colors[:5]) + ".")
    return " ".join(bits)


def size_text(details) -> str:
    """Estimated size and proportions, always worded as an estimate (a photo shows no exact size)."""
    if not details:
        return ""
    dims = [v for v in (details.width_m, details.depth_m, details.height_m)]
    if all(dims):
        return f"Its size is roughly {dims[0]:.2f} m wide, {dims[1]:.2f} m deep and {dims[2]:.2f} m tall (estimated from the photo)."
    if details.proportions:
        w, d, h = details.proportions
        return f"Its proportions are about width {w} : depth {d} : height {h}."
    return ""


def furniture_family(category: str | None, kind: str | None = None) -> str:
    """sofa / bed / wardrobe / table / chair / door, or the catalogue category, or furniture. From what the
    vendor selected only (the category and the furniture type typed), never from reading the photo."""
    blob = " ".join(x for x in (kind, category) if x).lower()
    for name, words in _FAMILIES:
        if any(re.search(rf"\b{re.escape(word)}\b", blob) for word in words):
            return name
    return (category or "").strip() or "furniture"


def modular_hint(category: str | None, kind: str | None = None, details=None) -> str:
    parts = _PARTS.get(furniture_family(category, kind))
    if not parts:
        return ""
    return f"Separate named meshes: {parts}; each part with its own material, looking exactly as in the photo."


def split_prompt(category: str | None, kind: str | None = None) -> str | None:
    """Meshy Auto Split `prompt` (mode by_parts): the parts to cut the model into, at most 10, 600 characters."""
    return _PARTS.get(furniture_family(category, kind))


def meshy_texture_prompt(category: str | None, title: str | None = None, details=None, kind: str | None = None) -> str:
    """Meshy image-to-3D `texture_prompt`: guides texturing; the geometry comes from the photo."""
    what = _what(category, title, details, kind)
    hint = modular_hint(category, kind, details)
    body = f"{colour_rule(what, category)} {details_text(details)}".strip()
    # The part list stays even when the colour notes are long: Meshy caps this field at 600 characters.
    if hint:
        room = MESHY_TEXTURE_PROMPT_MAX - len(hint) - 1
        if len(body) > room:
            cut = body[:room]
            body = (cut[: cut.rfind(".") + 1] or cut).strip()
        text = f"{body} {hint}".strip()
    else:
        text = body
    if len(text) <= MESHY_TEXTURE_PROMPT_MAX:
        return text
    cut = text[:MESHY_TEXTURE_PROMPT_MAX]
    return cut[: cut.rfind(".") + 1] or cut  # whole sentences only


def astra_prompt(category: str | None, title: str | None = None, details=None, kind: str | None = None) -> str:
    what = _what(category, title, details, kind)
    hidden = f" The photo does not show {details.hidden}: keep those parts simple and plausible." if details and details.hidden else ""
    return (
        f"Create a 3D model of {what} shown in the photo. {colour_rule(what, category)} {details_text(details)} {size_text(details)}{hidden} "
        f"{modular_hint(category, kind, details)} "
        "Keep its real proportions, standing upright as it does in a room. "
        "In the code interpreter, write a valid binary glTF file to /mnt/data/model.glb "
        "(magic bytes glTF), with the colours as materials or vertex colours. Use pygltflib, trimesh, or write "
        "the GLB bytes yourself. The file must be a real .glb, not a description. "
        "After writing it, print the absolute path."
    )
