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
# The parts each furniture type is cut into (Meshy Auto Split by_parts names 1–10 parts in at most 600 characters):
# each one a separate, independently editable mesh.
_PARTS = {
    "sofa": "wooden base frame, left armrest frame, right armrest frame, backrest frame, legs",
    "bed": "headboard, left side panel, right side panel, footboard, base supports, legs",
    "wardrobe": "left side panel, right side panel, top panel, bottom panel, back panel, doors, shelves, drawers",
    "table": "tabletop, apron rails, legs",
    "chair": "backrest frame, seat frame, front legs, rear legs, armrest frames",
    "door": "door leaf, door frame, handle",
}


def object_description(category: str | None, title: str | None = None, kind: str | None = None) -> str:
    """`kind`: the furniture type the vendor typed (e.g. "corner sofa", "bunk bed"); it replaces the generic
    furniture description, so the vision model, Meshy and Astra know exactly what to measure and model."""
    custom = " ".join((kind or "").split())[:60]
    category = (category or "").strip()
    if category == "auto":  # only a photo (the app's camera): the reading decides its type; a typed type says what it is
        return f"a single {custom}" if custom else "the single main product in the photo (furniture, a table, wardrobe, door, TV unit, lamp, ceiling light, air conditioner or wall art)"
    if custom and category not in ("", "furniture"):  # typed in the app for a type read from the photo
        what = f"{OBJECTS.get(category, OBJECTS['furniture'])}, specifically a {custom}"
    else:
        what = f"a single {custom} (a piece of furniture)" if custom else OBJECTS.get(category, OBJECTS["furniture"])
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


# Kept for callers that still pass a budget. Image to 3D does not remesh to it: a remesh replaces the generated
# surface. The file is shrunk afterwards by re-encoding textures (glb_size).
POLYCOUNT = {"sofa": 60000, "bed": 60000, "chair": 30000, "table": 30000, "wardrobe": 40000, "door": 20000}
DEFAULT_POLYCOUNT = 50000


def target_polycount(category: str | None, kind: str | None = None) -> int:
    return POLYCOUNT.get(furniture_family(category, kind), DEFAULT_POLYCOUNT)


_EXAMPLES = {"chair": "legs, seat, backrest", "sofa": "frame, seat cushions, back cushions, armrests, legs", "bed": "headboard, frame, mattress, pillows, legs", "table": "tabletop, legs", "wardrobe": "body, doors, drawers, handles", "door": "leaf, frame, handle"}


def _read_type(details) -> str | None:
    """What the photo reading says it is ("three-seat sofa"), for the size rule's examples when no type was typed."""
    return details.object_type if details and details.matches_category and details.object_type else None


def size_rule(category: str | None, kind: str | None = None) -> str:
    """The file-size instruction for both providers, with this furniture type's parts. A provider cannot promise a
    size: the backend checks and optimises the GLB it returns (services/glb_size.py)."""
    parts = _EXAMPLES.get(furniture_family(category, kind), "chair legs, seat, backrest")
    return (
        "Generate an optimized, lightweight 3D model with a final GLB file size under 8 MB. Minimize polygons and "
        "texture sizes while preserving visual quality, accurate dimensions, colors, and modular components "
        f"(e.g., {parts})."
    )


def split_prompt(category: str | None, kind: str | None = None) -> str | None:
    """Meshy Auto Split `prompt` (mode by_parts): the parts to cut the model into, at most 10, 600 characters."""
    return _PARTS.get(furniture_family(category, kind))


# What the product is called in Meshy's prompts when no furniture type was typed (by catalogue category).
TYPE_NAMES = {
    "furniture": "piece of furniture", "table": "table", "wardrobe": "wardrobe", "door": "door", "wall_art": "framed wall art",
    "tv": "TV unit", "lighting": "lamp", "ceiling_light": "ceiling light", "air_conditioner": "air conditioner",
}


def furniture_type(category: str | None, kind: str | None = None, details=None) -> str:
    """The furniture type for Meshy's prompts: what the vendor typed ("child wardrobe"), else what the photo reading
    says it is, else the category's name."""
    typed = " ".join((kind or "").split())[:60]
    noun = TYPE_NAMES.get((category or "").strip())
    if typed:
        # A specific category stays named: "wardrobe" with "kids" typed is a "wardrobe (kids)", never just "kids".
        specific = noun and category not in ("furniture", "auto") and noun.lower() not in typed.lower()
        return f"{noun} ({typed})" if specific else typed
    read = _read_type(details)
    return read or TYPE_NAMES.get((category or "").strip(), "piece of furniture")


def meshy_prompt(category: str | None, kind: str | None = None, details=None) -> str:
    """Prompt for the configured vision model that cleans the photo. Image to 3D then builds from that image, so this is
    an EDIT of the photo, never a redraw: the same camera, framing and furniture, only the loose objects gone. (It
    used to ask for "a new product" in "a three-quarter product view", which redrew the furniture from another angle
    and changed its proportions.) Whatever it returns is still compared with the original before Meshy sees it."""
    what = furniture_type(category, kind, details)
    return (
        f"Edit this photo into a clean product reference image of the {what} shown. This is an edit of the photo, not a new picture.\n\n"
        f"Remove only the loose objects placed on, inside, or in front of the {what}: decorations, plants, devices, books, bags, "
        "clothes, boxes, stored contents and loose accessories. Replace the room around it with a plain white background.\n\n"
        "Where a removed object covered the furniture, continue the furniture's own surface behind it: the same material, "
        "colour, grain and edges.\n\n"
        "Keep everything else exactly as photographed:\n"
        "- The same camera viewpoint, perspective, framing and image proportions. Do not rotate, re-frame, crop, stretch or re-pose it.\n"
        "- Every permanent component in place: panels, doors, drawers, shelves, frames, handles, legs, glass and mirrors, "
        "with the same count, size and position.\n"
        "- The same material boundaries, colours, patterns, wood grain, gloss and matte finishes.\n"
        "- Glass stays transparent and mirrors stay reflective.\n\n"
        f"Do not redesign, simplify, straighten, restyle or relight the {what}, and add nothing."
    )


# The reconstruction brief for every photo → 3D conversion ({object_type}: the furniture type). GPT Astra / Sol write
# the model themselves and get all of it. Meshy's Image to 3D takes text only as `texture_prompt` (texture guidance, at
# most 800 characters): it gets the brief's texture requirements (meshy_texture_prompt); its geometry follows the photo,
# kept as given (`image_enhancement: false`), remeshed and split by the settings that act on geometry.
RECONSTRUCTION_PROMPT = """Create a high-quality, photorealistic, structurally accurate 3D reconstruction of the {object_type} shown in the reference image.

Geometry and Structure:
- Reproduce the exact shape, proportions, dimensions, orientation, and visible components.
- Preserve all structural details, edges, curves, openings, and component relationships.
- Generate clean, solid, realistic geometry with correct topology.
- Keep flat surfaces flat, straight edges straight, and curved surfaces smooth.
- Preserve the original arrangement of components without inventing or removing parts.
- Avoid warped, melted, stretched, floating, overlapping, or disconnected geometry.

Materials and Textures:
- Match the reference image's colors, patterns, textures, and materials exactly. Do not recolor a surface or replace its texture with a flat colour.
- Preserve original material boundaries and component-specific colors.
- Generate realistic PBR materials with correct UV mapping. Patterned and photographed surfaces use a baseColorTexture taken from the reference image.
- Glass stays transparent: transmission with low roughness. Mirrors stay reflective: metallicFactor 1 and roughnessFactor near 0. Keep both effects.
- Avoid blurry textures, artificial wrinkles, discoloration, and texture stretching.

Modularity:
- Separate meaningful physical components into independently editable meshes where supported.
- Preserve each component's original geometry, materials, and position.
- Avoid arbitrary mesh fragmentation.

Accuracy Requirements:
- Reconstruct the same object, not a redesigned or stylized variation.
- Preserve visible details and mechanical relationships.
- Infer hidden surfaces conservatively using realistic construction.
- Prioritize structural accuracy, reference-image similarity, and clean geometry over unnecessary decorative details.

Strictly avoid: Geometry distortion, missing components, incorrect proportions, invented features, texture stretching, color changes, and malformed surfaces.

The final result should resemble a professionally manufactured 3D product model that faithfully matches the reference image."""


def reconstruction_prompt(category: str | None, kind: str | None = None, details=None) -> str:
    return RECONSTRUCTION_PROMPT.format(object_type=furniture_type(category, kind, details))


# Always the start of Meshy's texture_prompt, for every furniture type. Kept in front of the 800-character cut so a
# long colour list cannot push it off the end.
# Texture guidance only: removing objects is the Image to Image step's job (meshy_prompt), never texture_prompt's.
CARPENTER_LEAD = (
    "Carpenter-ready modular {object_type} only. "
    "Keep its colors, textures, clear glass, and reflective mirrors exactly as in the reference image. "
)
CARPENTER_LEAD_END = "in the reference image. "

# The brief's texture requirements, in its own words: what Meshy's texture_prompt can act on.
MESHY_TEXTURE_BRIEF = (
    "Photorealistic {object_type}: the same object as the reference image, not a redesigned or stylized variation. "
    "Match the reference image's colors, patterns, textures, and materials as closely as possible. "
    "Preserve original material boundaries and component-specific colors. "
    "Realistic PBR materials with correct UV mapping. "
    "Avoid blurry textures, artificial wrinkles, discoloration, texture stretching, and color changes."
)


def carpenter_lead(object_type: str) -> str:
    return CARPENTER_LEAD.format(object_type=object_type)


def split_carpenter_lead(text: str) -> tuple[str, str]:
    """The carpenter request, and the rest. Empty lead when this text is not a Meshy texture prompt."""
    end = text.find(CARPENTER_LEAD_END)
    if text.startswith("Carpenter-ready modular ") and end != -1:
        cut = end + len(CARPENTER_LEAD_END)
        return text[:cut], text[cut:]
    return "", text


def _fit_keeping_lead(lead: str, rest: str, limit: int) -> str:
    """The carpenter request stays. Only the text after it is shortened to Meshy's limit."""
    rest = rest.strip()
    if not lead:
        if len(rest) <= limit:
            return rest
        cut = rest[:limit]
        return cut[: cut.rfind(".") + 1] or cut
    if len(lead) + len(rest) <= limit:
        return f"{lead}{rest}".strip()
    room = limit - len(lead)
    if room <= 1:
        return lead[:limit].rstrip()
    tail = rest[:room].rstrip()
    dot = tail.rfind(".")
    if dot > 40:
        tail = tail[: dot + 1]
    return f"{lead}{tail}".strip()


def meshy_texture_prompt(category: str | None, title: str | None = None, details=None, kind: str | None = None) -> str:
    """Meshy Image to 3D `texture_prompt`. The carpenter request is always first. Texture guidance and the photo's
    colours follow it, and are what gets shortened when the whole prompt passes 800 characters."""
    what = furniture_type(category, kind, details)
    name = (title or "").strip()
    rest = f"{MESHY_TEXTURE_BRIEF.format(object_type=what)} {f'The product: “{name[:80]}”. ' if name else ''}{details_text(details)}"
    return _fit_keeping_lead(carpenter_lead(what), rest, MESHY_TEXTURE_PROMPT_MAX)


def astra_prompt(category: str | None, title: str | None = None, details=None, kind: str | None = None) -> str:
    """GPT Astra / Sol write the GLB themselves: the whole reconstruction brief, then what the photo reading measured
    and what the file must be."""
    what = _what(category, title, details, kind)
    hidden = f" The photo does not show {details.hidden}: keep those parts simple and plausible." if details and details.hidden else ""
    return (
        f"{reconstruction_prompt(category, kind, details)}\n\n"
        f"The object: {what}. {focus_rule(what, category)} {details_text(details)} {size_text(details)}{hidden} "
        f"{modular_hint(category, kind, details)} {size_rule(category, kind or _read_type(details))} "
        "Keep its real proportions, standing upright as it does in a room. "
        "Bake each photographed color and texture into the material. "
        "Glass uses transmission and low roughness. Mirrors use metallicFactor 1 and roughnessFactor near 0. "
        "Use few vertices per part and at most 1024 px textures. "
        "In the code interpreter, write a valid binary glTF file to /mnt/data/model.glb "
        "(magic bytes glTF). Use pygltflib, trimesh, or write "
        "the GLB bytes yourself. The file must be a real .glb, not a description. "
        "After writing it, print the absolute path."
    )
