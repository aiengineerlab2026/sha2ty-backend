"""What to extract from a vendor photo, by catalogue category, for Meshy and Astra.

Both prompts name the one object to model (so a sofa photo with a lamp and a rug yields only the
sofa) and insist on the photo's own colours and materials (no recolouring or stylising).
"""

from __future__ import annotations

# Category → what the object is (kept in sync with frontend lib/plan/categories.ts).
OBJECTS: dict[str, str] = {
    "furniture": "a single piece of furniture (such as a sofa, bed, armchair, chair or storage unit)",
    "table": "a single table (such as a dining, coffee or side table, or a desk)",
    "wardrobe": "a single wardrobe or closet (cabinet with doors, including its handles and legs or plinth)",
    "wall_art": "a single framed picture, painting, print or mirror for hanging on a wall, including its frame",
    "tv": "a TV or audio-visual unit (screen, TV stand or sound system), including its stand or base if shown",
    "lighting": "a floor or table lamp, including its shade, stem and base",
    "ceiling_light": "a ceiling light (pendant, chandelier or flush mount), including its canopy and cord or rod",
    "air_conditioner": "an air conditioner unit (wall-mounted split or standing unit)",
}

MESHY_TEXTURE_PROMPT_MAX = 600  # Meshy rejects longer texture prompts


def object_description(category: str | None, title: str | None = None) -> str:
    what = OBJECTS.get((category or "").strip(), OBJECTS["furniture"])
    name = (title or "").strip()
    return f"{what}{f' — the product “{name[:80]}”' if name else ''}"


def colour_rule(what: str) -> str:
    return (
        f"Model only {what}. Ignore the background, floor, walls, people and any other objects in the photo. "
        "Reproduce the exact colours, patterns and materials visible in the photo: do not recolour, "
        "stylise, brighten or add colours that are not in the photo."
    )


def meshy_texture_prompt(category: str | None, title: str | None = None) -> str:
    """Meshy image-to-3D `texture_prompt`: guides texturing; the geometry comes from the photo."""
    return colour_rule(object_description(category, title))[:MESHY_TEXTURE_PROMPT_MAX]


def astra_prompt(category: str | None, title: str | None = None) -> str:
    what = object_description(category, title)
    return (
        f"Create a 3D model of {what} shown in the photo. {colour_rule(what)} "
        "Keep its real proportions, standing upright as it does in a room. "
        "In the code interpreter, write a valid binary glTF file to /mnt/data/model.glb "
        "(magic bytes glTF), with the colours as materials or vertex colours. Use pygltflib, trimesh, or write "
        "the GLB bytes yourself. The file must be a real .glb, not a description. "
        "After writing it, print the absolute path."
    )
