"""Meshy Image to 3D `texture_prompt`, written for each furniture photo by the configured vision model (the admin's
choice, else PLAN_VISION_MODEL): the colours, woods, grain, fabrics, glass, metal and finish of each visible component,
as photographed. Meshy textures from this text (texture_source "prompt"); the geometry still comes from the cleaned
Image to Image task (input_task_id) or the photo (image_url).

One vision call per Meshy generation. It reads the ORIGINAL photo: the cleaned image can have lost colour (background
removal deleting light doors), the photo has not.
"""

from __future__ import annotations

import base64
import logging
import re

import httpx
from pydantic import BaseModel

from app.core.config import Settings
from app.services import vision
from app.services.generation_prompt import MESHY_TEXTURE_PROMPT_MAX, furniture_type

logger = logging.getLogger(__name__)

# Room for Meshy's 800-character limit after trimming to a sentence.
TARGET_CHARS = 700

INSTRUCTIONS = """You write the texture description for a 3D model of the {what} in this photo. A 3D generator paints
the model's surfaces from your text alone, so describe exactly what the photo shows, component by component.

For each visible component (doors, drawers, frame, side panels, top, shelves, handles, legs, cushions, glass, mirrors,
...), give:
- its exact colour (a plain name and a hex code read from the photo, e.g. "dusty pink #D9A5A8");
- its material: the wood species when it is recognisable (oak, walnut, ash, pine, birch plywood, MDF with a painted
  or laminate finish), fabric or upholstery type, glass (clear, frosted, tinted), mirror, plastic, metal (brushed
  steel, brass, chrome, black powder-coated);
- wood grain: its pattern and direction (vertical on the doors, horizontal on the top), or "no visible grain";
- the finish: matte, satin, gloss, lacquered, textured; its roughness and how reflective it is;
- any pattern, stitching, piping, fluting, carving or print.

Rules:
- Only what is visible in the photo. Never invent a material, colour or pattern; if you cannot tell, say what is
  visible ("light painted surface, matte") and nothing more.
- The furniture only: not the walls, floor, rug, toys, bedding or decorations around or on it.
- Keep components distinct: two doors of different colours are named separately.
- Plain declarative sentences, no headings or lists, at most {limit} characters in total.

Answer as JSON: {{"texture_prompt": "..."}}"""


class TexturePrompt(BaseModel):
    texture_prompt: str


class TexturePromptError(Exception):
    """The photo could not be described (vision model not configured, unreachable, or an empty answer)."""


def fit(text: str, limit: int = MESHY_TEXTURE_PROMPT_MAX) -> str:
    """One line, at most `limit` characters, cut at the last full sentence (else word) that fits."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: end + 1] if end > limit // 2 else cut.rsplit(" ", 1)[0]).strip()


async def read_texture_prompt(photo: bytes, mime: str, category: str | None, kind: str | None, settings: Settings, state, details=None) -> str:
    """The furniture-specific texture prompt for this photo (≤ 800 characters). `details`: the optional photo reading,
    whose object type names the furniture when the vendor typed none. Raises TexturePromptError."""
    if not photo:
        raise TexturePromptError("no photo to read")
    model = await vision.current_model(settings, state.auth_http)
    if not vision.configured(settings, model):
        raise TexturePromptError(f"the vision model ({model or 'none'}) is not configured")
    what = furniture_type(category, kind, details)
    content = [
        {"type": "input_text", "text": INSTRUCTIONS.format(what=what, limit=TARGET_CHARS)},
        {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(photo).decode('ascii')}"},
    ]
    try:
        answer = await vision.ask_json(content, TexturePrompt.model_json_schema(), "meshy_texture_prompt", settings, vision.client_for(state, model), model)
        text = fit(TexturePrompt.model_validate_json(answer or "").texture_prompt)
    except (httpx.HTTPError, ValueError) as exc:
        raise TexturePromptError(f"the vision model could not describe the photo ({type(exc).__name__})") from exc
    except Exception as exc:  # noqa: BLE001 - vision errors are reported, never a silent generic prompt
        raise TexturePromptError(f"the vision model could not describe the photo ({str(exc)[:120]})") from exc
    if len(text) < 40:
        raise TexturePromptError("the vision model's texture description was empty")
    logger.info("Texture prompt by %s for a %s: %d characters", model, what, len(text))
    return text
