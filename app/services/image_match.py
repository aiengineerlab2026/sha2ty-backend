"""Match through OpenAI image editing: the catalogue model's photo and the shopper's photo go to
POST /v1/images/edits as two reference images, with a prompt carrying both items' real sizes, a
room size that fits them with walking clearance, the placement rule and a per-variation camera.

The model is asked to keep both products' designs; it is a generative model, so fidelity and
proportions are close, not exact (no pixel- or CAD-level guarantees).
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

import httpx
from PIL import Image

from app.core.config import Settings

MatchType = Literal["carpet", "wall_paint", "furniture"]
Position = Literal["front", "left", "right", "behind"]
EDITS_URL = "https://api.openai.com/v1/images/edits"
REF_SIDE = 1536  # reference images are downscaled to this long side (uploads stay small)
CLEARANCE = 90  # cm of walking space kept around the furniture
CEILING = 270  # cm


class ImageMatchError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Item:
    """Real size in cm; depth may be unknown (wall paint, a flat carpet)."""

    name: str
    width: float
    height: float
    depth: float | None = None


def to_cm(value: float, unit: str) -> float:
    if unit not in ("cm", "m"):
        raise ImageMatchError("Unit must be cm or m.", 422)
    if not value or value <= 0 or not math.isfinite(value):
        raise ImageMatchError("Dimensions must be positive numbers.", 422)
    return value * 100 if unit == "m" else value


def room_size(kind: MatchType, position: Position | None, model: Item, item: Item) -> tuple[int, int]:
    """Floor width × depth (cm) that holds both pieces plus walking clearance, rounded up to 10 cm.
    Carpets lie under the model, wall finishes take no floor."""
    md = model.depth or model.width
    if kind == "furniture":
        idp = item.depth or 0.5 * item.width
        if position in ("left", "right"):
            width = model.width + item.width + 60  # a small gap between them
            depth = max(md, idp)
        else:
            width = max(model.width, item.width)
            depth = md + idp + 60
    elif kind == "carpet":
        width, depth = max(model.width, item.width), max(md, item.height)
    else:
        width, depth = max(model.width, item.width), md
    width, depth = width + 2 * CLEARANCE, depth + 2 * CLEARANCE
    return int(math.ceil(width / 10) * 10), int(math.ceil(depth / 10) * 10)


# One distinct, valid camera / arrangement per variation (cycled when fewer are asked for).
VARIATIONS = [
    "Camera at standing eye level (about 160 cm) from the room's entrance, looking straight at the arrangement.",
    "Three-quarter view from a front corner of the room, so the arrangement is seen at about 35 degrees.",
    "Three-quarter view from the opposite front corner, mirrored from a corner view.",
    "Wider view from further back that shows more floor and walls around both pieces.",
    "Slightly elevated view (about 190 cm high) looking down a little, showing the floor layout clearly.",
]

PLACEMENT = {
    "left": "to the LEFT of the {model} as seen from the camera, side by side, about 30 to 60 cm apart, both with their backs against the same wall",
    "right": "to the RIGHT of the {model} as seen from the camera, side by side, about 30 to 60 cm apart, both with their backs against the same wall",
    "front": "in FRONT of the {model} (between it and the camera), about 50 to 80 cm away from it, with a clear walking gap between them",
    "behind": "BEHIND the {model} (further from the camera, against the back wall), with the {model} standing about 50 cm in front of it",
}

WALLS = {
    "behind": "the wall BEHIND the {model}",
    "left": "the side wall to the LEFT of the {model}",
    "right": "the side wall to the RIGHT of the {model}",
    "front": "the wall FACING the {model}; choose a camera angle that shows that wall and the {model} together",
}


def fmt(item: Item) -> str:
    dims = f"{item.width:.0f} cm wide, {item.height:.0f} cm tall"
    return dims + (f", {item.depth:.0f} cm deep" if item.depth else "")


def build_prompt(kind: MatchType, position: Position | None, model: Item, item: Item, variation: int, color: str | None = None, tile_cm: float | None = None) -> str:
    room_w, room_d = room_size(kind, position, model, item)
    name = model.name or "product"
    lines = [
        "Create one photorealistic photograph of a furnished, lived-in interior room.",
        f"Image 1 is a product photo of the existing item: “{name}”. Its real size is {fmt(model)}.",
    ]
    if kind == "furniture":
        lines += [
            f"Image 2 is a product photo of a second piece of furniture. Its real size is {fmt(item)}.",
            f"Place the piece from Image 2 {PLACEMENT[position or 'left'].format(model=name)}.",
        ]
    elif kind == "carpet":
        lines += [
            f"Image 2 is a carpet. It is {item.width:.0f} cm wide and {item.height:.0f} cm long, lying flat.",
            f"Lay the carpet flat on the floor, centred underneath the {name}, so the {name} stands on it; the carpet's visible border must match its real size relative to the {name}.",
        ]
    else:
        what = f"the solid paint colour {color} shown in Image 2" if color else "the wall finish (texture) shown in Image 2"
        lines += [
            f"Image 2 is {'a paint colour swatch' if color else 'a wall texture'}.",
            f"Paint {WALLS[position or 'behind'].format(model=name)} with {what}, covering that whole wall evenly behind any furniture, keeping the room's light and shadows on it.",
        ]
        if tile_cm and not color:
            lines.append(f"The texture repeats every {tile_cm:.0f} cm on the wall, so its pattern is drawn at that real scale.")
    lines += [
        f"The room's floor is about {room_w} cm wide and {room_d} cm deep, with a {CEILING} cm ceiling: enough for both items plus about {CLEARANCE} cm of walking space.",
        "All measurements are in centimetres. Keep the two items at their correct real-world sizes relative to each other and to the room (door height 210 cm, ceiling 270 cm as scale references).",
        "Reproduce both products exactly as they appear in their reference images: same design, shape, proportions, colours, materials, textures and details. Do not redesign, recolour or simplify them.",
        "Both items must be completely visible in the frame, not cropped, not hidden, standing on the same floor with natural contact shadows.",
        "Realistic walls, flooring, daylight and lighting, consistent perspective and shadows. Light, neutral decor is fine, but no other large furniture that could be mistaken for the products.",
        "Output a single photograph: no collage, no split screen, no inset images, no white or studio background, no text, labels or watermarks.",
        f"Camera: {VARIATIONS[variation % len(VARIATIONS)]}",
    ]
    return "\n".join(lines)


def reference_png(data: bytes) -> bytes:
    """Any uploaded format → PNG, long side ≤ REF_SIDE, original colours untouched."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types for bad files
        raise ImageMatchError("The file is not a readable image.", 422) from exc
    image = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
    image.thumbnail((REF_SIDE, REF_SIDE), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, "PNG", optimize=True)
    return out.getvalue()


def swatch_png(color: str) -> bytes:
    """A solid colour reference for "paint this colour" (both requests always carry two images)."""
    value = color.strip().lstrip("#")
    if len(value) != 6 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ImageMatchError("Choose a paint colour like #D8C9B0.", 422)
    out = io.BytesIO()
    Image.new("RGB", (512, 512), tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))).save(out, "PNG")
    return out.getvalue()


def _openai_message(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])
    except Exception:  # noqa: BLE001
        return response.text[:300] or f"HTTP {response.status_code}"


class ImageMatcher:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    async def edit(self, prompt: str, references: list[bytes]) -> bytes:
        """One generated image from the prompt and the reference images (PNG bytes)."""
        s = self.settings
        data = {"model": s.openai_image_model, "prompt": prompt, "size": s.openai_image_size, "quality": s.openai_image_quality, "n": "1"}
        if s.openai_image_input_fidelity:
            data["input_fidelity"] = s.openai_image_input_fidelity
        files = [("image[]", (f"reference-{i + 1}.png", ref, "image/png")) for i, ref in enumerate(references)]
        try:
            response = await self.client.post(EDITS_URL, data=data, files=files, timeout=s.openai_image_timeout_seconds)
        except httpx.TimeoutException as exc:
            raise ImageMatchError("Image generation timed out. Try again, or ask for fewer variations.", 504) from exc
        except httpx.HTTPError as exc:
            raise ImageMatchError("Could not reach the image generation service.", 502) from exc
        if response.status_code >= 400:
            message = _openai_message(response)
            status = 422 if response.status_code == 400 else 429 if response.status_code == 429 else 502
            raise ImageMatchError(f"Image generation failed: {message}", status)
        try:
            b64 = response.json()["data"][0]["b64_json"]
            return base64.b64decode(b64)
        except Exception as exc:  # noqa: BLE001
            raise ImageMatchError("Image generation returned no image.", 502) from exc

    async def variations(self, prompts: list[str], references: list[bytes]) -> tuple[list[tuple[int, bytes]], list[str]]:
        """Run every variation at once; keep the successes (index, image) and report the failures."""
        result: tuple[list[tuple[int, bytes]], list[str]] = ([], [])
        async for kind, value in self.variations_with_progress(prompts, references):
            if kind == "result":
                result = value  # type: ignore[assignment]
        return result

    async def variations_with_progress(self, prompts: list[str], references: list[bytes], tick: float = 1.0) -> AsyncIterator[tuple[str, object]]:
        """Like `variations`, yielding ("progress", percent) about every `tick` s, then ("result",
        (images, errors)). Cancels the requests still running if the caller goes away."""
        tasks = [asyncio.create_task(self.edit(p, references)) for p in prompts]
        start = time.monotonic()
        try:
            while True:
                done = [t.done() for t in tasks]
                yield "progress", progress(done, time.monotonic() - start, self.settings.openai_image_expected_seconds)
                if all(done):
                    break
                await asyncio.wait(tasks, timeout=tick, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
        images: list[tuple[int, bytes]] = []
        failures: list[ImageMatchError] = []
        for i, t in enumerate(tasks):
            exc = t.exception()
            if isinstance(exc, ImageMatchError):
                failures.append(exc)
            elif exc is not None:
                raise exc
            else:
                images.append((i, t.result()))
        if not images:
            raise failures[0] if failures else ImageMatchError("Image generation failed.", 502)
        yield "result", (images, [str(f) for f in failures])


def progress(done: list[bool], elapsed: float, expected: float) -> int:
    """Percent for the shopper: 5 once the references are ready, then each image's share of 90
    fills with time against the usual generation time (held at 95 % of its share until it really
    finishes, so it never claims to be done early); 100 only comes with the saved result."""
    shares = [1.0 if d else min(0.95, elapsed / max(expected, 1.0)) for d in done]
    return min(95, round(5 + 90 * sum(shares) / max(len(shares), 1)))
