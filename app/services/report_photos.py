"""Real photos for the request report: the model's thumbnail and each part's picture go through the image model (image_match.ImageMatcher.edit, gpt-image-2) and comes back as a photograph of
the same piece. Results are cached on disk by the render's hash, so a report downloaded again costs nothing.
A render the model cannot turn into a photo stays in the report as it is (logged), never failing the download.
The dimension views are never sent: the image model would redraw their lines and numbers, and they must stay exact."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
from typing import Any

from PIL import Image

from app.core.config import Settings
from app.services.image_match import ImageMatcher, ImageMatchError, reference_png

logger = logging.getLogger(__name__)
PARALLEL = 6  # image calls at once (a model with many parts has many pictures)

PIECE = (
    "Image 1 is a plain 3D render of a piece of furniture. Turn it into a real, high-end photograph of exactly this piece: "
    "the same silhouette, proportions, parts, colours and camera angle, with real materials, wood grain, fabric texture, "
    "physically correct light, reflections and soft shadows, on a clean light studio background. Do not redesign it, add or "
    "remove parts. No people, logos or watermarks; a single photo, not a collage."
)
PART = (
    "Image 1 is a plain 3D render of one furniture part (a panel, door, drawer, shelf or similar). Turn it into a real "
    "product photograph of exactly this part: same shape, proportions, colour and angle, real wood grain or material "
    "texture and natural light, on a plain white background. Nothing else in the picture, no text."
)


def _size(width: int, height: int) -> str:
    ratio = width / max(height, 1)
    return "1536x1024" if ratio > 1.2 else "1024x1536" if ratio < 0.83 else "1024x1024"


async def _photo(render: bytes, prompt: str, settings: Settings, client: Any, gate: asyncio.Semaphore) -> bytes:
    """`render` as a photo of the same pixel size (JPEG); the render itself when the model fails."""
    source = Image.open(io.BytesIO(render))
    key = hashlib.sha256(prompt.encode() + render).hexdigest()
    # ponytail: local disk cache, move it to R2 if the API runs on several hosts.
    path = settings.storage_dir / "report-photos" / f"{key}.jpg"
    if path.exists():
        return path.read_bytes()
    try:
        async with gate:
            image = await ImageMatcher(settings, client).edit(prompt, [reference_png(render)], size=_size(*source.size), quality=settings.openai_image_quality)
    except ImageMatchError as exc:
        logger.warning("Report photo not made, the render is used: %s", exc)
        return render
    photo = Image.open(io.BytesIO(image)).convert("RGB").resize(source.size, Image.LANCZOS)
    out = io.BytesIO()
    photo.save(out, "JPEG", quality=88, optimize=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out.getvalue())
    return out.getvalue()


async def _sheet(sheet: bytes, cells: set[int], size: int, cols: int, settings: Settings, client: Any, gate: asyncio.Semaphore) -> bytes:
    """The parts sheet with each used cell replaced by its photo."""
    image = Image.open(io.BytesIO(sheet)).convert("RGB")
    boxes = [(c % cols * size, c // cols * size) for c in sorted(cells)]
    boxes = [(x, y) for x, y in boxes if x + size <= image.width and y + size <= image.height]

    def crop(x: int, y: int) -> bytes:
        out = io.BytesIO()
        image.crop((x, y, x + size, y + size)).save(out, "PNG")
        return out.getvalue()

    photos = await asyncio.gather(*(_photo(crop(x, y), PART, settings, client, gate) for x, y in boxes))
    for (x, y), photo in zip(boxes, photos):
        image.paste(Image.open(io.BytesIO(photo)).convert("RGB"), (x, y))
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


async def realistic(picture: bytes | None, sheet: bytes | None, meta: dict[str, Any], settings: Settings, client: Any) -> tuple[bytes | None, bytes | None]:
    """(thumbnail, parts sheet) as photos; all made at once."""
    gate = asyncio.Semaphore(PARALLEL)
    cells = {int(r["pic"]) for r in meta.get("parts") or [] if r.get("pic") is not None}
    return await asyncio.gather(
        _photo(picture, PIECE, settings, client, gate) if picture else asyncio.sleep(0, None),
        _sheet(sheet, cells, int(meta.get("pic_size") or 512), int(meta.get("pic_cols") or 8), settings, client, gate) if sheet and cells else asyncio.sleep(0, sheet),
    )
