"""Client for Meshy's Image to 3D REST API (https://docs.meshy.ai/en/api/image-to-3d)."""

import base64
import logging
from pathlib import Path
from typing import Any, Literal

import anyio
import httpx

from app.core.config import Settings
from app.schemas.vendor import GenerationOptions

logger = logging.getLogger(__name__)

IMAGE_TO_3D_PATH = "/image-to-3d"
# Meshy's photo-to-3D model (docs: `latest` is meshy-7.1). Named explicitly so a photo is never sent to an image editor.
IMAGE_TO_3D_MODEL = "meshy-7.1"
# Meshy Image to Image cleans the photo before Image to 3D (vendor.generate_model, Meshy only).
IMAGE_TO_IMAGE_PATH = "/image-to-image"
IMAGE_TO_IMAGE_MODEL = "gpt-image-2-5-sunburst"
# docs.meshy.ai/en/api/image-to-image (nano-banana-pro needs a Pro plan: Meshy answers 402 otherwise).
IMAGE_TO_IMAGE_MODELS = frozenset({
    "nano-banana", "nano-banana-2", "nano-banana-2-lite", "nano-banana-2-1", "nano-banana-pro", "gpt-image-2",
    "gpt-image-2-5-flare", "gpt-image-2-5-sunburst", "gpt-image-2-6-flare", "gpt-image-2-6-sunburst",
})
# Every Image to Image model takes these; GPT Image models also 3:2 / 2:3 (nano-banana answers 400). Default 1:1
# would re-frame most photos.
ASPECT_RATIOS = {"1:1": 1.0, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4}
GPT_ASPECT_RATIOS = {**ASPECT_RATIOS, "3:2": 3 / 2, "2:3": 2 / 3}
TEXTURE_MODELS = frozenset({"meshy-7", "meshy-n1", "meshy-6-lite"})
GEOMETRY_MODELS = {"2k": {"meshy-7.1", "latest", "meshy-7"}, "4k": {"meshy-7.1", "latest"}}
TEXTURE_PROMPT_MAX = 800
MAX_REFERENCE_BYTES = 20 * 1024 * 1024  # Meshy's own image input limit
SPLIT_PATH = "/print/split"  # Auto Split (https://docs.meshy.ai/en/api/auto-split)


def _redact(image: str) -> str:
    return f"<data URI, {len(image)} chars>" if image.startswith("data:") else image.split("?", 1)[0]


def _loggable(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload as logged: photos replaced by their kind and size (never their data or a signed URL's query); the
    API key is a header, never in the payload."""
    out = dict(payload)
    for key in ("image_url", "texture_image_url"):
        if isinstance(out.get(key), str):
            out[key] = _redact(out[key])
    if isinstance(out.get("reference_image_urls"), list):
        out["reference_image_urls"] = [_redact(u) if isinstance(u, str) else u for u in out["reference_image_urls"]]
    return out


TextureSource = Literal["image", "input", "prompt"]


def quality_options(settings: Settings) -> dict[str, Any]:
    """The Image to 3D quality settings, as image_to_3d_payload's keyword arguments."""
    return {"geometry_resolution": settings.meshy_geometry_resolution, "texture_resolution": settings.meshy_texture_resolution,
            "texture_model": settings.meshy_texture_model, "image_enhancement": settings.meshy_image_enhancement,
            "should_remesh": settings.meshy_should_remesh}


def nearest_aspect_ratio(image: bytes | None, model: str = IMAGE_TO_IMAGE_MODEL) -> str | None:
    """The Image to Image aspect_ratio closest to the photo's, so the cleaned image keeps its framing (None: unknown)."""
    if not image:
        return None
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        w, h = Image.open(io.BytesIO(image)).size
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    ratios = GPT_ASPECT_RATIOS if model.startswith("gpt-image") else ASPECT_RATIOS
    return min(ratios, key=lambda k: abs(ratios[k] - w / max(h, 1)))


def check_image_to_3d(payload: dict[str, Any]) -> list[str]:
    """Combinations Meshy documents as 400s, caught before a paid request is sent."""
    errors = []
    ai, geometry, tm = payload.get("ai_model"), payload.get("geometry_resolution", "standard"), payload.get("texture_model")
    if geometry not in ("standard", "2k", "4k"):
        errors.append(f"geometry_resolution {geometry!r} is not standard, 2k or 4k")
    elif geometry in GEOMETRY_MODELS and ai not in GEOMETRY_MODELS[geometry]:
        errors.append(f"geometry_resolution {geometry} needs ai_model {' or '.join(sorted(GEOMETRY_MODELS[geometry]))}")
    if tm is not None and tm not in TEXTURE_MODELS:
        errors.append(f"texture_model {tm!r} is not one of {', '.join(sorted(TEXTURE_MODELS))}")
    if payload.get("texture_resolution", "2k") not in ("2k", "4k", "8k"):
        errors.append("texture_resolution must be 2k, 4k or 8k")
    if payload.get("texture_resolution") in ("4k", "8k") and tm in ("meshy-n1", "meshy-6-lite"):
        errors.append(f"texture_resolution {payload['texture_resolution']} needs the meshy-7 texture model, not {tm}")
    if payload.get("enable_pbr") and not payload.get("should_texture"):
        errors.append("enable_pbr needs should_texture")
    if tm == "meshy-n1" and payload.get("enable_pbr"):
        errors.append("texture_model meshy-n1 does not take enable_pbr")
    if tm == "meshy-n1" and payload.get("texture_prompt"):
        errors.append("texture_model meshy-n1 does not take texture_prompt")
    if len(payload.get("texture_prompt") or "") > TEXTURE_PROMPT_MAX:
        errors.append(f"texture_prompt is over {TEXTURE_PROMPT_MAX} characters")
    if payload.get("texture_prompt") and payload.get("texture_image_url"):
        errors.append("texture_prompt and texture_image_url together: Meshy would ignore the image")
    if not (payload.get("image_url") or payload.get("input_task_id")):
        errors.append("no image_url or input_task_id")
    return errors


def image_to_3d_payload(options: GenerationOptions, source: dict[str, Any], texture_prompt: str | None, target_polycount: int | None,
                        texture_source: TextureSource = "image", *, geometry_resolution: str = "standard", texture_resolution: str = "2k",
                        texture_model: str = "", image_enhancement: bool = False, should_remesh: bool = False) -> dict[str, Any]:
    """The Image to 3D request body. Meshy reads ONE texture guide: with both `texture_prompt` and `texture_image_url`
    in the body, texture_prompt wins (docs.meshy.ai/en/api/image-to-3d), so a generic text used to replace the photo as
    the texture reference. Exactly one is sent now:
      image  → texture_image_url = the image being meshed (the photo's own colours, grain, glass, metal); no prompt
      input  → neither: Meshy textures from the input image itself
      prompt → texture_prompt only (text guidance, the old behaviour)."""
    payload: dict[str, Any] = {
        **source,
        "ai_model": IMAGE_TO_3D_MODEL,
        # false keeps the photo as given. true restyles it before meshing, which is a different model than the one selected.
        "image_enhancement": image_enhancement,
        "target_formats": ["glb"],
        "should_texture": options.should_texture,
        # Remeshing replaces the generated surface with a decimated one and is what warped furniture. The photo
        # is meshed as Meshy generated it. File size is reduced afterwards by re-encoding textures (glb_size).
        "should_remesh": should_remesh,
    }
    if should_remesh and target_polycount:
        payload["target_polycount"] = target_polycount
    if geometry_resolution != "standard":
        payload["geometry_resolution"] = geometry_resolution
    if options.should_texture:
        payload["texture_resolution"] = texture_resolution
        if texture_model:
            payload["texture_model"] = texture_model
        # metallic, roughness and normal maps besides the colour (meshy-n1 makes base colour only)
        payload["enable_pbr"] = options.enable_pbr and texture_model != "meshy-n1"
        # With input_task_id (the cleaned Image to Image result) there is no image_url: Meshy textures from that input
        # image, which is the cleaned one, never the cluttered original.
        if texture_source == "image" and "image_url" in source:
            payload["texture_image_url"] = source["image_url"]
        elif texture_source == "prompt" and texture_prompt and texture_model != "meshy-n1":
            payload["texture_prompt"] = texture_prompt
    return payload


def texture_conditioning(payload: dict[str, Any]) -> str:
    """What Meshy will actually texture from, by its documented precedence."""
    if not payload.get("should_texture"):
        return "none"
    if payload.get("texture_prompt"):
        return "texture_prompt"
    if payload.get("texture_image_url"):
        return "texture_image_url"
    return "input_image"


class MeshyError(Exception):
    """An error that is safe to surface to API clients."""

    def __init__(self, status_code: int, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or response.reason_phrase
    if isinstance(body, dict):
        return str(body.get("message") or body.get("detail") or body)[:300]
    return str(body)[:300]


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    status = response.status_code
    message = _error_message(response)
    logger.warning("Meshy API error %s: %s", status, message)

    if status == 400:
        raise MeshyError(400, f"Meshy rejected the request: {message}")
    if status in (401, 403):
        raise MeshyError(502, "Meshy rejected the server's API key. Check MESHY_API_KEY in backend/.env.")
    if status == 402:
        raise MeshyError(402, f"Meshy account cannot run this task: {message}")
    if status == 404:
        raise MeshyError(404, "Meshy task not found.")
    if status == 429:
        if message.startswith(("NoMorePendingTasks", "NoMoreConcurrentTasks")):
            raise MeshyError(429, "Too many Meshy tasks are running. Wait for one to finish, then retry.")
        raise MeshyError(429, "Meshy rate limit exceeded. Retry shortly.", _retry_after(response))
    raise MeshyError(502, f"Meshy service error ({status}). Retry later.")


class MeshyService:
    def __init__(self, settings: Settings, api_client: httpx.AsyncClient, download_client: httpx.AsyncClient):
        self._settings = settings
        self._api = api_client
        # Asset downloads go to a different host, so they use a client without the API key.
        self._downloads = download_client

    @classmethod
    def create_clients(cls, settings: Settings) -> tuple[httpx.AsyncClient, httpx.AsyncClient]:
        api = httpx.AsyncClient(
            base_url=settings.meshy_base_url,
            headers={"Authorization": f"Bearer {settings.meshy_api_key.get_secret_value().strip()}"},
            timeout=httpx.Timeout(settings.meshy_request_timeout_seconds, connect=10.0),
        )
        downloads = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(settings.meshy_download_timeout_seconds, connect=10.0),
        )
        return api, downloads

    def _ensure_configured(self) -> None:
        if not self._settings.meshy_configured:
            raise MeshyError(503, "MESHY_API_KEY is not configured on the server (backend/.env).")

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self._ensure_configured()
        try:
            response = await self._api.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise MeshyError(504, "Timed out waiting for Meshy. Retry shortly.") from exc
        except httpx.HTTPError as exc:
            logger.warning("Meshy request failed: %s", exc)
            raise MeshyError(502, "Could not reach Meshy. Check the server's network connection.") from exc
        _raise_for_status(response)
        return response

    async def create_image_to_3d_task(
        self,
        options: GenerationOptions,
        image: bytes | None = None,
        mime_type: str = "image/png",
        image_url: str | None = None,
        texture_prompt: str | None = None,
        target_polycount: int | None = None,
        input_task_id: str | None = None,
        texture_source: TextureSource | None = None,
    ) -> str:
        """`image` is exactly what Meshy meshes (the original photo, or a cleaned one that passed validation).
        `input_task_id`: a SUCCEEDED Image to Image task with exactly one image (the cleaned photo), used as the input
        without re-uploading it. `texture_source`: settings.meshy_texture_source. Invalid combinations are refused
        here (400) before anything is paid for."""
        source: dict[str, Any] = {"input_task_id": input_task_id} if input_task_id else {"image_url": await self._image_source(image, mime_type, image_url)}
        payload = image_to_3d_payload(options, source, texture_prompt, target_polycount, texture_source or self._settings.meshy_texture_source,
                                      **quality_options(self._settings))
        errors = check_image_to_3d(payload)
        if errors:
            raise MeshyError(400, "Invalid Meshy Image to 3D settings: " + "; ".join(errors))
        return await self._create(IMAGE_TO_3D_PATH, payload)

    async def _image_source(self, image: bytes | None, mime_type: str, image_url: str | None) -> str:
        source = (image_url or "").strip()
        if source:
            return source
        if not image:
            raise MeshyError(400, "Provide a photo to convert.")
        encoded = await anyio.to_thread.run_sync(base64.b64encode, image)
        return f"data:{mime_type};base64,{encoded.decode('ascii')}"

    async def create_image_to_image_task(self, prompt: str, image: bytes | None = None, mime_type: str = "image/png", image_url: str | None = None,
                                         aspect_ratio: str | None = None) -> str:
        """Meshy Image to Image: the photo cleaned for 3D (vendor.generate_model, Meshy only)."""
        model = self._settings.meshy_image_model or IMAGE_TO_IMAGE_MODEL
        if model not in IMAGE_TO_IMAGE_MODELS:
            raise MeshyError(400, f"Invalid Meshy Image to Image model {model!r} (MESHY_IMAGE_MODEL).")
        if not prompt.strip():
            raise MeshyError(400, "The Meshy Image to Image prompt is empty.")
        payload = {
            "ai_model": model,
            "prompt": prompt,
            "reference_image_urls": [await self._image_source(image, mime_type, image_url)],
            # Off by default: Meshy's background removal deleted light furniture surfaces (doors came back RGBA 0,0,0,0 and
            # Image to 3D textured the holes dark). The prompt asks for a neutral backdrop instead (generation_prompt).
            "remove_background": self._settings.meshy_remove_background,
            "generate_multi_view": False,  # Image to 3D's input_task_id takes exactly one image
        }
        if aspect_ratio in (GPT_ASPECT_RATIOS if model.startswith("gpt-image") else ASPECT_RATIOS):  # one view: allowed
            payload["aspect_ratio"] = aspect_ratio
        return await self._create(IMAGE_TO_IMAGE_PATH, payload)

    async def download_bytes(self, url: str) -> bytes | None:
        """A small Meshy asset (the cleaned image) in memory; None when it cannot be fetched or is too big. The signed
        URL is never logged."""
        try:
            response = await self._downloads.get(url)
        except httpx.HTTPError as exc:
            logger.warning("Meshy asset download failed (%s)", type(exc).__name__)
            return None
        if not response.is_success or len(response.content) > MAX_REFERENCE_BYTES:
            return None
        return response.content

    async def get_image_to_image_task(self, task_id: str) -> dict[str, Any]:
        response = await self._request("GET", f"{IMAGE_TO_IMAGE_PATH}/{task_id}")
        task = response.json()
        if not isinstance(task, dict) or "status" not in task:
            raise MeshyError(502, "Meshy returned an unexpected task object.")
        return task

    async def _create(self, path: str, payload: dict[str, Any]) -> str:
        logger.info("Meshy POST %s fields=%s payload=%s", path, sorted(payload), _loggable(payload))
        response = await self._request("POST", path, json=payload)
        task_id = response.json().get("result")
        logger.info("Meshy POST %s -> %s task_id=%s", path, response.status_code, task_id)
        if not isinstance(task_id, str) or not task_id:
            raise MeshyError(502, "Meshy returned an unexpected response (missing task ID).")
        return task_id

    async def create_split_task(self, input_task_id: str, parts: str, keep_texture: bool) -> str:
        """Cut a finished image-to-3D model into the named parts; each part is its own object in the GLB."""
        payload = {"input_task_id": input_task_id, "mode": "by_parts", "prompt": parts[:600], "keep_texture": keep_texture}
        return await self._create(SPLIT_PATH, payload)

    async def get_split_task(self, task_id: str) -> dict[str, Any]:
        response = await self._request("GET", f"{SPLIT_PATH}/{task_id}")
        task = response.json()
        if not isinstance(task, dict) or "status" not in task:
            raise MeshyError(502, "Meshy returned an unexpected task object.")
        return task

    async def get_image_to_3d_task(self, task_id: str) -> tuple[dict[str, Any], float | None]:
        response = await self._request("GET", f"{IMAGE_TO_3D_PATH}/{task_id}")
        task = response.json()
        if not isinstance(task, dict) or "status" not in task:
            raise MeshyError(502, "Meshy returned an unexpected task object.")
        return task, _retry_after(response)

    async def download_file(self, url: str, destination: Path) -> int:
        """Stream a Meshy asset to disk atomically. Returns the number of bytes written."""
        temp_path = destination.with_suffix(destination.suffix + ".part")
        written = 0
        try:
            async with self._downloads.stream("GET", url) as response:
                if not response.is_success:
                    raise MeshyError(502, f"Could not download the generated model ({response.status_code}).")
                async with await anyio.open_file(temp_path, "wb") as file:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        written += len(chunk)
                        await file.write(chunk)
            await anyio.Path(temp_path).replace(destination)
        except httpx.HTTPError as exc:
            raise MeshyError(502, "Downloading the generated model failed. Retry to try again.") from exc
        finally:
            if await anyio.Path(temp_path).exists():
                await anyio.Path(temp_path).unlink()
        return written
