"""Client for Meshy's Image to 3D REST API (https://docs.meshy.ai/en/api/image-to-3d)."""

import base64
import logging
from pathlib import Path
from typing import Any

import anyio
import httpx

from app.core.config import Settings
from app.schemas.vendor import GenerationOptions

logger = logging.getLogger(__name__)

IMAGE_TO_3D_PATH = "/image-to-3d"


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
    ) -> str:
        source = (image_url or "").strip()
        if not source:
            if not image:
                raise MeshyError(400, "Provide a photo to convert.")
            encoded = await anyio.to_thread.run_sync(base64.b64encode, image)
            source = f"data:{mime_type};base64,{encoded.decode('ascii')}"
        payload: dict[str, Any] = {
            "image_url": source,
            "ai_model": "latest",
            "target_formats": ["glb"],
            "should_texture": options.should_texture,
        }
        if options.should_texture:
            payload["enable_pbr"] = options.enable_pbr
            if texture_prompt:
                # What the object is and "keep the photo's colours" (see generation_prompt).
                payload["texture_prompt"] = texture_prompt

        response = await self._request("POST", IMAGE_TO_3D_PATH, json=payload)
        task_id = response.json().get("result")
        if not isinstance(task_id, str) or not task_id:
            raise MeshyError(502, "Meshy returned an unexpected response (missing task ID).")
        return task_id

    async def get_image_to_3d_task(self, task_id: str) -> tuple[dict[str, Any], float | None]:
        response = await self._request("GET", f"{IMAGE_TO_3D_PATH}/{task_id}")
        task = response.json()
        if not isinstance(task, dict) or "status" not in task:
            raise MeshyError(502, "Meshy returned an unexpected task object.")
        return task, _retry_after(response)

    async def download_file(self, url: str, destination: Path, max_bytes: int) -> int:
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
                        if written > max_bytes:
                            raise MeshyError(502, "The generated model is larger than the server allows.")
                        await file.write(chunk)
            await anyio.Path(temp_path).replace(destination)
        except httpx.HTTPError as exc:
            raise MeshyError(502, "Downloading the generated model failed. Retry to try again.") from exc
        finally:
            if await anyio.Path(temp_path).exists():
                await anyio.Path(temp_path).unlink()
        return written
