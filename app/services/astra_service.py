from __future__ import annotations

import base64
import json
import logging
import struct
from typing import Any

import httpx

from app.services.generation_prompt import astra_prompt
from app.core.config import Settings

logger = logging.getLogger(__name__)


class AstraError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AstraService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    async def generate_glb(self, image: bytes, mime: str, prompt: str | None = None) -> bytes:
        image_b64 = base64.b64encode(image).decode("ascii")
        payload = {
            "model": self.settings.openai_astra_model,
            "store": True,
            "include": ["code_interpreter_call.outputs"],
            "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt or astra_prompt(None),
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime};base64,{image_b64}",
                        },
                    ],
                }
            ],
        }
        body = await self._post_responses(payload)
        glb = await self._extract_glb(body)
        if not glb and body.get("id"):
            body = await self._post_responses(
                {
                    "model": self.settings.openai_astra_model,
                    "store": True,
                    "previous_response_id": body["id"],
                    "include": ["code_interpreter_call.outputs"],
                    "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
                    "input": (
                        "The previous turn did not attach a GLB. Write /mnt/data/model.glb as binary glTF now "
                        "and do not reply with only text."
                    ),
                }
            )
            glb = await self._extract_glb(body)
        if not glb:
            raise AstraError(
                "Astra did not return a GLB file. Use Meshy for photo-to-3D, or upload a GLB directly.",
                502,
            )
        return glb

    async def _post_responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post("https://api.openai.com/v1/responses", json=payload)
        except httpx.TimeoutException as exc:
            raise AstraError("Astra timed out while generating the model.", 504) from exc
        except httpx.HTTPError as exc:
            raise AstraError("Could not reach the OpenAI Astra API.", 502) from exc

        if response.status_code == 401:
            raise AstraError("OPENAI_API_KEY was rejected by OpenAI.", 502)
        if response.status_code >= 400:
            detail = _openai_message(response)
            raise AstraError(detail or f"Astra failed ({response.status_code}).", 502 if response.status_code >= 500 else response.status_code)

        body = response.json()
        response_id = body.get("id") if isinstance(body, dict) else None
        status = str(body.get("status") or "")
        for _ in range(24):
            if status in {"completed", "failed", "cancelled", "incomplete"}:
                break
            if not response_id:
                break
            await _sleep(5)
            polled = await self.client.get(f"https://api.openai.com/v1/responses/{response_id}")
            if polled.status_code >= 400:
                break
            body = polled.json()
            status = str(body.get("status") or "")
        if status == "failed":
            raise AstraError(_response_error(body) or "Astra failed.", 502)
        return body if isinstance(body, dict) else {}

    async def _extract_glb(self, body: dict[str, Any]) -> bytes | None:
        types = [item.get("type") for item in body.get("output") or [] if isinstance(item, dict)]
        logger.info("Astra output types: %s status=%s", types, body.get("status"))

        file_ids: list[str] = []
        container_ids: list[str] = []
        container_files: list[tuple[str, str]] = []
        for node in _walk(body):
            if not isinstance(node, dict):
                continue
            cid = node.get("container_id")
            if isinstance(cid, str) and cid not in container_ids:
                container_ids.append(cid)
            fid = node.get("file_id")
            file_obj = node.get("file")
            if not fid and isinstance(file_obj, dict):
                fid = file_obj.get("id")
            node_id = node.get("id")
            if not fid and isinstance(node_id, str) and node_id.startswith(("file-", "cfile_")):
                fid = node_id
            if isinstance(fid, str) and fid not in file_ids:
                file_ids.append(fid)
            if isinstance(cid, str) and isinstance(node.get("file_id"), str):
                container_files.append((cid, node["file_id"]))

        for container_id in container_ids:
            for file_id, path in await self._list_container_files(container_id):
                container_files.append((container_id, file_id))
                data = await self._download_container_file(container_id, file_id)
                glb = _as_glb(data, path)
                if glb:
                    return glb

        for file_id in file_ids:
            data = await self._download_file(file_id)
            glb = _as_glb(data, file_id)
            if glb:
                return glb
        return None

    async def _download_file(self, file_id: str) -> bytes | None:
        response = await self.client.get(f"https://api.openai.com/v1/files/{file_id}/content")
        if response.status_code >= 400:
            logger.warning("Could not download Astra file %s: %s", file_id, response.status_code)
            return None
        return response.content

    async def _download_container_file(self, container_id: str, file_id: str) -> bytes | None:
        response = await self.client.get(
            f"https://api.openai.com/v1/containers/{container_id}/files/{file_id}/content"
        )
        if response.status_code >= 400:
            logger.warning(
                "Could not download Astra container file %s/%s: %s",
                container_id,
                file_id,
                response.status_code,
            )
            return None
        return response.content

    async def _list_container_files(self, container_id: str) -> list[tuple[str, str]]:
        response = await self.client.get(f"https://api.openai.com/v1/containers/{container_id}/files")
        if response.status_code >= 400:
            logger.warning("Could not list Astra container %s: %s", container_id, response.status_code)
            return []
        payload = response.json()
        items = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        out: list[tuple[str, str]] = []
        for item in items:
            if isinstance(item, dict) and item.get("id"):
                out.append((str(item["id"]), str(item.get("path") or item.get("filename") or "")))
        return out


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk(nested)


def _as_glb(data: bytes | None, name: str = "") -> bytes | None:
    if not data:
        return None
    if data[:4] == b"glTF":
        return data
    lowered = name.lower()
    if lowered.endswith(".glb"):
        return data
    stripped = data.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        try:
            gltf = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(gltf, dict) and "asset" in gltf:
            try:
                return json_gltf_to_glb(gltf)
            except Exception:
                logger.exception("Could not pack glTF JSON from Astra")
                return None
    return None


def json_gltf_to_glb(gltf: dict[str, Any]) -> bytes:
    buffers = gltf.get("buffers") or []
    chunks: list[bytes] = []
    for buffer in buffers:
        uri = buffer.get("uri")
        if not isinstance(uri, str) or not uri.startswith("data:"):
            continue
        payload = uri.split(",", 1)[1]
        raw = base64.b64decode(payload)
        buffer["byteLength"] = len(raw)
        buffer.pop("uri", None)
        chunks.append(raw)
    binary = b"".join(chunks)
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_padded = binary + (b"\x00" * ((4 - len(binary) % 4) % 4))
    total = 12 + 8 + len(json_bytes) + (8 + len(bin_padded) if binary else 0)
    header = struct.pack("<4sII", b"glTF", 2, total)
    json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    if binary:
        bin_chunk = struct.pack("<I4s", len(bin_padded), b"BIN\x00") + bin_padded
        return header + json_chunk + bin_chunk
    return header + json_chunk


async def _sleep(seconds: float) -> None:
    import anyio

    await anyio.sleep(seconds)


def _response_error(body: dict[str, Any]) -> str | None:
    error = body.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return None


def _openai_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except json.JSONDecodeError:
        return response.text[:300]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if body.get("message"):
            return str(body["message"])
    return ""
