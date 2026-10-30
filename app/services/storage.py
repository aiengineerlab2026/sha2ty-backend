"""File-system persistence for generated models, saved projects and task records."""

import asyncio
from pathlib import Path

import anyio
from fastapi import UploadFile

from app.schemas.vendor import ModelKind, ModelRecord, TaskRecord

GLB_MAGIC = b"glTF"


class StorageError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class Storage:
    def __init__(self, root: Path):
        self.models_dir = root / "models"
        self.tasks_dir = root / "tasks"
        self._locks: dict[str, asyncio.Lock] = {}

    def init(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    # Callers must validate ids against ResourceId before building paths.
    def model_file(self, model_id: str) -> Path:
        return self.models_dir / f"{model_id}.glb"

    def _model_meta(self, model_id: str) -> Path:
        return self.models_dir / f"{model_id}.json"

    def _task_meta(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    @staticmethod
    async def _write_atomic(path: Path, text: str) -> None:
        temp = anyio.Path(path.with_suffix(path.suffix + ".tmp"))
        await temp.write_text(text, encoding="utf-8")
        await temp.replace(path)

    async def read_model(self, model_id: str) -> ModelRecord | None:
        path = anyio.Path(self._model_meta(model_id))
        if not await path.exists() or not await anyio.Path(self.model_file(model_id)).exists():
            return None
        return ModelRecord.model_validate_json(await path.read_text(encoding="utf-8"))

    async def write_model(self, record: ModelRecord) -> None:
        await self._write_atomic(self._model_meta(record.id), record.model_dump_json(indent=2))

    async def delete_model(self, model_id: str) -> bool:
        removed = False
        for path in (self.model_file(model_id), self._model_meta(model_id)):
            apath = anyio.Path(path)
            if await apath.exists():
                await apath.unlink()
                removed = True
        return removed

    async def list_models(self, kind: ModelKind | None = None) -> list[ModelRecord]:
        records: list[ModelRecord] = []
        async for path in anyio.Path(self.models_dir).glob("*.json"):
            try:
                record = ModelRecord.model_validate_json(await path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if kind is None or record.kind == kind:
                records.append(record)
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records

    async def save_glb_upload(self, model_id: str, upload: UploadFile, max_bytes: int) -> int:
        destination = self.model_file(model_id)
        temp = anyio.Path(destination.with_suffix(".glb.part"))
        written = 0
        try:
            async with await anyio.open_file(temp, "wb") as file:
                while chunk := await upload.read(1024 * 1024):
                    if written == 0 and not chunk.startswith(GLB_MAGIC):
                        raise StorageError(400, "The uploaded scene is not a binary glTF (GLB) file.")
                    written += len(chunk)
                    if written > max_bytes:
                        raise StorageError(413, "The scene file is too large.")
                    await file.write(chunk)
            if written == 0:
                raise StorageError(400, "The uploaded scene file is empty.")
            await temp.replace(destination)
        finally:
            if await temp.exists():
                await temp.unlink()
        return written

    async def read_task(self, task_id: str) -> TaskRecord | None:
        path = anyio.Path(self._task_meta(task_id))
        if not await path.exists():
            return None
        return TaskRecord.model_validate_json(await path.read_text(encoding="utf-8"))

    async def write_task(self, record: TaskRecord) -> None:
        await self._write_atomic(self._task_meta(record.task_id), record.model_dump_json(indent=2))
