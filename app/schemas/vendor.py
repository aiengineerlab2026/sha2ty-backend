from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ResourceId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
HexColor = Annotated[str, StringConstraints(pattern=r"^#[0-9a-fA-F]{6}$")]
Vec3 = tuple[float, float, float]

MeshyTaskStatus = Literal["PENDING", "IN_PROGRESS", "SUCCEEDED", "FAILED", "CANCELED"]


class GenerationOptions(BaseModel):
    should_texture: bool = True
    enable_pbr: bool = False

    @model_validator(mode="after")
    def pbr_requires_texture(self) -> "GenerationOptions":
        if self.enable_pbr and not self.should_texture:
            raise ValueError("enable_pbr requires should_texture to be true")
        return self


class GenerateResponse(BaseModel):
    task_id: str
    status: MeshyTaskStatus = "PENDING"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: MeshyTaskStatus
    progress: int = Field(ge=0, le=100)
    preceding_tasks: int | None = None
    elapsed_seconds: float | None = None
    retry_after_seconds: float | None = None
    model_id: str | None = None
    model_url: str | None = None
    error: str | None = None


class CameraState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: Vec3
    target: Vec3
    zoom: float = Field(default=1.0, gt=0)


class EditorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    background: HexColor = "#1f2128"
    show_grid: bool = True
    show_axes: bool = True
    shadows: bool = True
    camera_mode: Literal["perspective", "orthographic"] = "perspective"
    transform_space: Literal["world", "local"] = "world"
    snap: bool = False


class ProjectMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    editor: EditorSettings = EditorSettings()
    camera: CameraState | None = None
    object_count: int = Field(default=0, ge=0)
    source_model_ids: list[ResourceId] = Field(default_factory=list, max_length=100)


ModelKind = Literal["generated", "project"]


class ModelRecord(BaseModel):
    id: ResourceId
    kind: ModelKind
    name: str
    created_at: datetime
    updated_at: datetime
    size_bytes: int
    source_task_id: str | None = None
    project: ProjectMetadata | None = None


class ModelResponse(ModelRecord):
    file_url: str


class TaskRecord(BaseModel):
    task_id: str
    created_at: datetime
    source_filename: str
    options: GenerationOptions


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    meshy_configured: bool
    astra_configured: bool = False
    supabase_configured: bool = False
