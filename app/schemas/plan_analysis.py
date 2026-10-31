"""Architectural JSON from the vision model (app/services/plan/vision_analysis.py).

Coordinates are NORMALISED image space: (0, 0) top-left, x right, y down, 1 = image width/height.
The backend converts them to pixels (× image size) and, after scale estimation, to metres. Sizes the
model gives are hints: `dimension_source` says whether they were read off the drawing ("annotated"),
inferred from standard objects ("inferred") or are not known ("unknown").
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Point = Annotated[list[float], Field(min_length=2, max_length=2)]  # normalised [x, y]; lists, not tuples (JSON-schema friendly)
DimensionSource = Literal["annotated", "inferred", "unknown"]
RoomType = Literal["living", "kitchen", "dining", "bedroom", "bathroom", "hall", "balcony", "closet", "laundry", "office", "other"]
FurnitureType = Literal[
    "bed", "sofa", "table", "dining_table", "chair", "wardrobe", "cabinet", "countertop", "stove", "fridge",
    "sink", "toilet", "shower", "bathtub", "washer", "desk", "tv", "other",
]
ReferenceKind = Literal["dimension_annotation", "door", "bed", "bathtub", "countertop", "other"]


class Room(BaseModel):
    id: str
    name: str
    type: RoomType = "other"
    polygon: list[Point] = Field(description="Room boundary, normalised, clockwise or counter-clockwise")
    estimated_width_m: float | None = None
    estimated_length_m: float | None = None
    area_m2: float | None = None
    dimension_source: DimensionSource = "unknown"
    dimension_text: str | None = Field(default=None, description="The size exactly as written on the plan, if any")
    connects_to: list[str] = Field(default_factory=list, description="Ids of rooms reachable through a door or open side")
    confidence: float = Field(default=0.5, ge=0, le=1)


class Wall(BaseModel):
    id: str
    start: Point
    end: Point
    exterior: bool = False
    thickness_m: float | None = None
    rooms: list[str] = Field(default_factory=list, description="Room ids on either side (shared wall = two)")
    confidence: float = Field(default=0.5, ge=0, le=1)


class Door(BaseModel):
    id: str
    start: Point = Field(description="One jamb of the opening")
    end: Point = Field(description="The other jamb")
    width_m: float | None = None
    swing: Literal["left", "right", "sliding", "double", "none", "unknown"] = "unknown"
    opens_into: str | None = None
    connects: list[str] = Field(default_factory=list, description="Room ids (or 'outside') on both sides")
    wall_id: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)


class Window(BaseModel):
    id: str
    start: Point
    end: Point
    width_m: float | None = None
    wall_id: str | None = None
    room_id: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)


class Furniture(BaseModel):
    id: str
    type: FurnitureType = "other"
    room_id: str | None = None
    bbox: Annotated[list[float], Field(min_length=4, max_length=4, description="x_min, y_min, x_max, y_max, normalised")]
    orientation_deg: float = 0.0
    width_m: float | None = None
    depth_m: float | None = None
    dimension_source: DimensionSource = "inferred"
    confidence: float = Field(default=0.5, ge=0, le=1)


class ScaleReference(BaseModel):
    """Something whose real size is known: the backend measures `start`→`end` in pixels itself."""

    kind: ReferenceKind
    element_id: str | None = None
    start: Point
    end: Point
    real_m: float = Field(gt=0)
    dimension_source: DimensionSource = "inferred"
    text: str | None = Field(default=None, description="Annotation text, for dimension_annotation")
    confidence: float = Field(default=0.5, ge=0, le=1)


class LlmPlan(BaseModel):
    """What the model returns."""

    rooms: list[Room] = Field(default_factory=list)
    walls: list[Wall] = Field(default_factory=list)
    doors: list[Door] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    furniture: list[Furniture] = Field(default_factory=list)
    scale_references: list[ScaleReference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalysisScale(BaseModel):
    pixels_per_meter: float | None = None
    confidence: float = 0.0
    source: str = "pending_estimation"


class ImageSize(BaseModel):
    width: int
    height: int


class PlanAnalysis(LlmPlan):
    """Stored with the plan for later editing: the model output after validation and correction."""

    image: ImageSize
    scale: AnalysisScale = Field(default_factory=AnalysisScale)
    model: str = ""
    schema_version: str = ""
    cached: bool = False
    issues: list[str] = Field(default_factory=list, description="Validation findings left after correction")
