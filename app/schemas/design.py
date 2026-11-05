"""The shared design document (`design_versions.data`), version sha2ty.design/1.

Mirrored by frontend/lib/designs/schema.ts and mobile/src/lib/designs.ts; change all three together and bump
DESIGN_SCHEMA when a change is not backwards compatible. Metres are the only unit stored.

- item: one catalogue model as the customer customised it. `source` is the catalogue original at the time the
  design started; it is filled by the server and never changed by a save (the vendor's listing is not touched).
- room / apartment: a snapshot of the planner's plan (geometry, placed items, camera), taken by the server from
  the linked floor plan when a version is saved.
"""

from __future__ import annotations

import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DESIGN_SCHEMA = "sha2ty.design/1"
DesignKind = Literal["item", "room", "apartment"]

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class Dimensions(BaseModel):
    """Overall size in metres (width × height × depth)."""

    model_config = ConfigDict(extra="forbid")
    width: float = Field(gt=0, le=50)
    height: float = Field(gt=0, le=50)
    depth: float = Field(gt=0, le=50)


MaterialUnit = Literal["m", "m2", "m3", "sheet", "kg", "l"]


class MaterialFinish(BaseModel):
    """A decorative surface on a structural board (MDF with walnut veneer)."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(max_length=64)
    name: str = Field(max_length=200)
    wood_type: str | None = Field(default=None, max_length=32)
    price: float | None = Field(default=None, ge=0)
    unit: MaterialUnit


class PartMaterial(BaseModel):
    """The fabric or wood chosen for a part, with its estimate (frontend/lib/designs/materials.ts PartMaterial)."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(max_length=64)
    kind: Literal["fabric", "wood", "paint"]
    name: str = Field(max_length=200)
    wood_type: str | None = Field(default=None, max_length=32)
    thickness_mm: float | None = Field(default=None, gt=0, le=500)
    price: float | None = Field(default=None, ge=0)
    unit: MaterialUnit
    sheet: tuple[float, float] | None = None
    finish: MaterialFinish | None = None
    # Paint: the palette colour, how far one kg / litre goes per coat, and the coats.
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    color_name: str | None = Field(default=None, max_length=80)
    coverage_m2: float | None = Field(default=None, gt=0, le=100)
    coats: int | None = Field(default=None, ge=1, le=5)
    quantity: float | None = Field(default=None, ge=0)
    finish_quantity: float | None = Field(default=None, ge=0)
    total: float | None = Field(default=None, ge=0)


class MaterialEdit(BaseModel):
    """One part's material change (frontend/lib/vendor/parts.ts MaterialEdit)."""

    model_config = ConfigDict(extra="forbid")
    color: str | None = None
    mapUrl: str | None = None
    repeat: tuple[float, float] | None = None
    rotation: float | None = None
    offset: tuple[float, float] | None = None
    roughness: float | None = Field(default=None, ge=0, le=1)
    metalness: float | None = Field(default=None, ge=0, le=1)
    tile: float | None = Field(default=None, gt=0, le=50)
    mapping: Literal["uv", "box"] | None = None
    normalUrl: str | None = Field(default=None, max_length=2000)
    material: PartMaterial | None = None

    @field_validator("color")
    @classmethod
    def _hex(cls, value: str | None) -> str | None:
        if value is not None and not _HEX.match(value):
            raise ValueError("color must be #rrggbb")
        return value


class Component(BaseModel):
    """A part that exists in the GLB (services/glb_parts.py); extra keys (mesh, node, box…) are kept as written."""

    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1, max_length=120)
    name: str = Field(max_length=120)
    editable: bool = True


class PartDocument(BaseModel):
    model_config = ConfigDict(extra="allow")
    modelType: str | None = None
    separated: bool | None = None
    components: list[Component] = Field(default_factory=list, max_length=500)
    limits: str | None = None


class SourceModel(BaseModel):
    """The original a design started from (server-filled): a catalogue listing, or the customer's own model."""

    model_config = ConfigDict(extra="forbid")
    model_id: UUID
    title: str
    vendor_id: UUID | None = None
    # Who made it: a vendor (requests go to them only), the platform's own catalogue or the customer's own model
    # made from their photo (user_models; both open to every approved vendor).
    source_type: Literal["vendor", "platform", "user"] | None = None
    glb_url: str
    source_glb_url: str | None = None
    width_m: float | None = None
    height_m: float | None = None
    depth_m: float | None = None


class PartPose(BaseModel):
    """One mesh moved, turned or scaled inside a customer design. Degrees and metres; scale is a multiplier."""

    model_config = ConfigDict(extra="forbid")
    scale: tuple[float, float, float] | None = None
    position: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None


class ItemEdits(BaseModel):
    """What a save of an item design may change."""

    model_config = ConfigDict(extra="forbid")
    dimensions: Dimensions | None = None
    components: PartDocument | None = None
    material_edits: dict[str, MaterialEdit] = Field(default_factory=dict, max_length=500)
    part_poses: dict[str, PartPose] | None = None
    # Parts the customer removed from the model (left out of its GLB, thumbnail and requests).
    removed_parts: list[str] | None = Field(default=None, max_length=500)
    # The vendor's part variant picked per part (part id → variant id, from components[].variants).
    part_variants: dict[str, str] | None = Field(default=None, max_length=500)
    # The customer's own .glb this design is built on instead of the source's file (their upload: checked on save).
    base_glb_url: str | None = Field(default=None, max_length=2000)
    # How the overall size was last changed: one factor for all axes, or each axis on its own (stretches parts).
    resize_mode: Literal["proportional", "independent"] | None = None
    # dimensions ÷ the catalogue size per axis; written by the server from `dimensions` (a sent value is replaced).
    scale: tuple[float, float, float] | None = None


class ItemDesign(ItemEdits):
    source: SourceModel


class PlanSnapshot(BaseModel):
    """The planner's plan as saved (floor_plans row)."""

    model_config = ConfigDict(extra="forbid")
    plan_id: UUID | None = None
    name: str = ""
    source: str = "dimensions"
    source_image_url: str | None = None
    geometry: dict[str, Any] = Field(default_factory=dict)
    items: list[dict[str, Any]] = Field(default_factory=list, max_length=2000)
    camera: dict[str, Any] | None = None
    warnings: list[Any] = Field(default_factory=list)

    @property
    def room_count(self) -> int:
        rooms = self.geometry.get("rooms")
        return len(rooms) if isinstance(rooms, list) else 0


class DesignData(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_: Literal["sha2ty.design/1"] = Field(default=DESIGN_SCHEMA, alias="schema")
    kind: DesignKind
    units: Literal["m"] = "m"
    item: ItemDesign | None = None
    plan: PlanSnapshot | None = None

    @model_validator(mode="after")
    def _shape(self) -> DesignData:
        if self.kind == "item":
            if self.item is None or self.plan is not None:
                raise ValueError("An item design has one item and no plan.")
        else:
            if self.plan is None or self.item is not None:
                raise ValueError("A room or apartment design has a plan and no single item.")
            if self.kind == "room" and self.plan.room_count > 1:
                raise ValueError(f"This plan has {self.plan.room_count} rooms: save it as an apartment.")
        return self

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


def plan_kind(snapshot: PlanSnapshot) -> DesignKind:
    return "apartment" if snapshot.room_count > 1 else "room"


# ---- request bodies ----


class FromUserModelBody(BaseModel):
    """A design from one of the customer's own models (user_models): sizes, parts and materials edited like a
    customised catalogue model, and requested from every vendor."""

    user_model_id: UUID
    title: str = Field(default="", max_length=200)


class FromModelBody(BaseModel):
    """The first save of a customised catalogue model: the design and its v1 (with the edits) in one call.
    `client_token` makes a repeated first save (double click, lost connection) return the same design."""

    model_id: UUID
    title: str = Field(default="", max_length=200)
    item: ItemEdits | None = None
    glb_url: str | None = None
    thumbnail_url: str | None = None
    note: str = Field(default="", max_length=500)
    client_token: UUID | None = None


class FromPlanBody(BaseModel):
    plan_id: UUID
    title: str = Field(default="", max_length=200)
    kind: Literal["room", "apartment"] | None = None  # default: from the plan's room count


class SaveVersionBody(BaseModel):
    """Item designs send their edits (and the exported GLB / thumbnail they uploaded); room and apartment designs
    send nothing but a note: the server snapshots the linked plan."""

    item: ItemEdits | None = None
    glb_url: str | None = None
    thumbnail_url: str | None = None
    note: str = Field(default="", max_length=500)
    client_token: UUID | None = None


class RestoreBody(BaseModel):
    number: int = Field(gt=0)
    client_token: UUID | None = None


class DuplicateBody(BaseModel):
    title: str = Field(default="", max_length=200)
    number: int | None = Field(default=None, gt=0)  # which version to copy (default: the current one)


class RenameBody(BaseModel):
    title: str = Field(min_length=1, max_length=200)
