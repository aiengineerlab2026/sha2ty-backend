"""What the vision model reads off a vendor product photo before 3D generation (services/product_details.py).

Boxes are NORMALISED image coordinates (x_min, y_min, x_max, y_max; 0..1, origin top-left). Sizes are
estimates from the photo unless `dimension_source` is "annotated" (written on the image): a photo
shows neither hidden geometry nor exact size.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Box = Annotated[list[float], Field(min_length=4, max_length=4, description="x_min, y_min, x_max, y_max, normalised 0..1")]


class Component(BaseModel):
    name: str = Field(description="A visible part: seat, backrest, leg, door, handle, frame, shade…")
    bbox: Box | None = None
    material: str | None = None
    color: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)


class ProductDetails(BaseModel):
    object_type: str = Field(description="What the main object is, e.g. 'three-seat sofa'")
    matches_category: bool = Field(default=True, description="False when the photo shows something else than the vendor's category")
    # The catalogue listing, used when the vendor leaves the model's name or description empty.
    name: str = Field(default="", description="In the requested listing language: a short catalogue name for the product")
    description: str = Field(default="", description="In the requested listing language: one or two plain sentences for a shop listing (style, materials, colours, notable parts); only what is visible")
    bbox: Box | None = Field(default=None, description="The main object's boundary in the photo")
    orientation: Literal["front", "three_quarter", "side", "back", "top", "unknown"] = "unknown"
    proportions: Annotated[list[float], Field(min_length=3, max_length=3)] | None = Field(
        default=None, description="width : depth : height, scaled so the largest is 1"
    )
    width_m: float | None = None
    depth_m: float | None = None
    height_m: float | None = None
    dimension_source: Literal["annotated", "inferred", "unknown"] = "inferred"
    materials: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    components: list[Component] = Field(default_factory=list)
    hidden: str | None = Field(default=None, description="What the photo cannot show (back, underside)")
    confidence: float = Field(default=0.5, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    model: str = ""  # filled by the backend: which vision model read it
