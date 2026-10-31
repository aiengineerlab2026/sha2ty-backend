from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from app.domain.errors import DomainError

VendorStatus = Literal["pending", "approved", "rejected"]
INITIAL_STATUS: VendorStatus = "approved"  # vendors sell right after signup; the admin can suspend later


@dataclass(frozen=True)
class Vendor:
    id: str
    name: str
    location: str
    full_name: str = ""
    email: str = ""
    lat: float | None = None
    lng: float | None = None
    status: VendorStatus = INITIAL_STATUS

    @classmethod
    def register(cls, id: str, name: str, location: str, **extra: object) -> Vendor:
        name, location = name.strip(), location.strip()
        if not name or not location:
            raise DomainError("Company name and location are required.")
        return cls(id=id, name=name, location=location, **extra)  # type: ignore[arg-type]

    def row(self) -> dict[str, object]:
        return asdict(self)
