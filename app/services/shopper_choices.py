"""What a customer may change on a vendor's model: the parts the vendor lets them add (PartDocument.addable, by part
type, with an optional price difference) and the part variants the vendor offers (variants[].offered, on unless
switched off). Anything else could not be made or priced by the vendor, so a save with it is refused."""

from __future__ import annotations

from typing import Any


def allowed_additions(components: dict[str, Any] | None) -> dict[str, float | None]:
    """Part type → price difference (EGP, None when not priced) of every part the vendor lets customers add."""
    out: dict[str, float | None] = {}
    for entry in (components or {}).get("addable") or []:
        if isinstance(entry, dict) and isinstance(entry.get("type"), str):
            delta = entry.get("price_delta")
            out[entry["type"]] = float(delta) if isinstance(delta, (int, float)) else None
    return out


def offered_variants(components: dict[str, Any] | None) -> dict[str, set[str]]:
    """Part id → the ids of its variants offered to customers."""
    return {
        str(c.get("id")): {str(v.get("id")) for v in c.get("variants") or [] if isinstance(v, dict) and v.get("offered", True) is not False}
        for c in (components or {}).get("components") or []
        if isinstance(c, dict)
    }


def shopper_choice_errors(item: dict[str, Any], components: dict[str, Any] | None) -> list[str]:
    """Why a customer's design of this model cannot be saved: added parts or variants the vendor does not offer."""
    errors: list[str] = []
    allowed = allowed_additions(components)
    for spec in ((item.get("components") or {}).get("added")) or []:
        kind = spec.get("type") if isinstance(spec, dict) else None
        if kind not in allowed:
            errors.append(f"The vendor does not offer added “{kind}” parts for this model.")
    offered = offered_variants(components)
    for part, variant in (item.get("part_variants") or {}).items():
        if str(variant) not in offered.get(str(part), set()):
            errors.append(f"The vendor does not offer this variant of “{part}”.")
    return errors
