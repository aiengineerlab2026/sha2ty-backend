"""A price range for one piece of a customised model, from what vendors bid on earlier requests for the same model.

Each earlier bid becomes a price per piece (a total bid divided by the pieces it covers), then is moved to the size
asked now by surface area (size ratio to the power 2/3: boards and fabric grow with the surface, not the volume).
Only with enough bids (MIN_BIDS from MIN_REQUESTS requests) is a range given; otherwise there is none and the editor
says the price is set by vendors' bids. Nothing here is a quote: vendors set the price.
"""

from __future__ import annotations

from typing import Any

MIN_BIDS = 3
MIN_REQUESTS = 2


def _volume(d: dict[str, Any] | None) -> float | None:
    try:
        v = float(d["width"]) * float(d["height"]) * float(d["depth"])  # type: ignore[index]
    except (TypeError, KeyError, ValueError):
        return None
    return v if v > 0 else None


def price_guide(requests: list[dict[str, Any]], bids: list[dict[str, Any]], size: dict[str, Any] | None) -> dict[str, Any]:
    """`requests`: earlier requests ({id, quantity, dimensions}); `bids`: their bids ({task_id, price, price_basis,
    quantity, status}); `size`: the size asked now. Returns {count, requests, low, high, median} or {count, requests,
    low: None, …} when the data is too thin."""
    by_task = {str(r["id"]): r for r in requests}
    target = _volume(size)
    per_piece: list[float] = []
    used: set[str] = set()
    for bid in bids:
        req = by_task.get(str(bid.get("task_id")))
        if not req or bid.get("status") == "withdrawn":
            continue
        try:
            price = float(bid["price"])
        except (TypeError, KeyError, ValueError):
            continue
        pieces = int(bid.get("quantity") or req.get("quantity") or 1) or 1
        each = price if bid.get("price_basis") == "per_item" else price / pieces
        then = _volume(req.get("dimensions"))
        if target and then:
            each *= (target / then) ** (2 / 3)
        if each > 0:
            per_piece.append(each)
            used.add(str(req["id"]))
    per_piece.sort()
    out: dict[str, Any] = {"count": len(per_piece), "requests": len(used), "low": None, "high": None, "median": None}
    if len(per_piece) >= MIN_BIDS and len(used) >= MIN_REQUESTS:
        mid = len(per_piece) // 2
        median = per_piece[mid] if len(per_piece) % 2 else (per_piece[mid - 1] + per_piece[mid]) / 2
        # Whole tens of pounds: a guide, not a quote.
        out.update(low=round(per_piece[0], -1), high=round(per_piece[-1], -1), median=round(median, -1))
    return out
