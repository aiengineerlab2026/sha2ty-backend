"""Rebuild a saved model's structural (carpenter) model with the current rules, and repair its photo links.

    python scripts/rebuild_structural.py <model id>            # says what would change; writes storage/repaired/<id>.carpenter.glb
    python scripts/rebuild_structural.py <model id> --apply    # uploads it and saves the model's component document

What changes (the visual model is never touched):
  - the photo-reading links (`analysisId`, the vision names) whose surface layer contradicts the part's own name: a
    mirror's backing named "door mirror", a mirror named "door frame" (mesh_analysis.layer). Those parts get their
    own file name back; a link that agrees stays.
  - the structural model, rebuilt from the saved photo reading at its saved size: glass and mirror panes keep their
    own thickness in front of the wood behind them (carpenter._thicken).
--apply keeps the previous structural file's URL in `carpenter.previousUrl`, so the change can be undone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api.vendor import structural_model, with_structural  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.services.auth import service_headers  # noqa: E402
from app.services.cdn import Cdn  # noqa: E402
from app.services.mesh_analysis import layer  # noqa: E402


def relink(doc: dict[str, Any]) -> list[str]:
    """Drops photo links that name another layer of the panel than the part's own id says. In place; what changed."""
    by_id = {c.get("id"): c for c in (doc.get("analysis") or {}).get("components") or []}
    changed = []
    for part in doc.get("components") or []:
        if part.get("naming") != "vision" or not part.get("analysisId"):
            continue
        own = layer(str(part.get("id") or ""))
        photo = by_id.get(part["analysisId"]) or {}
        said = layer(f"{part.get('name') or ''} {photo.get('material') or ''}") or ({"mirror": "mirror", "glass": "glass"}.get(str(photo.get("materialClass") or "")))
        clash = said != own if said in ("mirror", "glass") else own in ("mirror", "glass")
        if not clash:
            continue
        name = re.sub(r"\s+", " ", re.sub(r"[_\-.]+", " ", str(part["id"]))).strip().capitalize()
        changed.append(f"{part['id']}: {part.get('name')!r} ({part['analysisId']}) → {name!r}")
        part.update(name=name, naming="file")
        for key in ("analysisId", "nameAr"):
            part.pop(key, None)
    return changed


def details_of(doc: dict[str, Any]) -> dict[str, Any] | None:
    """The overall size the structural model was built at (its saved report), as the spec's details."""
    size = (doc.get("carpenter") or {}).get("sizeM") or (doc.get("structural") or {}).get("sizeM")
    if not size:
        return None
    known = (doc.get("carpenter") or {}).get("scaleKnown") or (doc.get("structural") or {}).get("scaleKnown")
    return {"width_m": size["width"], "height_m": size["height"], "depth_m": size["depth"], **({"dimension_source": "annotated"} if known else {})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_id")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    settings = get_settings()
    rest = f"{settings.supabase_url.rstrip('/')}/rest/v1/models"
    with httpx.Client(timeout=60) as http:
        rows = http.get(rest, params={"id": f"eq.{args.model_id}", "select": "*"}, headers=service_headers(settings)).json()
    if not rows:
        sys.exit("No such model.")
    row = rows[0]
    doc = json.loads(json.dumps(row.get("components") or {}))
    links = relink(doc)
    print(f"{len(links)} photo link(s) on the wrong layer" + (":" if links else "."))
    for line in links:
        print(f"  {line}")
    details = details_of(doc)
    analysis = doc.get("analysis")
    rebuilt = None
    if analysis and details:
        spec = {"details": details, "analysis": analysis}
        tag = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
        owner = str(row["glb_url"]).split("?", 1)[0].rsplit("/", 2)[-2]
        cdn = Cdn(settings) if args.apply else _Local(ROOT / "storage/repaired")
        rebuilt = structural_model(spec, row.get("category"), row.get("kind") or "", cdn, f"{owner}/{args.model_id}.carpenter.{tag}.glb")
        print("Structural model: " + ("rebuilt" if rebuilt.get("built") else f"not built ({'; '.join(rebuilt.get('uncertain') or [])})"))
    else:
        print("Structural model: none saved for this model (no photo reading or size): nothing to rebuild.")
    if not args.apply:
        print("Run again with --apply to save it on the model.")
        return
    if rebuilt and rebuilt.get("built"):
        previous = (doc.get("carpenter") or {}).get("url")
        doc = with_structural(doc, rebuilt, {"details": details, "analysis": analysis})
        if previous and doc.get("carpenter"):
            doc["carpenter"]["previousUrl"] = previous
    with httpx.Client(timeout=30) as http:
        r = http.patch(rest, params={"id": f"eq.{args.model_id}"}, json={"components": doc}, headers={**service_headers(settings), "Prefer": "return=minimal"})
        r.raise_for_status()
    print("Applied.")


class _Local:
    """A dry run's CDN: the file is written under storage/repaired, nothing is uploaded."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def upload(self, key: str, data: bytes, _type: str) -> str:
        out = self.folder / key.replace("/", "_")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)


if __name__ == "__main__":
    main()
