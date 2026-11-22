"""Repair a saved model whose panels show photographed lighting tiled as rectangular patches (services/surface_finish).

    python scripts/repair_surfaces.py <model id>            # writes storage/repaired/<id>.glb and says what changes
    python scripts/repair_surfaces.py <model id> --apply    # uploads it and points the model at it

--apply keeps the previous file as the model's source_glb_url (when it has none), so the change can be undone.
Geometry, names, UVs, other textures and the model's saved material edits are not touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.cdn import Cdn  # noqa: E402
from app.services.auth import service_headers  # noqa: E402
from app.services.surface_finish import flatten  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_id")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    settings = get_settings()
    rest = f"{settings.supabase_url.rstrip('/')}/rest/v1/models"
    with httpx.Client(timeout=60) as http:
        rows = http.get(rest, params={"id": f"eq.{args.model_id}", "select": "id,glb_url,source_glb_url"}, headers=service_headers(settings)).json()
        if not rows:
            sys.exit("No such model.")
        row = rows[0]
        data = http.get(row["glb_url"]).content
    fixed, changed = flatten(data)
    if not changed:
        print("Nothing to repair: no repeated colour texture carries photographed lighting.")
        return
    out = ROOT / "storage/repaired" / f"{args.model_id}.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(fixed)
    print(f"{len(changed)} material(s) → their own even colour; written to {out}")
    for c in changed[:10]:
        print(f"  {c['material']}: {c['colour']} (shading {c['shading']}, tiled ×{c['tiles']})")
    if not args.apply:
        print("Run again with --apply to put it on the model.")
        return
    owner = row["glb_url"].split("?", 1)[0].rsplit("/", 2)[-2]  # the folder the current file is in
    url = Cdn(settings).upload(f"{owner}/{args.model_id}.surfaces.glb", fixed, "model/gltf-binary")
    patch = {"glb_url": url, **({} if row.get("source_glb_url") else {"source_glb_url": row["glb_url"]})}
    with httpx.Client(timeout=30) as http:
        r = http.patch(rest, params={"id": f"eq.{args.model_id}"}, json=patch, headers={**service_headers(settings), "Prefer": "return=minimal"})
        r.raise_for_status()
    print(f"Applied: glb_url → {url}")


if __name__ == "__main__":
    main()
