"""Rewrite saved file URLs that name a local dev server (http://localhost:8010/api/files/…, :8000, 127.0.0.1) to the
API's public origin, in every table the Supabase REST API exposes (plain columns and JSON documents alike).

Only the origin in front of /api/files/ changes; the file key after it stays byte for byte, so every file is the same
R2 object. Idempotent: a second run finds nothing. Dry run unless --apply; with --apply every row is first written to
--backup (JSON, one file per table) and then PATCHed by its id.

    .venv/bin/python scripts/fix_file_urls.py --origin https://sha2ty-773972750372.europe-west1.run.app
    .venv/bin/python scripts/fix_file_urls.py --origin … --apply --backup backups/file-urls-2026-11-24
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import get_settings  # noqa: E402

LOCAL = re.compile(r"http://(?:localhost|127\.0\.0\.1)(?::\d+)?(?=/api/files/)")


def rewrite(value, origin: str):
    """The value with every local file origin replaced (strings anywhere inside lists and dicts)."""
    if isinstance(value, str):
        return LOCAL.sub(origin, value)
    if isinstance(value, list):
        return [rewrite(v, origin) for v in value]
    if isinstance(value, dict):
        return {k: rewrite(v, origin) for k, v in value.items()}
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", required=True, help="the API's public origin, e.g. https://….run.app (no /api/files)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup", default="backups/file-urls")
    args = ap.parse_args()
    origin = args.origin.rstrip("/")
    assert origin.startswith("https://") and "localhost" not in origin, "the origin must be the public https API"
    s = get_settings()
    key = s.supabase_service_role_key.get_secret_value()
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    base = f"{s.supabase_url.rstrip('/')}/rest/v1"
    with httpx.Client(timeout=60) as http:
        tables = sorted(http.get(f"{base}/", headers=h).json().get("definitions", {}))
        for table in tables:
            rows, offset = [], 0
            while True:
                r = http.get(f"{base}/{table}", params={"select": "*", "limit": 1000, "offset": offset}, headers=h)
                if r.status_code != 200:
                    break
                rows += r.json()
                if len(r.json()) < 1000:
                    break
                offset += 1000
            changed = [(row, {c: rewrite(v, origin) for c, v in row.items() if rewrite(v, origin) != v}) for row in rows]
            changed = [(row, patch) for row, patch in changed if patch]
            if not changed:
                continue
            urls = sum(len(LOCAL.findall(json.dumps(row[c]))) for row, patch in changed for c in patch)
            print(f"{table}: {len(changed)} rows, {urls} URLs, columns {sorted({c for _, p in changed for c in p})}")
            if not args.apply:
                continue
            if any("id" not in row for row, _ in changed):
                print(f"  skipped: {table} has no id column")
                continue
            out = Path(args.backup)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{table}.json").write_text(json.dumps([row for row, _ in changed], ensure_ascii=False))
            done = 0
            for row, patch in changed:
                res = http.patch(f"{base}/{table}", params={"id": f"eq.{row['id']}"}, headers={**h, "Prefer": "return=minimal"}, json=patch)
                if res.status_code == 400 and done == 0:
                    # A table that refuses updates by design (design_versions: an append-only history) is left as it is.
                    print(f"  skipped: {table} refuses updates ({res.json().get('message')})")
                    break
                res.raise_for_status()
                done += 1
            else:
                print(f"  updated {done} rows; backup in {out / f'{table}.json'}")
    if not args.apply:
        print("Dry run: nothing written. Add --apply to update.")


if __name__ == "__main__":
    main()
