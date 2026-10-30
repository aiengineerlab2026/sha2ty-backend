"""Apply backend/supabase/schema.sql using DATABASE_URL from backend/.env."""

from pathlib import Path
import sys

import psycopg

ROOT = Path(__file__).resolve().parents[1]
env_file = ROOT / ".env"
schema = ROOT / "supabase" / "schema.sql"

for line in env_file.read_text().splitlines():
    if line.startswith("DATABASE_URL="):
        url = line.split("=", 1)[1].strip()
        break
else:
    sys.exit("DATABASE_URL is missing from backend/.env")

if "[YOUR-PASSWORD]" in url:
    sys.exit("Replace [YOUR-PASSWORD] in DATABASE_URL, then run this script again.")

sql = schema.read_text()
with psycopg.connect(url, autocommit=True, connect_timeout=20) as conn:
    with conn.cursor() as cur:
        cur.execute(sql)

print("Applied", schema)
