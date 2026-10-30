# Shatably API

FastAPI proxy for Meshy and GPT Astra. Generated GLBs are uploaded to the Supabase `cads` bucket. Keys never go to the browser.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` (git-ignored): `MESHY_API_KEY`, `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, S3 access keys, `DATABASE_URL`.

Apply tables and storage policies:

```bash
python scripts/apply_schema.py
```

Or paste `supabase/schema.sql` into the Supabase SQL editor.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

- `GET /api/health`
- `POST /api/vendor/generate` — `Authorization: Bearer <supabase access token>`, multipart `image`, `provider=meshy|astra`, optional `model_id`
- `GET /api/vendor/tasks/{task_id}` — Meshy poll; on success uploads the GLB to CDN

## Floor plans (`/plan`)

Traditional CV only (OpenCV, Shapely, Trimesh). No LLM.

```bash
pip install -r requirements.txt
pytest tests/test_plan.py
```

OCR: on macOS the built-in Vision framework is used (`pyobjc-framework-Vision`, in requirements). Elsewhere install [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), RapidOCR, or Tesseract. If none are present, the import reports it and asks you to enter the scale. **Restart uvicorn after installing an OCR engine** — `--reload` does not watch site-packages. Stage images for every run are written to `storage/plan-debug/`.

- `POST /api/plans/from-dimensions` — rectangular room in meters
- `POST /api/plans/from-image` — OpenCV walls + OCR scale when available
- `GET /api/plans/models` — ready catalog GLBs grouped by vendor
- `POST/GET/PUT /api/plans` — save/load projects (furniture is referenced, not copied)
- `POST /api/plans/{id}/export?format=glb` — room mesh GLB (1 unit = 1 m)

GLB/GLTF load in the browser. STL/OBJ/PLY convert through Trimesh. STEP is not converted without CadQuery/OpenCascade; mesh furniture is not claimed as parametric CAD.
