"""Controlled Meshy experiment on one product photo, every result rendered by the same camera (fidelity.front_render).

  A  original photo → Image to 3D, Meshy's default texturing, no cleaning, no split
  B  original photo → Image to 3D, texture_image_url = the photo, then Auto Split (by_parts, the reading's parts)
  C  cleaned photo  → Image to 3D (texture from it), only if the cleaned photo passes fidelity.compare_images
  D  each raw GLB   → Sha2ty's processing (geometry repair, size fit) → raw vs processed

Offline (no credits): D and the renders for GLBs you already have:
    python scripts/meshy_experiment.py --photo wardrobe.jpg --glb storage/models/<task>.glb [--spec storage/furniture-spec/<key>.json]

Paid (Meshy credits for A, B, C and one split; one vision call for the spec and one image call for C):
    python scripts/meshy_experiment.py --photo wardrobe.jpg --paid [--only A,B]

Results: storage/meshy-experiment/<time>/ (each GLB, its front render PNG, report.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.schemas.vendor import GenerationOptions  # noqa: E402
from app.services import fidelity, furniture_spec, geometry_quality, mesh_analysis, vision  # noqa: E402
from app.services.generation_prompt import meshy_prompt  # noqa: E402
from app.services.glb_size import fit_glb  # noqa: E402
from app.services.glb_parts import inspect_glb  # noqa: E402
from app.services.meshy_service import MeshyService  # noqa: E402


def state(settings) -> SimpleNamespace:
    """The HTTP clients the app keeps in app.state (main.py), for the vision calls."""
    return SimpleNamespace(
        auth_http=httpx.AsyncClient(timeout=60),
        openai_http=httpx.AsyncClient(timeout=300, headers={"Authorization": f"Bearer {settings.openai_api_key.get_secret_value()}"}),
    )


def processed(raw: bytes) -> bytes:
    doc = inspect_glb(raw, "furniture")
    out, _, _ = geometry_quality.process(raw, doc, None, False)
    return fit_glb(out)[0]


def evaluate(name: str, raw: bytes, photo: bytes | None, spec: dict | None, out: Path, cleaned: bytes | None = None) -> dict:
    details = (spec or {}).get("details") or {}
    (out / f"{name}.raw.glb").write_bytes(raw)
    (out / f"{name}.png").write_bytes(fidelity.render_png(raw))
    final = processed(raw)
    (out / f"{name}.final.png").write_bytes(fidelity.render_png(final))
    return {
        "provider": fidelity.provider(raw, photo, details.get("bbox"), cleaned, details.get("proportions"), True),
        "rawVsFinal": fidelity.compare_glbs(raw, final),
        "meshes": fidelity.mesh_count(raw),
    }


async def poll(get, task_id: str, what: str) -> dict:
    while True:
        task = await get(task_id)
        task = task[0] if isinstance(task, tuple) else task
        print(f"  {what} {task_id}: {task.get('status')} {task.get('progress')}%")
        if task.get("status") in ("SUCCEEDED", "FAILED", "CANCELED", "EXPIRED"):
            return task
        await asyncio.sleep(10)


async def paid(photo: bytes, mime: str, spec: dict | None, only: set[str], out: Path, report: dict) -> None:
    settings = get_settings()
    api, downloads = MeshyService.create_clients(settings)
    meshy = MeshyService(settings, api, downloads)
    options = GenerationOptions(should_texture=True, enable_pbr=True)

    async def generate(name: str, image: bytes, image_mime: str, texture: str) -> tuple[str, bytes] | None:
        task_id = await meshy.create_image_to_3d_task(options, image=image, mime_type=image_mime, target_polycount=1, texture_source=texture)
        task = await poll(meshy.get_image_to_3d_task, task_id, name)
        if task.get("status") != "SUCCEEDED":
            report[name] = {"status": task.get("status"), "error": task.get("task_error")}
            return None
        path = out / f"{name}.download.glb"
        await meshy.download_file(task["model_urls"]["glb"], path)
        return task_id, path.read_bytes()

    if "A" in only and (a := await generate("A", photo, mime, "input")):
        report["A"] = evaluate("A", a[1], photo, spec, out)
    if "B" in only and (b := await generate("B", photo, mime, "image")):
        report["B"] = evaluate("B", b[1], photo, spec, out)
        analysis = mesh_analysis.MeshAnalysis.model_validate(spec["analysis"]) if spec and spec.get("analysis") else None
        parts = mesh_analysis.split_prompt(analysis) if analysis else None
        if parts:
            split_id = await meshy.create_split_task(b[0], parts, True)
            task = await poll(meshy.get_split_task, split_id, "B split")
            split = None
            if task.get("status") == "SUCCEEDED":
                await meshy.download_file(task["model_urls"]["glb"], out / "B.split.glb")
                split = (out / "B.split.glb").read_bytes()
                (out / "B.split.png").write_bytes(fidelity.render_png(split))
            err = task.get("task_error")
            report["B"]["split"] = fidelity.split_check(b[1], split, {"status": task.get("status"), "taskId": split_id, "partCount": task.get("part_count"),
                                                                     "promptIgnored": task.get("prompt_ignored"), "error": (err or {}).get("message") if isinstance(err, dict) else err, "prompt": parts})
    if "C" in only:
        if not spec:
            report["C"] = {"skipped": "no spec"}
        else:
            st = state(settings)
            model = await vision.current_model(settings, st.auth_http)
            client = vision.client_for(st, model)
            cleaned = await vision.clean_image(meshy_prompt(spec["details"].get("category"), None, None), photo, mime, settings, client, model)
            (out / "C.cleaned.png").write_bytes(cleaned)
            check = fidelity.compare_images(photo, cleaned, spec["details"].get("bbox"))
            report["C"] = {"cleaning": check}
            if check.get("ok") is not False and (c := await generate("C", cleaned, "image/png", "image")):
                report["C"].update(evaluate("C", c[1], photo, spec, out, cleaned))
            elif check.get("ok") is False:
                report["C"]["skipped"] = "the cleaned image failed validation: it is not meshed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", type=Path, help="the product photo (needed for A, B, C and the photo comparison)")
    ap.add_argument("--glb", type=Path, action="append", default=[], help="existing raw GLB(s): offline D and renders")
    ap.add_argument("--spec", type=Path, help="a cached furniture spec JSON (else, with --paid, the photo is read once)")
    ap.add_argument("--paid", action="store_true", help="call Meshy (and the vision model for C): spends credits")
    ap.add_argument("--only", default="A,B,C")
    args = ap.parse_args()
    if args.paid and not args.photo:
        sys.exit("--paid needs --photo")
    photo = args.photo.read_bytes() if args.photo else None
    mime = "image/jpeg" if photo and photo[:2] == b"\xff\xd8" else "image/png"
    spec = json.loads(args.spec.read_text()) if args.spec else None
    out = ROOT / "storage/meshy-experiment" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"photo": str(args.photo), "spec": str(args.spec) if args.spec else None}
    for i, glb in enumerate(args.glb):
        report[f"D{i}:{glb.name}"] = evaluate(f"D{i}", glb.read_bytes(), photo, spec, out)
    if args.paid:
        if spec is None:
            settings = get_settings()

            async def read() -> dict | None:
                st = state(settings)
                model = await vision.current_model(settings, st.auth_http)
                s = await furniture_spec.read(photo, mime, "auto", "", settings, vision.client_for(st, model), model, lang="en")
                return s.model_dump(mode="json") if s else None

            spec = asyncio.run(read())
        asyncio.run(paid(photo, mime, spec, set(args.only.split(",")), out, report))
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
