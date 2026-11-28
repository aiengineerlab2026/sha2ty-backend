"""Complex components of a rebuilt model (services/hybrid.py), generated one by one by Meshy and set into the model.

  start(...)         at generation: one Image to 3D task per DISTINCT complex component (repeats share one), from its
                     cutout of the original photo; a cutout generated before (same pixels) is reused from model_cache
  poll(...)          GET /api/vendor/tasks/{id} while stage == "components": Meshy's tasks, then assembly
  assemble(...)      each result checked (component_mesh.check), fitted into its box and appended as its own node,
                     attached to its host part (a door's ornament moves with the door); a failure is reported per
                     component, never replaced by a block
  regenerate: one component again (api/vendor.py POST /api/vendor/hybrid/components/regenerate), the rest kept

Costs: one Meshy Image to 3D task (~30 credits) per distinct complex component not already cached; none otherwise.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.concurrency import run_in_threadpool

from app.schemas.vendor import GenerationOptions
from app.services import component_mesh, vision
from app.services.cdn import Cdn, CdnError
from app.services.meshy_service import MeshyError, MeshyService

logger = logging.getLogger(__name__)

CACHE_VERSION = "meshy-component-1"
TIMEOUT_SECONDS = 1200.0  # a component task still running after this is reported failed


def _key(image: bytes) -> str:
    return hashlib.sha256(image + CACHE_VERSION.encode()).hexdigest()


async def start(request: Request, meshy: MeshyService, cdn: Cdn, settings, user_id: str, object_id: str, doc: dict[str, Any],
                images: dict[str, bytes], force: bool = False) -> dict[str, dict[str, Any]]:
    """{group: {"task": id} | {"glb_url": cached file} | {"error": why}} for the model's pending components. The
    cutouts are uploaded beside the model (so one component can be regenerated later) and recorded on each part."""
    jobs: dict[str, dict[str, Any]] = {}
    for part in doc.get("pendingComponents") or []:
        group = part["group"]
        image = images.get(group)
        if group in jobs or image is None:
            continue
        key = _key(image)
        try:
            part_url = await run_in_threadpool(cdn.upload, f"{user_id}/{object_id}/components/{key[:16]}.png", image, "image/png")
        except CdnError:
            part_url = None
        for p in doc["pendingComponents"]:
            if p["group"] == group:
                p["imageUrl"], p["imageKey"] = part_url, key
        hit = None if force else await vision.cache_get(settings, request.app.state.auth_http, key, table="model_cache")
        if hit and hit.get("glb_url"):
            jobs[group] = {"glb_url": hit["glb_url"], "cached": True, "key": key}
            continue
        if not settings.meshy_configured:
            jobs[group] = {"error": "Meshy is not configured: complex components cannot be generated.", "key": key}
            continue
        try:
            task = await meshy.create_image_to_3d_task(GenerationOptions(should_texture=True, enable_pbr=True), image=image, mime_type="image/png",
                                                       texture_source="image")  # its own pixels as the texture guide
            jobs[group] = {"task": task, "key": key, "started": time.monotonic()}
            logger.info("Hybrid component %r: Meshy image-to-3d task %s", part["name"], task)
        except MeshyError as exc:
            jobs[group] = {"error": f"Meshy refused it: {exc.message}", "key": key}
    return jobs


async def poll(request: Request, meshy: MeshyService, cdn: Cdn, settings, pending: dict[str, Any]) -> bool:
    """One look at the running component tasks. True when every one has finished (succeeded or failed)."""
    done = True
    for group, job in pending["jobs"].items():
        if "task" not in job or job.get("glb") or job.get("error"):
            continue
        try:
            task, _ = await meshy.get_image_to_3d_task(job["task"])
        except MeshyError as exc:
            if exc.status_code in (429, 502, 503, 504):
                done = False
                continue
            job["error"] = exc.message
            continue
        state = str(task.get("status") or "")
        if state in ("PENDING", "IN_PROGRESS", ""):
            if time.monotonic() - job.get("started", time.monotonic()) > TIMEOUT_SECONDS:
                job["error"] = "Meshy took too long."
            else:
                job["progress"] = int(task.get("progress") or 0)
                done = False
            continue
        url = (task.get("model_urls") or {}).get("glb")
        if state != "SUCCEEDED" or not url:
            err = task.get("task_error")
            job["error"] = (err.get("message") if isinstance(err, dict) else err) or f"Meshy ended {state}."
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part.glb"
            await meshy.download_file(url, path)
            job["glb"] = path.read_bytes()
        try:
            stored = await run_in_threadpool(cdn.upload, f"{pending['user_id']}/components/{job['key'][:24]}.glb", job["glb"], "model/gltf-binary")
            await vision.cache_put(settings, request.app.state.auth_http, job["key"], "hybrid_component", "meshy", {"glb_url": stored}, table="model_cache")
            job["glb_url"] = stored
        except CdnError as exc:
            logger.warning("Generated component not stored: %s", exc.message)
    return done


async def fetch(cdn: Cdn, url: str, settings) -> bytes | None:
    """One of our stored files (read from the bucket by its key); None for anything else."""
    from app.services.cdn import file_key

    key = file_key(url, settings)
    if not key:
        return None
    try:
        obj = await run_in_threadpool(cdn.get, key)
        return obj["Body"].read()
    except CdnError:
        return None


async def assemble(cdn: Cdn, base: bytes, doc: dict[str, Any], jobs: dict[str, dict[str, Any]], settings) -> tuple[bytes, dict[str, Any]]:
    """Every finished component set into the model; failures listed, never replaced by a placeholder."""
    comps = doc["components"]
    anims = {a["component"]: a for a in (doc.get("animation") or {}).get("animations") or []}
    failures, made = [], 0
    for part in doc.get("pendingComponents") or []:
        job = jobs.get(part["group"]) or {}
        glb = job.get("glb") or (await fetch(cdn, job["glb_url"], settings) if job.get("glb_url") else None)
        issues = [job["error"]] if job.get("error") else [] if glb else ["The generated file is missing."]
        if glb and not issues:
            issues = await run_in_threadpool(component_mesh.check, glb, part["box"])
        if issues:
            part.update(status="failed", reason=" ".join(issues), regenerable=True)
            failures.append(f"{part['name']} could not be generated ({part['reason']}); regenerate it from the editor.")
            continue
        fitted = await run_in_threadpool(component_mesh.fit, glb, part["box"])
        extras = {"partId": part["id"], "generation": "meshy", "specId": part["specId"], **({"attachedTo": part["attachedTo"]} if part.get("attachedTo") else {}),
                  **({"taskId": job["task"]} if job.get("task") else {})}
        base, node = await run_in_threadpool(component_mesh.append, base, fitted, part["name"], extras)
        comps[:] = [c for c in comps if c["id"] != part["id"]]  # a regenerated part replaces its earlier record
        comps.append({"id": part["id"], "name": part["name"], "nameAr": part.get("nameAr"), "mesh": node, "node": node, "editable": True,
                      "box": part["box"], "analysisId": part["specId"], "generation": "meshy", "attachedTo": part.get("attachedTo"),
                      **({"taskId": job["task"]} if job.get("task") else {}), **({"cached": True} if job.get("cached") else {}),
                      "material": {"class": "generated", "fromPhoto": True, "confirmed": False}})
        host = anims.get(part.get("attachedTo") or "")
        if host is not None and part["id"] not in host["follow"]:
            host["follow"].append(part["id"])  # an ornament on a door opens with the door
        part.update(status="done", reason=None)
        made += 1
    doc["componentFailures"] = [p for p in doc.get("pendingComponents") or [] if p.get("status") == "failed"]
    if failures:
        doc["limits"] = " ".join(x for x in (doc.get("limits"), *failures) if x)
    doc.setdefault("cost", {})["meshGeneratorCalls"] = sum(1 for j in jobs.values() if j.get("task"))
    logger.info("Hybrid components: %d set, %d failed", made, len(failures))
    return base, doc
