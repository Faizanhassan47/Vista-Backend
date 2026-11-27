"""One-at-a-time TotalSegmentator jobs. Results stay in VISTA storage."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from services.totalsegmentator_service import (
    MODEL_NAME,
    TotalSegmentatorError,
    color_for,
    command,
    empty_measurement,
    measure_nifti,
    normalize_rois,
    settings,
    task_for_modality,
    worker_env,
)

_LOCK = threading.Lock()
_JOBS = {}
_ACTIVE = None
_STALE_RECOVERED = False


def _now():
    return datetime.now(timezone.utc).isoformat()


def _work():
    path = Path(settings()["work_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _public(job):
    hidden = {"input_path", "output_dir", "log_path"}
    return {key: value for key, value in job.items() if key not in hidden}


def _mark_failed(job, message):
    job["status"] = "failed"
    job["stage"] = "Failed"
    job["error"] = message
    job["finished_at"] = _now()
    if job.get("runtime_seconds") is None and job.get("started_at"):
        try:
            started = datetime.fromisoformat(job["started_at"])
            job["runtime_seconds"] = round((datetime.now(timezone.utc) - started).total_seconds(), 2)
        except ValueError:
            job["runtime_seconds"] = None


def recover_stale_jobs():
    """Mark orphaned queued/running jobs failed after API restart (daemon threads die)."""
    global _ACTIVE, _STALE_RECOVERED
    root = _work()
    if not root.is_dir():
        _STALE_RECOVERED = True
        return 0
    recovered = 0
    with _LOCK:
        active = _ACTIVE
        for path in root.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            job_id = job.get("job_id") or path.parent.name
            if job.get("status") not in {"queued", "running"}:
                continue
            if active and job_id == active:
                continue
            _mark_failed(
                job,
                "Segmentation was interrupted when the API restarted. Run segmentation again.",
            )
            path.write_text(json.dumps(job, indent=2), encoding="utf-8")
            _JOBS[job_id] = job
            recovered += 1
        if active and active not in _JOBS:
            # Active id from a previous process cannot survive restart.
            _ACTIVE = None
        _STALE_RECOVERED = True
    return recovered


def _ensure_recovered():
    if not _STALE_RECOVERED:
        recover_stale_jobs()


def get_job(job_id):
    _ensure_recovered()
    with _LOCK:
        job = _JOBS.get(job_id)
    if job:
        return _public(job)
    path = _work() / job_id / "job.json"
    if not path.is_file():
        return None
    job = json.loads(path.read_text(encoding="utf-8"))
    with _LOCK:
        _JOBS[job_id] = job
    return _public(job)


def latest_for_slide(slide_id):
    _ensure_recovered()
    root = _work()
    found = []
    if not root.is_dir():
        return None
    for path in root.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(job.get("slide_id") or 0) == int(slide_id):
            found.append(job)
    if not found:
        return None
    found.sort(key=lambda item: item.get("created_at") or "")
    return _public(found[-1])


def start_job(slide, source_nifti, modality, rois, user):
    global _ACTIVE
    _ensure_recovered()
    task = task_for_modality(modality)
    selected = normalize_rois(rois)
    config = settings()
    if not config["python"] or not Path(config["python"]).is_file():
        raise TotalSegmentatorError("TotalSegmentator Python is not configured", "not_configured")
    with _LOCK:
        if _ACTIVE is not None:
            raise TotalSegmentatorError("A TotalSegmentator job is already running", "busy")
        job_id = uuid.uuid4().hex
        _ACTIVE = job_id
        job = {
            "job_id": job_id,
            "status": "queued",
            "slide_id": int(slide["id"]),
            "case_id": int(slide["case_id"]),
            "model_name": MODEL_NAME,
            "model_version": config["model_version"],
            "modality": "CT" if task == "total" else "MR",
            "task": task,
            "rois": selected,
            "generated_rois": [],
            "fast": config["fast"],
            "stage": "Queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "runtime_seconds": None,
            "error": None,
            "measurements": [],
            "requested_by": user,
            "input_path": str(source_nifti),
            "output_dir": str(_work() / job_id / "masks"),
            "research_use_only": True,
            "disclaimer": "Predicted anatomical structures and measurements. Not a diagnosis.",
        }
        _JOBS[job_id] = job
    _persist(job)
    thread = threading.Thread(target=_run, args=(job_id,), name=f"totalseg-{job_id[:8]}", daemon=True)
    thread.start()
    return _public(job)


def _persist(job):
    folder = _work() / job["job_id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(_public(job) | {
        "input_path": job.get("input_path"),
        "output_dir": job.get("output_dir"),
    }, indent=2), encoding="utf-8")


def _run(job_id):
    global _ACTIVE
    with _LOCK:
        job = _JOBS[job_id]
        job["status"] = "running"
        job["stage"] = "Preparing volume"
        job["started_at"] = _now()
    _persist(job)
    started = time.perf_counter()
    config = settings()
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / "inference.log"
    try:
        with _LOCK:
            job["stage"] = "Running inference"
        _persist(job)
        argv = command(config["python"], job["input_path"], output, job["task"], job["rois"], config["fast"])
        env = worker_env()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                argv,
                cwd=str(output.parent),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=config["timeout_seconds"],
                check=False,
            )
        if completed.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-500:]
            raise TotalSegmentatorError(f"TotalSegmentator failed ({completed.returncode}). {tail}".strip(), "inference_failed")
        with _LOCK:
            job["stage"] = "Saving masks"
        _persist(job)
        report_hu = job["task"] == "total"
        measurements = []
        generated = []
        for roi in job["rois"]:
            mask = output / f"{roi}.nii.gz"
            if not mask.is_file():
                measurements.append(empty_measurement(roi, report_hu))
                continue
            measurements.append(measure_nifti(job["input_path"], mask, roi, report_hu))
            generated.append(roi)
        with _LOCK:
            job["status"] = "completed"
            job["stage"] = "Completed"
            job["measurements"] = measurements
            job["generated_rois"] = generated
    except subprocess.TimeoutExpired:
        with _LOCK:
            job["status"] = "failed"
            job["stage"] = "Failed"
            job["error"] = "TotalSegmentator exceeded the configured time limit"
    except Exception as exc:
        with _LOCK:
            job["status"] = "failed"
            job["stage"] = "Failed"
            job["error"] = str(exc)
    finally:
        with _LOCK:
            job["finished_at"] = _now()
            job["runtime_seconds"] = round(time.perf_counter() - started, 2)
            _ACTIVE = None if _ACTIVE == job_id else _ACTIVE
        _persist(job)


_LABEL_CACHE = {}


def _job_file(job_id):
    path = _work() / job_id / "job.json"
    if not path.is_file():
        raise TotalSegmentatorError("No completed segmentation is available", "missing_output")
    job = json.loads(path.read_text(encoding="utf-8"))
    if job.get("status") != "completed":
        raise TotalSegmentatorError("No completed segmentation is available", "missing_output")
    return job


def aligned_labels(job_id, rois):
    """Organ labels on the same voxel grid as the source study.

    Coronal, sagittal, and 3D use this grid, so a slice index matches the CT
    slice the viewer already shows.
    """
    import nibabel as nib

    job = _job_file(job_id)
    requested = normalize_rois(rois or job.get("generated_rois") or job["rois"], job.get("task") or "total")
    output = Path(job["output_dir"])
    # Only ROIs that actually have mask files — UI may still list structures
    # that were unchecked/deselected after a previous run.
    selected = tuple(roi for roi in requested if (output / f"{roi}.nii.gz").is_file())
    if not selected:
        available = tuple(roi for roi in (job.get("generated_rois") or job.get("rois") or []) if (output / f"{roi}.nii.gz").is_file())
        selected = available
    stamp = tuple((output / f"{roi}.nii.gz").stat().st_mtime_ns for roi in selected)
    key = (job_id, selected, stamp)
    cached = _LABEL_CACHE.get(key)
    if cached is not None:
        return cached
    source = nib.load(str(job["input_path"]))
    shape = tuple(int(size) for size in source.shape[:3])
    labels = np.zeros(shape, dtype=np.uint8)
    for label_id, roi in enumerate(selected, start=1):
        path = output / f"{roi}.nii.gz"
        mask = nib.load(str(path))
        data = np.asanyarray(mask.dataobj)
        if tuple(int(size) for size in data.shape[:3]) != shape:
            from nibabel.processing import resample_from_to
            data = np.asanyarray(resample_from_to(mask, source, order=0).dataobj)
        labels[np.asanyarray(data) > 0] = np.uint8(label_id)
    _LABEL_CACHE.clear()
    _LABEL_CACHE[key] = (labels, source, selected)
    return labels, source, selected


def _slice_labels(labels, source, plane, index):
    from services.radiology_frames import _axis_for, _display_plane, _nifti_geometry
    import cv2

    codes, zooms = _nifti_geometry(source)
    plane = plane if plane in {"axial", "coronal", "sagittal"} else "axial"
    through = {"axial": "SI", "coronal": "AP", "sagittal": "LR"}[plane]
    axis = _axis_for(codes, through)
    count = int(labels.shape[axis])
    index = min(max(int(index), 0), count - 1)
    raw = np.asanyarray(labels[tuple(slice(None) if part != axis else index for part in range(3))])
    row_axis, col_axis = [part for part in range(3) if part != axis]
    displayed, row_spacing, col_spacing = _display_plane(
        raw, codes[row_axis], codes[col_axis], zooms[row_axis], zooms[col_axis], plane,
    )
    target = min(row_spacing, col_spacing)
    height = max(1, int(round(displayed.shape[0] * row_spacing / target)))
    width = max(1, int(round(displayed.shape[1] * col_spacing / target)))
    if displayed.shape != (height, width):
        displayed = cv2.resize(displayed, (width, height), interpolation=cv2.INTER_NEAREST)
    return displayed


def overlay_png(job_id, plane, index, rois):
    from io import BytesIO
    from PIL import Image

    labels, source, selected = aligned_labels(job_id, rois)
    displayed = _slice_labels(labels, source, plane, index)
    canvas = np.zeros(displayed.shape + (4,), dtype=np.uint8)
    for label_id, roi in enumerate(selected, start=1):
        color = color_for(roi)
        on = displayed == label_id
        canvas[on, 0] = color[0]
        canvas[on, 1] = color[1]
        canvas[on, 2] = color[2]
        canvas[on, 3] = 160
    buffer = BytesIO()
    Image.fromarray(canvas, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def label_volume_bytes(job_id, rois, quality="balanced"):
    """Downsampled label volume in the same axis order as the 3D CT payload."""
    from services.radiology_frames import _downsample

    labels, source, selected = aligned_labels(job_id, rois)
    zooms = tuple(float(value) for value in source.header.get_zooms()[:3])
    chosen = quality if quality in {"preview", "balanced", "high"} else "balanced"
    reduced, spacing, _preview = _downsample(labels, zooms, chosen, nearest=True)
    colors = [color_for(roi) for roi in selected]
    return reduced.astype(np.uint8).tobytes(order="F"), {
        "dimensions": list(reduced.shape),
        "spacing": list(spacing),
        "rois": list(selected),
        "colors": colors,
    }
