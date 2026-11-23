"""Authenticated TotalSegmentator jobs for verified CT and MRI volumes."""

import json
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core import AI_READ_ROLES, AI_RUN_ROLES
from services.radiology_frames import nifti_calibration
from services.totalsegmentator_jobs import get_job, label_volume_bytes, latest_for_slide, overlay_png, start_job
from services.totalsegmentator_service import TotalSegmentatorError, apply_modality_verification, class_ids, settings, structures_for_modality, task_for_modality

logger = logging.getLogger("pathology.totalsegmentator")


class VerifyRequest(BaseModel):
    slide_id: int
    modality: str


class JobRequest(BaseModel):
    slide_id: int
    rois: list[str] = Field(default_factory=list)
    plane: str = "axial"
    index: int = Field(0, ge=0)


def create_totalsegmentator_router(
    connect_func,
    require_roles_func,
    get_current_user_func,
    get_slide_func,
    get_case_func,
    stored_path_func,
    require_assigned_case_func,
    audit_func=None,
):
    router = APIRouter(tags=["TotalSegmentator"])

    def record(user, action, slide_id, details):
        if not audit_func or not connect_func:
            return
        try:
            from contextlib import closing
            with closing(connect_func()) as db:
                audit_func(db, user.get("username", ""), action, "slide", str(slide_id or ""), details)
        except Exception:
            logger.exception("TotalSegmentator audit write failed")

    def authorized_slide(slide_id, user):
        slide = get_slide_func(slide_id)
        case = get_case_func(slide["case_id"])
        require_assigned_case_func(case, user)
        return slide

    @router.get("/api/ai/totalsegmentator/status", dependencies=[Depends(require_roles_func(AI_READ_ROLES))])
    def totalseg_status():
        config = settings()
        return {
            "model_name": "TotalSegmentator",
            "model_version": config["model_version"],
            "configured": bool(config["python"] and Path(config["python"]).is_file()),
            "fast": config["fast"],
            "device": "cpu",
            "max_concurrent_jobs": 1,
            "tasks": {"CT": "total", "MR": "total_mr"},
            "structure_counts": {"CT": len(class_ids("total")), "MR": len(class_ids("total_mr"))},
            "research_use_only": True,
        }

    @router.get("/api/ai/totalsegmentator/structures", dependencies=[Depends(require_roles_func(AI_READ_ROLES))])
    def totalseg_structures(modality: str = "CT"):
        try:
            return structures_for_modality(modality)
        except TotalSegmentatorError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.post("/api/ai/totalsegmentator/jobs", dependencies=[Depends(require_roles_func(AI_RUN_ROLES))])
    async def totalseg_start(body: JobRequest, current_user: dict = Depends(get_current_user_func)):
        slide = authorized_slide(body.slide_id, current_user)
        if slide["file_type"] not in {"dicom-stack", "nifti"}:
            raise HTTPException(400, "TotalSegmentator accepts a verified CT or MRI DICOM series or NIfTI volume")
        try:
            source, modality = await run_in_threadpool(_prepare_input, slide, stored_path_func, connect_func)
            task_for_modality(modality)
            job = start_job(slide, source, modality, body.rois, current_user.get("username", ""))
        except TotalSegmentatorError as exc:
            status = 409 if exc.category == "busy" else 503 if exc.category == "not_configured" else 400
            raise HTTPException(status, str(exc)) from exc
        record(current_user, "totalsegmentator.start", body.slide_id, f"job={job['job_id']}; task={job['task']}")
        return job

    @router.get("/api/ai/totalsegmentator/jobs/{job_id}", dependencies=[Depends(require_roles_func(AI_READ_ROLES))])
    def totalseg_job(job_id: str, current_user: dict = Depends(get_current_user_func)):
        job = get_job(job_id)
        if not job:
            raise HTTPException(404, "Segmentation job was not found")
        authorized_slide(job["slide_id"], current_user)
        return job

    @router.get("/api/ai/totalsegmentator/slides/{slide_id}", dependencies=[Depends(require_roles_func(AI_READ_ROLES))])
    def totalseg_latest(slide_id: int, current_user: dict = Depends(get_current_user_func)):
        authorized_slide(slide_id, current_user)
        return latest_for_slide(slide_id) or {"status": "none"}

    @router.get("/api/ai/totalsegmentator/jobs/{job_id}/overlay", dependencies=[Depends(require_roles_func(AI_READ_ROLES))])
    def totalseg_overlay(job_id: str, plane: str = "axial", index: int = 0, rois: str = "", current_user: dict = Depends(get_current_user_func)):
        job = get_job(job_id)
        if not job:
            raise HTTPException(404, "Segmentation job was not found")
        authorized_slide(job["slide_id"], current_user)
        try:
            payload = overlay_png(job_id, plane, index, [item for item in rois.split(",") if item])
        except TotalSegmentatorError as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(content=payload, media_type="image/png")

    @router.get("/api/ai/totalsegmentator/jobs/{job_id}/volume", dependencies=[Depends(require_roles_func(AI_READ_ROLES))])
    def totalseg_volume(job_id: str, quality: str = "balanced", rois: str = "", current_user: dict = Depends(get_current_user_func)):
        job = get_job(job_id)
        if not job:
            raise HTTPException(404, "Segmentation job was not found")
        authorized_slide(job["slide_id"], current_user)
        try:
            payload, meta = label_volume_bytes(job_id, [item for item in rois.split(",") if item], quality)
        except TotalSegmentatorError as exc:
            raise HTTPException(404, str(exc)) from exc
        dims = meta["dimensions"]
        spacing = meta["spacing"]
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={
                "X-Dim-I": str(dims[0]),
                "X-Dim-J": str(dims[1]),
                "X-Dim-K": str(dims[2]),
                "X-Spacing-I": str(spacing[0]),
                "X-Spacing-J": str(spacing[1]),
                "X-Spacing-K": str(spacing[2]),
                "X-Rois": ",".join(meta["rois"]),
                "X-Colors": "|".join(",".join(str(channel) for channel in color) for color in meta["colors"]),
            },
        )

    @router.post("/api/ai/totalsegmentator/verify-modality", dependencies=[Depends(require_roles_func(AI_RUN_ROLES))])
    def totalseg_verify(body: VerifyRequest, current_user: dict = Depends(get_current_user_func)):
        slide = authorized_slide(body.slide_id, current_user)
        if slide["file_type"] != "nifti":
            raise HTTPException(400, "Modality verification is only for a NIfTI study")
        if not connect_func:
            raise HTTPException(503, "Modality verification is not available")
        from datetime import datetime, timezone
        from contextlib import closing
        verified_at = datetime.now(timezone.utc).isoformat()
        try:
            with closing(connect_func()) as db:
                version = db.execute(
                    "SELECT TOP 1 id, metadata_json FROM image_versions WHERE slide_id = ? ORDER BY version DESC",
                    (body.slide_id,),
                ).fetchone()
                if not version:
                    raise HTTPException(404, "This study has no stored metadata to verify")
                metadata = json.loads(version.get("metadata_json") or "{}")
                updated = apply_modality_verification(metadata, body.modality, current_user.get("username", ""), verified_at)
                db.execute("UPDATE image_versions SET metadata_json = ? WHERE id = ?", (json.dumps(updated), version["id"]))
                db.commit()
        except TotalSegmentatorError as exc:
            raise HTTPException(400, str(exc)) from exc
        calibration = nifti_calibration(updated)
        record(current_user, "totalsegmentator.verify_modality", body.slide_id, f"modality={calibration.get('modality')}; source=authorized_user")
        return {"modality": calibration.get("modality"), "calibration": calibration}

    return router


def _metadata(connect_func, slide_id):
    if not connect_func:
        return {}
    from contextlib import closing
    with closing(connect_func()) as db:
        version = db.execute(
            "SELECT metadata_json FROM image_versions WHERE slide_id = ? ORDER BY version DESC",
            (slide_id,),
        ).fetchone()
    if not version or not version.get("metadata_json"):
        return {}
    return json.loads(version["metadata_json"])


def _prepare_input(slide, stored_path_func, connect_func):
    path = Path(stored_path_func(slide["filename"]))
    metadata = _metadata(connect_func, slide["id"])
    if slide["file_type"] == "nifti":
        source = path if path.is_file() else next(path.parent.glob("*.nii*"), None)
        if source is None or not Path(source).is_file():
            raise TotalSegmentatorError("The stored NIfTI volume is missing", "missing_image")
        calibration = nifti_calibration(metadata)
        modality = str(calibration.get("modality") or "").upper()
        if modality not in {"CT", "MR", "MRI"}:
            raise TotalSegmentatorError(
                "This NIfTI volume has no verified CT or MRI modality.",
                "unsupported_modality",
            )
        return Path(source), modality
    instances = [Path(item["path"]) for item in metadata.get("instances") or [] if item.get("path")]
    files = instances or sorted(item for item in path.parent.iterdir() if item.is_file()) if path.parent.is_dir() else [path]
    destination = Path(settings()["work_dir"]) / "inputs" / f"slide-{slide['id']}.nii.gz"
    modality = _write_dicom_nifti(files, destination)
    return destination, modality


def _write_dicom_nifti(files, destination):
    import pydicom
    import nibabel as nib
    from services.radiology_frames import calibrated_ct_pixels, dicom_intensity_pixels

    datasets = []
    for file in files:
        if not file.is_file():
            continue
        try:
            datasets.append(pydicom.dcmread(str(file), force=True))
        except Exception:
            continue
    if len(datasets) < 2:
        raise TotalSegmentatorError("The DICOM series does not contain a usable volume", "invalid_dicom")
    modality = str(getattr(datasets[0], "Modality", "") or "").upper()
    if modality not in {"CT", "MR"}:
        raise TotalSegmentatorError("TotalSegmentator accepts CT or MR DICOM only", "unsupported_modality")
    series = str(getattr(datasets[0], "SeriesInstanceUID", "") or "")
    orientation = tuple(float(value) for value in datasets[0].ImageOrientationPatient)
    rows = int(datasets[0].Rows)
    cols = int(datasets[0].Columns)
    for dataset in datasets[1:]:
        if str(getattr(dataset, "SeriesInstanceUID", "") or "") != series:
            raise TotalSegmentatorError("The DICOM folder mixes series", "invalid_dicom")
        if tuple(float(value) for value in dataset.ImageOrientationPatient) != orientation:
            raise TotalSegmentatorError("The DICOM series has inconsistent orientation", "invalid_dicom")
        if int(dataset.Rows) != rows or int(dataset.Columns) != cols:
            raise TotalSegmentatorError("The DICOM series has inconsistent image size", "invalid_dicom")
    row = np.array(orientation[:3], dtype=np.float64)
    col = np.array(orientation[3:], dtype=np.float64)
    normal = np.cross(row, col)
    normal = normal / np.linalg.norm(normal)

    def position(dataset):
        return np.array([float(value) for value in dataset.ImagePositionPatient], dtype=np.float64)

    datasets.sort(key=lambda dataset: float(np.dot(position(dataset), normal)))
    reader = calibrated_ct_pixels if modality == "CT" else dicom_intensity_pixels
    slices = [np.asarray(reader(dataset, 0), dtype=np.float32) for dataset in datasets]
    volume = np.stack(slices, axis=-1)
    spacing = [float(value) for value in datasets[0].PixelSpacing]
    step = float(np.dot(position(datasets[1]) - position(datasets[0]), normal))
    if abs(step) < 1e-6:
        raise TotalSegmentatorError("The DICOM series has no slice spacing", "invalid_dicom")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = col * spacing[1]
    affine[:3, 1] = row * spacing[0]
    affine[:3, 2] = normal * step
    affine[:3, 3] = position(datasets[0])
    # pixels are (row, column); store (column, row, slice) to match the affine columns
    stored = np.transpose(volume, (1, 0, 2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(stored, affine), str(destination))
    return modality
