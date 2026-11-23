"""Authenticated NeuroMRI slice classification. Research use only."""

from datetime import datetime, timezone
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core import AI_READ_ROLES, AI_RUN_ROLES
from services.neuromri_engine import NeuromriUnavailable, predict_image, status as model_status
from services.neuromri_gradcam import explain_image
from services.neuromri_preprocessing import NeuromriInputError, apply_mri_verification, dicom_mr_slice, image_from_upload, nifti_mr_slice

logger = logging.getLogger("pathology.neuromri")

# Set True to allow heatmap generation again. The Grad-CAM service stays in place.
GRADCAM_ENABLED = False


class SliceRequest(BaseModel):
    slide_id: int
    plane: str = "axial"
    index: int = Field(0, ge=0)
    frame_index: int = Field(0, ge=0)
    target_class: str | None = None


def create_neuromri_router(
    connect_func,
    require_roles_func,
    get_current_user_func,
    get_slide_func,
    get_case_func,
    stored_path_func,
    require_assigned_case_func,
    audit_func=None,
):
    router = APIRouter(tags=["NeuroMRI"])

    def record(user, action, slide_id, details):
        if not audit_func or not connect_func:
            return
        try:
            from contextlib import closing
            with closing(connect_func()) as db:
                audit_func(db, user.get("username", ""), action, "slide", str(slide_id or ""), details)
        except Exception:
            logger.exception("NeuroMRI audit write failed")

    @router.get("/api/ai/neuromri/status", dependencies=[Depends(require_roles_func(AI_READ_ROLES))])
    def neuromri_status():
        return model_status()

    @router.post("/api/ai/neuromri/predict", dependencies=[Depends(require_roles_func(AI_RUN_ROLES))])
    async def neuromri_predict(body: SliceRequest, current_user: dict = Depends(get_current_user_func)):
        slide = get_slide_func(body.slide_id)
        case = get_case_func(slide["case_id"])
        require_assigned_case_func(case, current_user)
        if slide["file_type"] not in {"dicom-stack", "nifti", "raster"}:
            raise HTTPException(400, "NeuroMRI classifies a JPG, PNG, MR DICOM slice, or MRI NIfTI slice")
        started = datetime.now(timezone.utc)
        try:
            image, source = await run_in_threadpool(_slice_image, slide, body, stored_path_func, connect_func)
            result = await run_in_threadpool(predict_image, image, source)
        except NeuromriInputError as exc:
            record(current_user, "neuromri.predict_failed", body.slide_id, f"category={exc.category}")
            raise HTTPException(400, str(exc)) from exc
        except NeuromriUnavailable as exc:
            record(current_user, "neuromri.predict_failed", body.slide_id, "category=model_unavailable")
            raise HTTPException(503, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("NeuroMRI inference failed")
            record(current_user, "neuromri.predict_failed", body.slide_id, "category=inference_failed")
            raise HTTPException(500, "NeuroMRI could not classify this slice") from exc
        result["applies_to"] = "selected_2d_slice_only"
        record(
            current_user,
            "neuromri.predict",
            body.slide_id,
            f"status=completed; version={result['model_version']}; plane={source.get('plane') or 'frame'}; index={source.get('slice_index', source.get('frame_index'))}; seconds={result['inference_seconds']}",
        )
        result["requested_at"] = started.isoformat()
        return result

    @router.post("/api/ai/neuromri/predict-image", dependencies=[Depends(require_roles_func(AI_RUN_ROLES))])
    async def neuromri_predict_image(
        file: UploadFile = File(...),
        current_user: dict = Depends(get_current_user_func),
    ):
        name = (file.filename or "").lower()
        if not name.endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(400, "Upload a JPG or PNG image")
        payload = await file.read()
        try:
            image = image_from_upload(payload, file.content_type)
            result = await run_in_threadpool(predict_image, image, {"source_format": "upload", "name": Path(name).name})
        except NeuromriInputError as exc:
            raise HTTPException(400, str(exc)) from exc
        except NeuromriUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        record(current_user, "neuromri.predict_upload", "", f"status=completed; version={result['model_version']}; seconds={result['inference_seconds']}")
        return result

    @router.post("/api/ai/neuromri/gradcam", dependencies=[Depends(require_roles_func(AI_RUN_ROLES))])
    async def neuromri_gradcam(body: SliceRequest, current_user: dict = Depends(get_current_user_func)):
        slide = get_slide_func(body.slide_id)
        case = get_case_func(slide["case_id"])
        require_assigned_case_func(case, current_user)
        if not GRADCAM_ENABLED:
            raise HTTPException(503, "NeuroMRI Grad-CAM is temporarily disabled")
        if slide["file_type"] not in {"dicom-stack", "nifti", "raster"}:
            raise HTTPException(400, "NeuroMRI Grad-CAM accepts a JPG, PNG, MR DICOM slice, or MRI NIfTI slice")
        try:
            image, source = await run_in_threadpool(_slice_image, slide, body, stored_path_func, connect_func)
            result = await run_in_threadpool(explain_image, image, body.target_class)
        except NeuromriInputError as exc:
            raise HTTPException(400, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except NeuromriUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            logger.exception("NeuroMRI Grad-CAM failed")
            raise HTTPException(500, "NeuroMRI could not build a Grad-CAM heatmap") from exc
        result["status"] = "completed"
        result["source_type"] = source.get("modality") or source.get("source_format")
        result["plane"] = source.get("plane")
        result["slice_index"] = source.get("slice_index", source.get("frame_index"))
        record(
            current_user,
            "neuromri.gradcam",
            body.slide_id,
            f"status=completed; target={result['target_class']}; plane={result['plane']}; index={result['slice_index']}",
        )
        return result

    @router.post("/api/ai/neuromri/verify-modality", dependencies=[Depends(require_roles_func(AI_RUN_ROLES))])
    def neuromri_verify_modality(body: SliceRequest, current_user: dict = Depends(get_current_user_func)):
        slide = get_slide_func(body.slide_id)
        case = get_case_func(slide["case_id"])
        require_assigned_case_func(case, current_user)
        if slide["file_type"] != "nifti":
            raise HTTPException(400, "MRI verification is only for a NIfTI study with no stored modality")
        if not connect_func:
            raise HTTPException(503, "Modality verification is not available")
        import json
        from contextlib import closing
        from services.radiology_frames import nifti_calibration
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
                updated = apply_mri_verification(metadata, current_user.get("username", ""), verified_at)
                db.execute(
                    "UPDATE image_versions SET metadata_json = ? WHERE id = ?",
                    (json.dumps(updated), version["id"]),
                )
                db.commit()
        except NeuromriInputError as exc:
            raise HTTPException(400, str(exc)) from exc
        calibration = nifti_calibration(updated)
        record(current_user, "neuromri.verify_modality", body.slide_id, "modality=MR; source=authorized_user")
        return {"modality": "MR", "calibration": calibration, "verification": updated.get("modality_verification")}

    return router


def _metadata(connect_func, slide_id):
    if not connect_func:
        return {}
    from contextlib import closing
    import json
    with closing(connect_func()) as db:
        version = db.execute(
            "SELECT metadata_json FROM image_versions WHERE slide_id = ? ORDER BY version DESC",
            (slide_id,),
        ).fetchone()
    if not version or not version.get("metadata_json"):
        return {}
    return json.loads(version["metadata_json"])


def _slice_image(slide, body, stored_path_func, connect_func):
    path = Path(stored_path_func(slide["filename"]))
    if slide["file_type"] == "raster":
        from PIL import Image
        if not path.is_file():
            raise NeuromriInputError("The stored image is missing", "missing_image")
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise NeuromriInputError("NeuroMRI accepts a JPG or PNG picture", "invalid_image")
        with Image.open(path) as image:
            loaded = image.convert("RGB")
        return loaded, {"source_format": "image", "modality": "image", "name": path.name, "slice_index": 0}
    if slide["file_type"] == "nifti":
        source = path if path.is_file() else next(path.parent.glob("*.nii*"), None)
        if source is None or not Path(source).is_file():
            raise NeuromriInputError("The stored NIfTI volume is missing", "missing_image")
        metadata = _metadata(connect_func, slide["id"])
        return nifti_mr_slice(source, body.plane, body.index, metadata)
    metadata = _metadata(connect_func, slide["id"])
    ordered = [Path(item["path"]) for item in metadata.get("instances") or [] if item.get("path")]
    instances = ordered or sorted(item for item in path.parent.iterdir() if item.is_file()) if path.parent.is_dir() else [path]
    if body.index >= len(instances):
        raise NeuromriInputError("The selected slice is outside this series", "invalid_slice")
    instance = instances[body.index]
    if not instance.is_file():
        raise NeuromriInputError("The stored DICOM instance is missing", "missing_image")
    image, info = dicom_mr_slice(instance, body.frame_index)
    info["instance_index"] = body.index
    info["slice_index"] = body.index
    return image, info
