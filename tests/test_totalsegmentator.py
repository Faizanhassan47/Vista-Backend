from pathlib import Path

import numpy as np
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from services.totalsegmentator_service import (
    apply_modality_verification,
    class_ids,
    command,
    measure_nifti,
    normalize_rois,
    structures_for_modality,
    task_for_modality,
    touches_boundary,
    voxel_volume_ml,
)

ROOT = Path(r"E:\total-segmentation")


def test_modality_selects_the_verified_task():
    assert task_for_modality("CT") == "total"
    assert task_for_modality("MR") == "total_mr"
    assert task_for_modality("MRI") == "total_mr"
    with pytest.raises(Exception):
        task_for_modality("NIfTI")
    with pytest.raises(Exception):
        task_for_modality("")


def test_catalog_matches_installed_tasks():
    ct = class_ids("total")
    mr = class_ids("total_mr")
    assert len(ct) == 117
    assert len(mr) == 50
    assert len(set(ct)) == 117
    assert "lung_left" not in ct
    assert "lung_upper_lobe_left" not in mr
    assert structures_for_modality("CT")["count"] == 117
    assert structures_for_modality("MR")["count"] == 50
    with pytest.raises(Exception):
        structures_for_modality("PT")


def test_unknown_roi_is_rejected():
    assert normalize_rois(["liver", "liver", "spleen"]) == ["liver", "spleen"]
    with pytest.raises(Exception):
        normalize_rois(["tumor"])
    with pytest.raises(Exception):
        normalize_rois([])
    with pytest.raises(Exception):
        normalize_rois(["lung_left"], task="total")
    assert len(normalize_rois(class_ids("total"), task="total")) == 117


def test_command_is_an_argument_list():
    argv = command("python", "in.nii.gz", "out", "total", ["liver", "spleen"], fast=True)
    assert argv[0] == "python"
    assert "-ta" in argv and argv[argv.index("-ta") + 1] == "total"
    assert "-d" in argv and argv[argv.index("-d") + 1] == "cpu"
    assert "-f" in argv
    assert "liver" in argv and "spleen" in argv
    assert "-rs" in argv
    full = command("python", "in.nii.gz", "out", "total", class_ids("total"), fast=True)
    assert "-rs" not in full
    assert not any(" " in part for part in argv)


def test_volume_and_boundary_use_physical_voxels():
    affine = np.diag([1.25, 1.25, 3.0, 1.0])
    assert voxel_volume_ml(affine) == pytest.approx(1.25 * 1.25 * 3.0 / 1000)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1, 1, 1] = 1
    assert touches_boundary(mask) is False
    mask[0, 1, 1] = 1
    assert touches_boundary(mask) is True


def test_existing_mri_masks_match_recorded_volumes():
    image = ROOT / "input" / "MRI" / "s0288.nii"
    masks = ROOT / "output" / "MRI" / "s0288"
    if not image.is_file():
        pytest.skip("Independent s0288 MRI is not present")
    liver = measure_nifti(image, masks / "liver.nii.gz", "liver", report_hu=False)
    left = measure_nifti(image, masks / "kidney_left.nii.gz", "kidney_left", report_hu=False)
    right = measure_nifti(image, masks / "kidney_right.nii.gz", "kidney_right", report_hu=False)
    assert liver["volume_ml"] == pytest.approx(976.48, abs=0.2)
    assert "hu_mean" not in liver
    assert left["potentially_truncated"] is True
    assert right["potentially_truncated"] is True
    assert left["coverage_warning"]


def test_existing_ct_mask_reports_hu():
    image = ROOT / "input" / "CT" / "s0015.nii"
    mask = ROOT / "output" / "CT" / "s0015" / "liver.nii.gz"
    if not image.is_file() or not mask.is_file():
        pytest.skip("Independent s0015 CT is not present")
    liver = measure_nifti(image, mask, "liver", report_hu=True)
    assert liver["voxel_count"] > 0
    assert liver["hu_mean"] is not None
    assert liver["volume_ml"] > 0


def test_verification_does_not_relabel_ct_as_mri():
    updated = apply_modality_verification({}, "CT", "etd", "2026-11-20T00:00:00+00:00")
    assert updated["modality"] == "CT"
    assert updated["modality_verification"]["source"] == "authorized_user"
    with pytest.raises(Exception):
        apply_modality_verification({"modality": "CT"}, "MR", "etd", "2026-11-20T00:00:00+00:00")


def test_job_endpoint_requires_assignment():
    from routes.totalsegmentator_api import create_totalsegmentator_router

    def allow(_roles):
        return lambda: {"username": "other"}

    denied = FastAPI()

    def forbid(_case, _user):
        raise HTTPException(403, "Not assigned")

    denied.include_router(create_totalsegmentator_router(
        connect_func=None,
        require_roles_func=allow,
        get_current_user_func=lambda: {"username": "other"},
        get_slide_func=lambda _slide_id: {"id": 3, "case_id": 1, "file_type": "nifti", "filename": "volume.nii"},
        get_case_func=lambda _case_id: {"id": 1},
        stored_path_func=lambda name: name,
        require_assigned_case_func=forbid,
        audit_func=None,
    ))
    response = TestClient(denied).post("/api/ai/totalsegmentator/jobs", json={"slide_id": 3, "rois": ["liver"]})
    assert response.status_code == 403

    anonymous = FastAPI()

    def reject(_roles):
        def dependency(user=Depends(lambda: None)):
            if user is None:
                raise HTTPException(401, "Authentication required")
            return user
        return dependency

    anonymous.include_router(create_totalsegmentator_router(
        connect_func=None,
        require_roles_func=reject,
        get_current_user_func=lambda: None,
        get_slide_func=lambda _slide_id: {"id": 3, "case_id": 1, "file_type": "nifti", "filename": "volume.nii"},
        get_case_func=lambda _case_id: {"id": 1},
        stored_path_func=lambda name: name,
        require_assigned_case_func=lambda _case, _user: None,
        audit_func=None,
    ))
    assert TestClient(anonymous).post("/api/ai/totalsegmentator/jobs", json={"slide_id": 3}).status_code == 401
