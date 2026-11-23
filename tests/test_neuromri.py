import hashlib
from pathlib import Path

import numpy as np
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from services.neuromri_engine import CHECKPOINT, EXPECTED_SHA256, get_model, predict_path
from services.neuromri_preprocessing import CLASS_NAMES, NeuromriInputError, dicom_mr_slice, nifti_mr_slice

ORIGINAL = Path(r"E:\NeuroMRI\outputs\checkpoints\best_model.pth")
SAMPLE = Path(r"C:\Users\USER\Desktop\testimage.jpg")


def test_checkpoint_hash_matches_training_file():
    assert CHECKPOINT.is_file()
    assert ORIGINAL.is_file()

    def digest(path):
        hasher = hashlib.sha256()
        hasher.update(path.read_bytes())
        return hasher.hexdigest()

    copied = digest(CHECKPOINT)
    assert copied == digest(ORIGINAL) == EXPECTED_SHA256


def test_model_outputs_ten_logits():
    torch = pytest.importorskip("torch")
    model = get_model()
    assert next(model.parameters()).device.type == "cpu"
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert tuple(logits.shape) == (1, 10)


def test_jpg_regression_matches_standalone_result():
    if not SAMPLE.is_file():
        pytest.skip("Desktop test image is not present")
    pytest.importorskip("torch")
    result = predict_path(SAMPLE)
    assert result["predicted_class"] == "AD Moderate"
    assert result["device"] == "cpu"
    assert result["top_softmax_score"] == pytest.approx(0.9999799728393555, abs=1e-5)
    assert list(result["class_scores"]) == CLASS_NAMES


def test_ct_dicom_is_rejected(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    path = tmp_path / "ct.dcm"
    dataset = pydicom.Dataset()
    dataset.Modality = "CT"
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows = 16
    dataset.Columns = 16
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PixelData = np.zeros((16, 16), dtype=np.uint16).tobytes()
    dataset.file_meta = pydicom.dataset.FileMetaDataset()
    dataset.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    dataset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    dataset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.save_as(path, enforce_file_format=True)
    with pytest.raises(NeuromriInputError) as raised:
        dicom_mr_slice(path, 0)
    assert raised.value.category == "unsupported_modality"


def test_mri_verification_is_recorded_and_ct_cannot_be_relabeled():
    from services.neuromri_preprocessing import apply_mri_verification
    from services.radiology_frames import nifti_calibration

    verified = apply_mri_verification({}, "etd", "2026-11-19T00:00:00+00:00")
    assert verified["modality"] == "MR"
    assert verified["modality_verification"]["source"] == "authorized_user"
    assert verified["modality_verification"]["verified_by"] == "etd"
    assert nifti_calibration(verified)["modality"] == "MR"
    with pytest.raises(NeuromriInputError):
        apply_mri_verification({"modality": "CT"}, "etd", "2026-11-19T00:00:00+00:00")
    unchanged = apply_mri_verification({"modality": "MR"}, "etd", "2026-11-19T00:00:00+00:00")
    assert "modality_verification" not in unchanged


def test_ct_nifti_metadata_is_rejected(tmp_path):
    nib = pytest.importorskip("nibabel")
    path = tmp_path / "ct.nii.gz"
    nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)).to_filename(path)
    with pytest.raises(NeuromriInputError) as raised:
        nifti_mr_slice(path, "axial", 0, {"modality": "CT", "intensity_unit": "HU", "calibration_source": "dicom_conversion_metadata"})
    assert raised.value.category == "unsupported_modality"


def test_unknown_nifti_modality_is_not_treated_as_mri(tmp_path):
    nib = pytest.importorskip("nibabel")
    path = tmp_path / "volume.nii.gz"
    nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)).to_filename(path)
    with pytest.raises(NeuromriInputError) as raised:
        nifti_mr_slice(path, "axial", 1, {})
    assert raised.value.category == "unverified_modality"


def test_nifti_slice_index_is_preserved(tmp_path):
    nib = pytest.importorskip("nibabel")
    path = tmp_path / "mr.nii.gz"
    volume = np.arange(32 * 32 * 16, dtype=np.float32).reshape(32, 32, 16)
    nib.Nifti1Image(volume, np.eye(4)).to_filename(path)
    _image, info = nifti_mr_slice(path, "sagittal", 3, {"modality": "MR"})
    assert info["plane"] == "sagittal"
    assert info["slice_index"] == 3
    with pytest.raises(NeuromriInputError) as raised:
        nifti_mr_slice(path, "sagittal", 99, {"modality": "MR"})
    assert raised.value.category == "invalid_slice"


def test_predict_requires_authentication_and_case_access():
    from routes.neuromri_api import create_neuromri_router

    def require_roles(_roles):
        def dependency(user=Depends(lambda: None)):
            if user is None:
                raise HTTPException(401, "Authentication required")
            return user
        return dependency

    def get_user():
        return None

    app = FastAPI()
    app.include_router(create_neuromri_router(
        connect_func=None,
        require_roles_func=require_roles,
        get_current_user_func=get_user,
        get_slide_func=lambda _slide_id: {"id": 1, "case_id": 1, "file_type": "nifti", "filename": "missing.nii"},
        get_case_func=lambda _case_id: {"id": 1},
        stored_path_func=lambda name: name,
        require_assigned_case_func=lambda _case, _user: None,
        audit_func=None,
    ))
    client = TestClient(app)
    response = client.post("/api/ai/neuromri/predict", json={"slide_id": 1, "plane": "axial", "index": 0})
    assert response.status_code == 401

    def allow(_roles):
        return lambda: {"username": "other"}

    denied = FastAPI()

    def forbid(_case, _user):
        raise HTTPException(403, "Not assigned")

    denied.include_router(create_neuromri_router(
        connect_func=None,
        require_roles_func=allow,
        get_current_user_func=lambda: {"username": "other"},
        get_slide_func=lambda _slide_id: {"id": 9, "case_id": 3, "file_type": "dicom-stack", "filename": "series"},
        get_case_func=lambda _case_id: {"id": 3},
        stored_path_func=lambda name: name,
        require_assigned_case_func=forbid,
        audit_func=None,
    ))
    blocked = TestClient(denied).post("/api/ai/neuromri/predict", json={"slide_id": 9, "index": 0})
    assert blocked.status_code == 403
    blocked_cam = TestClient(denied).post("/api/ai/neuromri/gradcam", json={"slide_id": 9, "index": 0})
    assert blocked_cam.status_code == 403

    enabled = FastAPI()
    enabled.include_router(create_neuromri_router(
        connect_func=None,
        require_roles_func=allow,
        get_current_user_func=lambda: {"username": "reader"},
        get_slide_func=lambda _slide_id: {"id": 4, "case_id": 1, "file_type": "raster", "filename": "brain.jpg"},
        get_case_func=lambda _case_id: {"id": 1},
        stored_path_func=lambda name: name,
        require_assigned_case_func=lambda _case, _user: None,
        audit_func=None,
    ))
    disabled = TestClient(enabled).post("/api/ai/neuromri/gradcam", json={"slide_id": 4, "index": 0})
    assert disabled.status_code == 503
    assert "temporarily disabled" in disabled.json()["detail"]
    assert client.post("/api/ai/neuromri/gradcam", json={"slide_id": 1}).status_code == 401


def test_gradcam_matches_classification_and_aligns():
    if not SAMPLE.is_file():
        pytest.skip("Desktop test image is not present")
    torch = pytest.importorskip("torch")
    from services.neuromri_gradcam import explain_image

    model = get_model()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    before_scores = predict_path(SAMPLE)["class_scores"]
    with Image.open(SAMPLE) as source:
        image = source.convert("RGB")
    import time
    started = time.perf_counter()
    explained = explain_image(image)
    elapsed = time.perf_counter() - started
    print(f"gradcam_seconds={elapsed:.3f}")
    assert explained["predicted_class"] == "AD Moderate"
    assert explained["target_class"] == "AD Moderate"
    assert explained["target_class_index"] == CLASS_NAMES.index("AD Moderate")
    for name, score in before_scores.items():
        assert explained["class_scores"][name] == pytest.approx(score, abs=1e-5)
    overlay = Image.open(__import__("io").BytesIO(__import__("base64").b64decode(explained["overlay_png_base64"])))
    assert overlay.size == image.size
    assert overlay.mode == "RGBA"
    heat = np.asarray(Image.open(__import__("io").BytesIO(__import__("base64").b64decode(explained["heatmap_png_base64"]))))
    assert heat.shape[:2] == (image.size[1], image.size[0])
    assert np.isfinite(heat).all()
    other = explain_image(image, "Normal")
    assert other["target_class"] == "Normal"
    assert other["target_class_index"] == CLASS_NAMES.index("Normal")
    assert other["predicted_class"] == "AD Moderate"
    layer = model.features[-1]
    assert len(layer._forward_hooks) == 0
    assert len(layer._forward_pre_hooks) == 0
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), before[name])
        assert parameter.grad is None
    wide = Image.new("RGB", (40, 20), (20, 40, 60))
    wide_cam = explain_image(wide, "Brain Stroke")
    wide_overlay = Image.open(__import__("io").BytesIO(__import__("base64").b64decode(wide_cam["overlay_png_base64"])))
    assert wide_overlay.size == (40, 20)
    assert wide_cam["target_class"] == "Brain Stroke"
    again = predict_path(SAMPLE)
    assert again["predicted_class"] == "AD Moderate"
    assert again["top_softmax_score"] == pytest.approx(0.9999799728393555, abs=1e-5)
