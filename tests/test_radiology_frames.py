from pathlib import Path

from PIL import Image
from io import BytesIO

import numpy as np
import nibabel as nib

from services.radiology_frames import dicom_thumbnail_jpeg, hu_mask_png, nifti_calibration, nifti_plane_counts, nifti_volume_bytes, render_dicom_frame, render_nifti_slice

SAMPLE = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages" / "pydicom" / "data" / "test_files" / "693_J2KI.dcm"


def test_jpeg2000_ct_decodes_to_visible_png():
    payload = render_dicom_frame(SAMPLE, None, None)
    image = Image.open(BytesIO(payload))
    assert image.size == (512, 512)
    assert image.getextrema()[1] > 0


def test_ct_window_presets_change_the_image():
    lung = render_dicom_frame(SAMPLE, -600, 1500)
    bone = render_dicom_frame(SAMPLE, 300, 1500)
    assert lung != bone
    assert lung.startswith(b"\x89PNG")


def test_hu_mask_uses_calibrated_range():
    air, air_stats = hu_mask_png(SAMPLE, -1000, -500)
    dense, dense_stats = hu_mask_png(SAMPLE, 300, 1500)
    assert air.startswith(b"\x89PNG")
    assert 0 < air_stats["eligible_pixels"] <= 512 * 512
    assert air_stats["matching_pixels"] != dense_stats["matching_pixels"]
    assert air != dense


def test_nifti_hu_is_not_inferred_from_intensities():
    assert nifti_calibration({})["hu_threshold_supported"] is False
    assert nifti_calibration({"modality": "MR"})["calibration_status"] == "not_applicable"
    verified = nifti_calibration({"modality": "CT", "intensity_unit": "HU", "calibration_source": "dicom_conversion_metadata"})
    assert verified["hu_threshold_supported"] is True
    assert nifti_calibration({"modality": "CT", "intensity_unit": "HU"})["calibration_status"] == "unverified"


def test_nifti_planes_use_voxel_spacing(tmp_path):
    volume = np.zeros((4, 6, 8), dtype=np.float32)
    image = nib.Nifti1Image(volume, np.diag([1.0, 2.0, 4.0, 1.0]))
    path = tmp_path / "spacing.nii"
    nib.save(image, path)
    assert nifti_plane_counts(path) == {"axial": 8, "coronal": 6, "sagittal": 4}
    coronal = Image.open(BytesIO(render_nifti_slice(path, "coronal", 3, None, None)[0]))
    sagittal = Image.open(BytesIO(render_nifti_slice(path, "sagittal", 2, None, None)[0]))
    assert coronal.size == (32, 4)
    assert sagittal.size == (16, 6)


def test_jpeg2000_thumbnail_is_jpeg():
    payload = dicom_thumbnail_jpeg(SAMPLE)
    image = Image.open(BytesIO(payload))
    assert image.format == "JPEG"
    assert image.size[0] <= 512
