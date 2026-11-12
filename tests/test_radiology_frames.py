from pathlib import Path

from PIL import Image
from io import BytesIO

import numpy as np
import nibabel as nib

from services.radiology_frames import (
    calibrated_ct_pixels,
    dicom_intensity_mask_png,
    dicom_thumbnail_jpeg,
    dicom_volume_bytes,
    hu_mask_png,
    nifti_calibration,
    nifti_plane_counts,
    nifti_volume_bytes,
    render_dicom_frame,
    render_nifti_slice,
)
import pydicom

SAMPLE = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages" / "pydicom" / "data" / "test_files" / "693_J2KI.dcm"
CT_SMALL = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages" / "pydicom" / "data" / "test_files" / "CT_small.dcm"


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
    assert air_stats["value_min"] is not None
    assert air_stats["value_max"] is not None


def test_dicom_intensity_mask_matches_hu_for_ct():
    air_hu, air_hu_stats = hu_mask_png(SAMPLE, -1000, -500)
    air_i, air_i_stats = dicom_intensity_mask_png(SAMPLE, -1000, -500)
    assert air_i == air_hu
    assert air_i_stats["matching_pixels"] == air_hu_stats["matching_pixels"]


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


def test_dicom_volume_preserves_calibrated_hu(tmp_path):
    """3D volume scalars must stay in HU — never display-windowed to 0–255."""
    import shutil

    assert CT_SMALL.is_file()
    dataset = pydicom.dcmread(str(CT_SMALL))
    frame = calibrated_ct_pixels(dataset)
    assert float(np.nanmin(frame)) < -500
    assert float(np.nanmax(frame)) > 400

    instances = []
    for index in range(12):
        target = tmp_path / f"ct_{index:03d}.dcm"
        shutil.copy(CT_SMALL, target)
        instances.append(target)

    payload, info = dicom_volume_bytes(instances, "preview")
    values = np.frombuffer(payload, dtype=np.float32)
    assert info["dtype"] == "float32"
    assert info["intensity_unit"] == "HU"
    assert info["value_min"] < -500
    assert info["value_max"] > 400
    assert float(values.min()) < -500
    assert float(values.max()) > 400
    # Must not look like 8-bit display data.
    assert not (0 <= float(values.min()) <= 1 and 200 <= float(values.max()) <= 255)


def test_nifti_volume_preserves_native_intensities(tmp_path):
    volume = np.zeros((32, 32, 16), dtype=np.float32)
    volume[8:24, 8:24, 4:12] = 180.0
    volume[10:14, 10:14, 6:10] = 900.0
    path = tmp_path / "native.nii"
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
    payload, info = nifti_volume_bytes(path, "preview")
    values = np.frombuffer(payload, dtype=np.float32)
    assert info["dtype"] == "float32"
    assert info["value_max"] >= 180
    assert float(values.max()) >= 180
