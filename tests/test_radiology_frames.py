from pathlib import Path

from PIL import Image
from io import BytesIO

import numpy as np
import nibabel as nib

from pydicom.dataset import Dataset

from services.radiology_frames import _apply_modality_values, _downsample, _mask_png_from_values, dicom_intensity_mask_png, dicom_thumbnail_jpeg, hu_mask_png, nifti_calibration, nifti_plane_counts, nifti_volume_bytes, render_dicom_frame, render_nifti_slice

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
    assert air_stats["value_min"] is not None
    assert air_stats["value_max"] is not None


def test_dicom_intensity_mask_matches_hu_for_ct():
    air_hu, air_hu_stats = hu_mask_png(SAMPLE, -1000, -500)
    air_i, air_i_stats = dicom_intensity_mask_png(SAMPLE, -1000, -500)
    assert air_i == air_hu
    assert air_i_stats["matching_pixels"] == air_hu_stats["matching_pixels"]


def test_rescale_is_applied_once_and_bounds_are_inclusive():
    dataset = Dataset()
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = -1024
    stored = np.array([[0, 512], [662, 100]], dtype=np.int16)
    values = _apply_modality_values(dataset, stored)
    assert values.tolist() == [[-1024, 0], [300, -824]]
    _png, stats = _mask_png_from_values(values, -1024, 0, "red")
    assert stats["matching_pixels"] == 3
    _png, above = _mask_png_from_values(values, 300, 300, "red")
    assert above["matching_pixels"] == 1


def test_nifti_volume_applies_slope_once(tmp_path):
    raw = np.zeros((2, 2, 2), dtype=np.int16)
    raw[0, 0, 0] = 10
    image = nib.Nifti1Image(raw, np.eye(4))
    image.header.set_slope_inter(2, -5)
    path = tmp_path / "scaled.nii"
    nib.save(image, path)
    payload, info = nifti_volume_bytes(path, quality="high")
    values = np.frombuffer(payload, dtype=np.float32).reshape(info["dimensions"], order="F")
    assert values[0, 0, 0] == np.float32(15)
    assert info["preview"] is False


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
    assert coronal.size == (4, 32)
    assert sagittal.size == (6, 16)


def test_threshold_interval_is_inclusive_and_exact():
    values = np.array([-1024, -950, -500, -190, 40, 300, 1500], dtype=np.float32)
    lung = (values >= -950) & (values <= -500)
    assert lung.tolist() == [False, True, True, False, False, False, False]
    exact = (values >= 300) & (values <= 300)
    assert exact.tolist() == [False, False, False, False, False, True, False]
    empty = (values >= 1300) & (values <= 1400)
    assert not empty.any()


def test_volume_downsample_keeps_source_samples(tmp_path):
    volume = np.zeros((64, 64, 520), dtype=np.float32)
    volume[0, 0, 0] = -1024
    volume[-1, -1, -1] = 300
    reduced, _, preview = _downsample(volume, (1, 1, 1), "preview", nearest=True)
    assert preview is True
    flat = set(np.unique(reduced).tolist())
    assert flat <= {0.0, -1024.0, 300.0}
    image = nib.Nifti1Image(volume, np.eye(4))
    path = tmp_path / "block.nii"
    nib.save(image, path)
    payload, info = nifti_volume_bytes(path, quality="preview")
    loaded = np.frombuffer(payload, dtype=np.float32)
    assert set(np.unique(loaded).tolist()) <= {0.0, -1024.0, 300.0}
    assert info["dimensions"] == list(reduced.shape)


def test_jpeg2000_thumbnail_is_jpeg():
    payload = dicom_thumbnail_jpeg(SAMPLE)
    image = Image.open(BytesIO(payload))
    assert image.format == "JPEG"
    assert image.size[0] <= 512
