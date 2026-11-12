from pathlib import Path

import numpy as np
import pytest

from services.image_validation import ImageValidationError, validate_image_file
from services.nrrd_reader import open_scalar_volume
from services.radiology_frames import nifti_volume_bytes, render_nifti_slice


def _write(path: Path, values, encoding="raw", directions=None):
    import nrrd

    header = {
        "space": "left-posterior-superior",
        "space directions": directions or [[1.5, 0, 0], [0, 2.0, 0], [0, 0, 3.0]],
        "space origin": [0, 0, 0],
        "encoding": encoding,
    }
    nrrd.write(str(path), np.ascontiguousarray(values), header)


def test_nrrd_preserves_negative_values_and_spacing(tmp_path):
    values = np.zeros((2, 3, 4), dtype=np.int16)
    values[0, 0, 0] = -1024
    values[1, 2, 3] = 589
    path = tmp_path / "ct.nrrd"
    _write(path, values)
    image = open_scalar_volume(path)
    assert tuple(image.shape) == (2, 3, 4)
    assert float(np.asanyarray(image.dataobj)[0, 0, 0]) == -1024
    zooms = image.header.get_zooms()[:3]
    assert zooms[0] == pytest.approx(1.5)
    assert zooms[1] == pytest.approx(2.0)
    assert zooms[2] == pytest.approx(3.0)
    details = validate_image_file(path)
    assert details["format"] == "NRRD"
    assert details["file_type"] == "nifti"


def test_extensionless_and_gzip_nrrd(tmp_path):
    values = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
    hashed = tmp_path / ("ab" * 16)
    _write(hashed, values, encoding="gzip")
    loaded = np.asanyarray(open_scalar_volume(hashed).dataobj)
    assert loaded[1, 1, 1] == 7


def test_invalid_nrrd_is_rejected(tmp_path):
    path = tmp_path / "bad.nrrd"
    path.write_text("not a volume", encoding="utf-8")
    with pytest.raises(ImageValidationError):
        open_scalar_volume(path)


def test_nrrd_slice_and_volume_match_source(tmp_path):
    values = np.zeros((2, 2, 2), dtype=np.float32)
    values[0, 0, 0] = -190
    path = tmp_path / "tiny.nrrd"
    _write(path, values, directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    payload, info = nifti_volume_bytes(path, quality="high")
    volume = np.frombuffer(payload, dtype=np.float32).reshape(info["dimensions"], order="F")
    assert volume[0, 0, 0] == np.float32(-190)
    png, count = render_nifti_slice(path, "axial", 0, None, None)
    assert count == 2
    assert png.startswith(b"\x89PNG")


def test_nhdr_rejects_parent_path(tmp_path):
    header = tmp_path / "vol.nhdr"
    header.write_text("NRRD0004\ndata file: ../secret.raw\n", encoding="utf-8")
    with pytest.raises(ImageValidationError, match="not allowed"):
        open_scalar_volume(header)
