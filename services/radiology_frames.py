"""Authenticated slice rendering for DICOM stacks and 3D NIfTI volumes."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from services.image_validation import ImageValidationError


def _png(values: np.ndarray) -> bytes:
    image = Image.fromarray(np.ascontiguousarray(values), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _window(values: np.ndarray, center: float | None, width: float | None) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    if center is None or width is None or width <= 0:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            low, high = float(finite.min()), float(finite.max() or finite.min() + 1)
    else:
        low, high = center - width / 2, center + width / 2
    scaled = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def _first_number(value):
    if value is None:
        return None
    item = value[0] if isinstance(value, (list, tuple)) else value
    try:
        return float(item)
    except (TypeError, ValueError):
        return None


def render_dicom_frame(path, center: float | None, width: float | None) -> bytes:
    import pydicom
    from pydicom.pixels import apply_modality_lut

    dataset = pydicom.dcmread(str(path))
    syntax = str(getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", "") or "")
    try:
        pixels = dataset.pixel_array
    except Exception as exc:
        raise ImageValidationError(f"Transfer syntax {syntax or 'unknown'} could not be decoded: {exc}") from exc
    if pixels.ndim == 3:
        pixels = pixels[0]
    try:
        pixels = apply_modality_lut(pixels, dataset).astype(np.float32)
    except Exception:
        pixels = pixels.astype(np.float32) * float(getattr(dataset, "RescaleSlope", 1) or 1) + float(getattr(dataset, "RescaleIntercept", 0) or 0)
    if center is None or width is None:
        center = center if center is not None else _first_number(getattr(dataset, "WindowCenter", None))
        width = width if width is not None else _first_number(getattr(dataset, "WindowWidth", None))
    display = _window(pixels, center, width)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        display = 255 - display
    return _png(display)


def _jpeg(png: bytes) -> bytes:
    image = Image.open(BytesIO(png)).convert("RGB")
    image.thumbnail((512, 512))
    output = BytesIO()
    image.save(output, format="JPEG", quality=82)
    return output.getvalue()


COLORS = {
    "red": (239, 68, 68),
    "green": (34, 197, 94),
    "blue": (59, 130, 246),
    "yellow": (234, 179, 8),
}


def calibrated_ct_pixels(dataset):
    """Return Hounsfield values for a CT dataset, or raise if they are not HU."""
    from pydicom.pixels import apply_modality_lut

    modality = str(getattr(dataset, "Modality", "") or "").upper()
    if modality != "CT":
        raise ImageValidationError("HU thresholding is only available for CT images")
    rescale_type = str(getattr(dataset, "RescaleType", "") or "").strip().upper()
    if rescale_type and rescale_type not in {"HU", "HOUNSFIELD"}:
        raise ImageValidationError(f"RescaleType {rescale_type} is not Hounsfield units")
    has_lut = bool(getattr(dataset, "ModalityLUTSequence", None))
    has_rescale = hasattr(dataset, "RescaleSlope") or hasattr(dataset, "RescaleIntercept")
    if not has_lut and not has_rescale:
        raise ImageValidationError("This CT image has no Hounsfield calibration")
    pixels = dataset.pixel_array
    if pixels.ndim == 3:
        pixels = pixels[0]
    try:
        values = apply_modality_lut(pixels, dataset).astype(np.float32)
    except Exception as exc:
        raise ImageValidationError(f"Modality transformation failed: {exc}") from exc
    padding = getattr(dataset, "PixelPaddingValue", None)
    if padding is not None:
        try:
            values = np.where(pixels.astype(np.float32) == float(padding), np.nan, values)
        except (TypeError, ValueError):
            pass
    return values


def hu_mask_png(path, min_hu: float, max_hu: float, color: str = "red") -> tuple[bytes, dict]:
    import pydicom

    if min_hu > max_hu:
        raise ImageValidationError("Maximum HU must be greater than or equal to minimum HU")
    dataset = pydicom.dcmread(str(path))
    syntax = str(getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", "") or "")
    try:
        values = calibrated_ct_pixels(dataset)
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(f"Transfer syntax {syntax or 'unknown'} could not be decoded: {exc}") from exc
    eligible = np.isfinite(values)
    mask = eligible & (values >= min_hu) & (values <= max_hu)
    rgb = COLORS.get(color, COLORS["red"])
    image = np.zeros((*mask.shape, 4), dtype=np.uint8)
    image[mask] = (*rgb, 255)
    buffer = BytesIO()
    Image.fromarray(image, mode="RGBA").save(buffer, format="PNG")
    eligible_count = int(eligible.sum())
    matching = int(mask.sum())
    return buffer.getvalue(), {
        "min_hu": min_hu,
        "max_hu": max_hu,
        "matching_pixels": matching,
        "eligible_pixels": eligible_count,
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "percentage": round(100 * matching / eligible_count, 2) if eligible_count else 0,
    }


def dicom_thumbnail_jpeg(path) -> bytes:
    return _jpeg(render_dicom_frame(path, None, None))


def nifti_thumbnail_jpeg(path) -> bytes:
    payload, _count = render_nifti_slice(path, "axial", 0, None, None)
    return _jpeg(payload)


def _nifti_geometry(image):
    import nibabel as nib
    if len(image.shape) != 3:
        raise ImageValidationError("Only 3D NIfTI volumes can be displayed")
    codes = nib.aff2axcodes(image.affine)
    if any(code == "?" or not code for code in codes):
        raise ImageValidationError("NIfTI affine does not define an anatomical orientation")
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    if any(value <= 0 for value in zooms):
        raise ImageValidationError("NIfTI voxel spacing is missing or invalid")
    return codes, zooms


def _axis_for(codes, letters):
    for index, code in enumerate(codes):
        if code in letters:
            return index
    raise ImageValidationError("NIfTI orientation is missing an anatomical axis")


def nifti_plane_counts(path) -> dict:
    import nibabel as nib
    image = nib.load(str(path))
    codes, _zooms = _nifti_geometry(image)
    return {
        "axial": int(image.shape[_axis_for(codes, "SI")]),
        "coronal": int(image.shape[_axis_for(codes, "AP")]),
        "sagittal": int(image.shape[_axis_for(codes, "LR")]),
    }


def _display_plane(values, row_code, col_code, plane):
    """Put superior or anterior at the top and the requested side at image left."""
    top = "A" if plane == "axial" else "S"
    left = "A" if plane == "sagittal" else "R"
    if row_code == top:
        values = np.flipud(values)
    if col_code == left:
        values = np.fliplr(values)
    return values


def _match_spacing(values, row_spacing, col_spacing):
    """One linear resample so displayed pixels match the finer in-plane spacing."""
    import cv2
    target = min(row_spacing, col_spacing)
    height = max(1, int(round(values.shape[0] * row_spacing / target)))
    width = max(1, int(round(values.shape[1] * col_spacing / target)))
    if (height, width) == values.shape:
        return values
    return cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)


def render_nifti_slice(path, plane: str, index: int, center: float | None, width: float | None) -> tuple[bytes, int]:
    import nibabel as nib
    image = nib.load(str(path))
    codes, zooms = _nifti_geometry(image)
    plane = plane if plane in {"axial", "coronal", "sagittal"} else "axial"
    through = {"axial": "SI", "coronal": "AP", "sagittal": "LR"}[plane]
    axis = _axis_for(codes, through)
    count = int(image.shape[axis])
    index = min(max(int(index), 0), count - 1)
    data = np.asanyarray(image.dataobj[tuple(slice(None) if part != axis else index for part in range(3))], dtype=np.float32)
    row_axis, col_axis = [part for part in range(3) if part != axis]
    data = _display_plane(data, codes[row_axis], codes[col_axis], plane)
    data = _match_spacing(data, zooms[row_axis], zooms[col_axis])
    return _png(_window(data, center, width)), count


TRUSTED_HU_SOURCES = {"dicom_conversion_metadata", "verified_source_series", "vista_calibration_record"}
# Keep 3D payloads small enough for browser volume ray casting.
MAX_VOLUME_VOXELS = 8_000_000
QUALITY_VOXELS = {"preview": 900_000, "balanced": 2_500_000, "high": MAX_VOLUME_VOXELS}
_VOLUME_CACHE: dict[tuple, tuple[bytes, dict]] = {}
_VOLUME_CACHE_LIMIT = 8


def _volume_cache_get(key: tuple) -> tuple[bytes, dict] | None:
    hit = _VOLUME_CACHE.get(key)
    if hit is None:
        return None
    # Simple LRU bump.
    _VOLUME_CACHE.pop(key, None)
    _VOLUME_CACHE[key] = hit
    return hit


def _volume_cache_put(key: tuple, value: tuple[bytes, dict]) -> None:
    _VOLUME_CACHE.pop(key, None)
    _VOLUME_CACHE[key] = value
    while len(_VOLUME_CACHE) > _VOLUME_CACHE_LIMIT:
        oldest = next(iter(_VOLUME_CACHE))
        _VOLUME_CACHE.pop(oldest, None)


def nifti_calibration(metadata: dict | None) -> dict:
    """HU is verified only from stored provenance, never from the intensity range or filename."""
    metadata = metadata or {}
    modality = str(metadata.get("modality") or "").upper()
    unit = str(metadata.get("intensity_unit") or "").upper()
    source = str(metadata.get("calibration_source") or "")
    if modality == "CT" and unit == "HU" and source in TRUSTED_HU_SOURCES:
        return {"modality": "CT", "intensity_unit": "HU", "calibration_status": "verified", "calibration_source": source, "hu_threshold_supported": True}
    if modality in {"MR", "MRI", "PT"}:
        return {"modality": modality, "intensity_unit": "native", "calibration_status": "not_applicable", "calibration_source": None, "hu_threshold_supported": False}
    return {"modality": modality or None, "intensity_unit": unit or "unknown", "calibration_status": "unverified", "calibration_source": source or None, "hu_threshold_supported": False}


def _nifti_display_values(path, plane: str, index: int):
    """Scaled NIfTI intensities. NiBabel's dataobj already applies scl_slope and scl_inter."""
    import nibabel as nib
    image = nib.load(str(path))
    codes, zooms = _nifti_geometry(image)
    plane = plane if plane in {"axial", "coronal", "sagittal"} else "axial"
    through = {"axial": "SI", "coronal": "AP", "sagittal": "LR"}[plane]
    axis = _axis_for(codes, through)
    count = int(image.shape[axis])
    index = min(max(int(index), 0), count - 1)
    data = np.asanyarray(image.dataobj[tuple(slice(None) if part != axis else index for part in range(3))], dtype=np.float32)
    row_axis, col_axis = [part for part in range(3) if part != axis]
    data = _display_plane(data, codes[row_axis], codes[col_axis], plane)
    data = _match_spacing(data, zooms[row_axis], zooms[col_axis])
    return data, count


def nifti_threshold_png(path, plane: str, index: int, low: float, high: float, color: str = "red") -> tuple[bytes, dict]:
    if low > high:
        raise ImageValidationError("Maximum value must be greater than or equal to the minimum")
    values, _count = _nifti_display_values(path, plane, index)
    eligible = np.isfinite(values)
    mask = eligible & (values >= low) & (values <= high)
    rgb = COLORS.get(color, COLORS["red"])
    image = np.zeros((*mask.shape, 4), dtype=np.uint8)
    image[mask] = (*rgb, 255)
    buffer = BytesIO()
    Image.fromarray(image, mode="RGBA").save(buffer, format="PNG")
    eligible_count = int(eligible.sum())
    matching = int(mask.sum())
    return buffer.getvalue(), {
        "matching_pixels": matching,
        "eligible_pixels": eligible_count,
        "percentage": round(100 * matching / eligible_count, 2) if eligible_count else 0,
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "denominator": "image pixels on this slice",
    }


def _resample_axis(volume: np.ndarray, axis: int, new_size: int) -> np.ndarray:
    import cv2
    if volume.shape[axis] == new_size:
        return volume
    moved = np.moveaxis(volume, axis, 0)
    flat = np.ascontiguousarray(moved.reshape(moved.shape[0], -1))
    resized = cv2.resize(flat, (flat.shape[1], new_size), interpolation=cv2.INTER_LINEAR)
    return np.moveaxis(resized.reshape((new_size,) + moved.shape[1:]), 0, axis)


def _downsample(volume: np.ndarray, spacings: tuple[float, float, float], quality: str = "high"):
    limit = QUALITY_VOXELS.get(quality, QUALITY_VOXELS["balanced"])
    voxels = int(np.prod(volume.shape))
    if voxels <= limit:
        return np.ascontiguousarray(volume), spacings, False
    factor = (limit / voxels) ** (1 / 3)
    target = tuple(max(1, int(round(size * factor))) for size in volume.shape)
    reduced = volume
    for axis, size in enumerate(target):
        reduced = _resample_axis(reduced, axis, size)
    spacing = tuple(old * (src / dst) for old, src, dst in zip(spacings, volume.shape, reduced.shape))
    return np.ascontiguousarray(reduced), spacing, True


def nifti_volume_bytes(path, quality: str = "preview") -> tuple[bytes, dict]:
    from pathlib import Path
    import nibabel as nib
    path = Path(path)
    chosen = quality if quality in QUALITY_VOXELS else "preview"
    cache_key = ("nifti", str(path.resolve()), path.stat().st_mtime_ns, chosen)
    cached = _volume_cache_get(cache_key)
    if cached is not None:
        return cached
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj, dtype=np.float32)
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    values, zooms, preview = _downsample(values, zooms, chosen)
    payload = (
        values.tobytes(order="F"),
        {
            "dimensions": list(values.shape),
            "spacing": list(zooms),
            "preview": preview,
            "quality": chosen,
            "order": "source-axes",
            "dtype": "float32",
        },
    )
    _volume_cache_put(cache_key, payload)
    return payload


def dicom_volume_bytes(instances, quality: str = "preview") -> tuple[bytes, dict]:
    import pydicom
    if len(instances) < 8:
        raise ImageValidationError("A single image or short series is not reconstructed as a 3D volume")
    chosen = quality if quality in QUALITY_VOXELS else "preview"
    stamp = tuple((str(path), path.stat().st_mtime_ns) for path in instances)
    cache_key = ("dicom", stamp, chosen)
    cached = _volume_cache_get(cache_key)
    if cached is not None:
        return cached
    frames = []
    spacing = None
    for path in instances:
        dataset = pydicom.dcmread(str(path))
        frames.append(calibrated_ct_pixels(dataset) if str(getattr(dataset, "Modality", "")).upper() == "CT" else dataset.pixel_array.astype(np.float32))
        if spacing is None:
            raw = getattr(dataset, "PixelSpacing", None)
            spacing = (float(raw[0]), float(raw[1])) if raw is not None and len(raw) == 2 else (1.0, 1.0)
    stack = np.stack(frames, axis=2)
    slice_spacing = 1.0
    values, zooms, preview = _downsample(stack.astype(np.float32), (spacing[1], spacing[0], slice_spacing), chosen)
    payload = (
        values.tobytes(order="F"),
        {
            "dimensions": list(values.shape),
            "spacing": list(zooms),
            "preview": preview,
            "quality": chosen,
            "order": "source-axes",
            "dtype": "float32",
            "spacing_note": "in-plane spacing from PixelSpacing; slice spacing is 1 mm unless positions were used only for ordering",
        },
    )
    _volume_cache_put(cache_key, payload)
    return payload
