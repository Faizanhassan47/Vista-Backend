"""Preprocessing for NeuroMRI. JPG/PNG matches the training eval transform.

MRI DICOM and NIfTI slices use a separate percentile adapter. That adapter is
not the training preprocessing and is recorded on every medical-image result.
"""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MRI_INTENSITY_METHOD = "percentile-1-99-monochrome-v1"

CLASS_NAMES = [
    "AD Mild",
    "AD Moderate",
    "AD Very Mild",
    "BT Glioma",
    "BT Meningioma",
    "BT Pituitary",
    "Brain Stroke",
    "Multiple Sclerosis",
    "Normal",
    "Parkinson Disease",
]


class NeuromriInputError(ValueError):
    def __init__(self, message, category="invalid_image"):
        super().__init__(message)
        self.category = category


def pad_to_square(image):
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    right = side - width - left
    bottom = side - height - top
    return ImageOps.expand(image, border=(left, top, right, bottom), fill=0)


def eval_tensor(image):
    """Same steps as NeuroMRI training.transforms.get_eval_transforms."""
    import torch
    from torchvision.transforms import functional as F

    image = image.convert("RGB")
    image = pad_to_square(image)
    image = F.resize(image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=F.InterpolationMode.BILINEAR)
    tensor = F.to_tensor(image)
    tensor = F.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0)


def tensor_from_path(path):
    with Image.open(path) as image:
        return eval_tensor(image)


def mri_array_to_image(values):
    """Map finite MRI intensities to an 8-bit RGB image. Not HU and not the JPG pipeline."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 4:
        raise NeuromriInputError("The selected MRI slice is not a usable 2D image")
    finite = array[np.isfinite(array)]
    if finite.size < 16:
        raise NeuromriInputError("The selected MRI slice has no usable intensities")
    low, high = np.percentile(finite, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        gray = np.full(array.shape, 128, dtype=np.uint8)
    else:
        scaled = np.clip((array - low) / (high - low), 0, 1)
        scaled = np.where(np.isfinite(array), scaled, 0)
        gray = np.round(scaled * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb, mode="RGB"), {
        "intensity_conversion": MRI_INTENSITY_METHOD,
        "percentile_low": None if not np.isfinite(low) else float(low),
        "percentile_high": None if not np.isfinite(high) else float(high),
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
    }


def dicom_mr_slice(path, frame_index=0):
    import pydicom

    from services.image_validation import ImageValidationError
    from services.radiology_frames import dicom_intensity_pixels

    dataset = pydicom.dcmread(str(path), force=False)
    modality = str(getattr(dataset, "Modality", "") or "").upper()
    if modality != "MR":
        raise NeuromriInputError(
            "NeuroMRI accepts MR DICOM only",
            "unsupported_modality",
        )
    photometric = str(getattr(dataset, "PhotometricInterpretation", "") or "").upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise NeuromriInputError(
            "NeuroMRI accepts monochrome MR DICOM only",
            "unsupported_dicom",
        )
    frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= frames:
        raise NeuromriInputError(
            "The selected frame is outside this DICOM instance",
            "invalid_slice",
        )
    try:
        values = dicom_intensity_pixels(dataset, frame_index)
    except ImageValidationError as exc:
        raise NeuromriInputError(str(exc), "decode_failed") from exc
    if photometric == "MONOCHROME1":
        finite = values[np.isfinite(values)]
        if finite.size:
            values = np.where(np.isfinite(values), float(finite.max()) + float(finite.min()) - values, values)
    image, info = mri_array_to_image(values)
    info.update({
        "source_format": "DICOM",
        "modality": "MR",
        "frame_index": frame_index,
        "photometric": photometric,
    })
    return image, info


def apply_mri_verification(metadata, username, verified_at):
    """Record an authorized MRI confirmation. Does not read pixels or filenames."""
    updated = dict(metadata or {})
    current = str(updated.get("modality") or "").upper()
    if current == "CT":
        raise NeuromriInputError("A CT volume cannot be verified as MRI", "unsupported_modality")
    if current in {"MR", "MRI"}:
        return updated
    updated["modality"] = "MR"
    updated["modality_verification"] = {
        "status": "verified",
        "modality": "MR",
        "source": "authorized_user",
        "verified_by": username,
        "verified_at": verified_at,
        "note": "Authorized user confirmed this NIfTI study is MRI. Not inferred from the filename or intensities.",
    }
    return updated


def nifti_mr_slice(path, plane, index, metadata):
    from services.radiology_frames import _nifti_display_values, nifti_calibration

    calibration = nifti_calibration(metadata)
    modality = str(calibration.get("modality") or "").upper()
    if modality == "CT":
        raise NeuromriInputError("NeuroMRI does not classify CT volumes", "unsupported_modality")
    if modality not in {"MR", "MRI"}:
        raise NeuromriInputError(
            "This NIfTI volume has no verified MRI modality, so NeuroMRI will not classify it",
            "unverified_modality",
        )
    plane = str(plane or "axial").lower()
    if plane not in {"axial", "coronal", "sagittal"}:
        raise NeuromriInputError("Plane must be axial, coronal, or sagittal", "invalid_slice")
    from services.radiology_frames import nifti_plane_counts
    counts = nifti_plane_counts(path)
    count = int(counts.get(plane) or 0)
    index = int(index)
    if index < 0 or index >= count:
        raise NeuromriInputError("The selected slice is outside this volume", "invalid_slice")
    values, _count = _nifti_display_values(path, plane, index)
    image, info = mri_array_to_image(values)
    info.update({
        "source_format": "NIfTI",
        "modality": modality,
        "plane": plane,
        "slice_index": index,
    })
    return image, info


def image_from_upload(payload, content_type):
    if len(payload) > 20 * 1024 * 1024:
        raise NeuromriInputError("The image is larger than 20 MB", "invalid_image")
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            if image.width * image.height > 16_000_000:
                raise NeuromriInputError("The image is too large to classify", "invalid_image")
            return image.convert("RGB")
    except NeuromriInputError:
        raise
    except Exception as exc:
        raise NeuromriInputError("The file is not a readable JPG or PNG", "invalid_image") from exc
