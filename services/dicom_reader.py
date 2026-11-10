"""DICOM VL Whole Slide Microscopy inspection and series staging.

Pixel decoding stays with OpenSlide. This module only identifies a series,
rejects non-pathology DICOM, and copies the original instances unchanged.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from shutil import copy2
from typing import Any
from uuid import uuid4

from services.image_validation import ImageValidationError

WSI_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.77.1.6"
RADIOLOGY_SOP_CLASS_UIDS = {
    "1.2.840.10008.5.1.4.1.1.2": "CT",
    "1.2.840.10008.5.1.4.1.1.4": "MR",
    "1.2.840.10008.5.1.4.1.1.7": "SC",
    "1.2.840.10008.5.1.4.1.1.1": "CR",
}


def is_dicom_file(path: Path) -> bool:
    try:
        with Path(path).open("rb") as handle:
            preamble = handle.read(132)
    except OSError:
        return False
    return len(preamble) >= 132 and preamble[128:132] == b"DICM"


def _pydicom():
    try:
        import pydicom
    except ImportError as exc:
        raise ImageValidationError("DICOM inspection requires pydicom") from exc
    return pydicom


def _text(dataset, name: str) -> str | None:
    value = getattr(dataset, name, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float(dataset, name: str) -> float | None:
    value = getattr(dataset, name, None)
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        try:
            value = value[0]
        except (TypeError, IndexError, KeyError):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _dataset(path: Path):
    pydicom = _pydicom()
    if not is_dicom_file(path):
        raise ImageValidationError(f"{path.name} is not a DICOM file")
    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except Exception as exc:
        raise ImageValidationError(f"{path.name} could not be read as DICOM") from exc
    return dataset


def read_instance(path: Path) -> dict[str, Any]:
    dataset = _dataset(path)
    sop_class = _text(dataset, "SOPClassUID")
    if sop_class != WSI_SOP_CLASS_UID:
        modality = _text(dataset, "Modality") or "unknown"
        raise ImageValidationError(
            f"Unsupported DICOM SOP Class {sop_class or 'missing'} (modality {modality}). "
            "Only VL Whole Slide Microscopy Image is accepted."
        )
    sop_instance = _text(dataset, "SOPInstanceUID")
    series_uid = _text(dataset, "SeriesInstanceUID")
    study_uid = _text(dataset, "StudyInstanceUID")
    if not sop_instance or not series_uid or not study_uid:
        raise ImageValidationError(f"{path.name} is missing Study, Series, or SOP Instance UID")
    rows = int(getattr(dataset, "TotalPixelMatrixRows", 0) or getattr(dataset, "Rows", 0) or 0)
    columns = int(getattr(dataset, "TotalPixelMatrixColumns", 0) or getattr(dataset, "Columns", 0) or 0)
    if rows < 1 or columns < 1:
        raise ImageValidationError(f"{path.name} has no image dimensions")
    return {
        "path": str(path),
        "sop_class_uid": sop_class,
        "sop_instance_uid": sop_instance,
        "series_instance_uid": series_uid,
        "study_instance_uid": study_uid,
        "transfer_syntax_uid": _text(dataset.file_meta, "TransferSyntaxUID") if getattr(dataset, "file_meta", None) else None,
        "specimen_id": _text(dataset, "SpecimenIdentifier") or _text(dataset, "ContainerIdentifier"),
        "container_id": _text(dataset, "ContainerIdentifier"),
        "manufacturer": _text(dataset, "Manufacturer"),
        "manufacturer_model": _text(dataset, "ManufacturerModelName"),
        "rows": rows,
        "columns": columns,
        "number_of_frames": int(getattr(dataset, "NumberOfFrames", 1) or 1),
        "pixel_spacing_x": _float(dataset, "ImagedVolumeWidth"),
        "pixel_spacing_row": (_float(dataset, "PixelSpacing") or 0) * 1000 or None,
        "optical_path_id": _text(dataset, "OpticalPathIdentifier"),
        "instance_number": int(getattr(dataset, "InstanceNumber", 0) or 0),
        "file_size": path.stat().st_size,
    }


def _position(dataset):
    value = getattr(dataset, "ImagePositionPatient", None)
    if value is None:
        return None
    try:
        return tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return None


def _radiology_series(anchor: Path) -> dict[str, Any]:
    """Group CT/MR and other single-frame radiology instances that share one series."""
    anchor = Path(anchor)
    files = sorted(path for path in (anchor.iterdir() if anchor.is_dir() else anchor.parent.iterdir()) if path.is_file() and is_dicom_file(path))
    if anchor.is_file() and anchor not in files:
        files.append(anchor)
    if not files:
        raise ImageValidationError("Directory contains no DICOM files")
    seed = _dataset(anchor if anchor.is_file() else files[0])
    sop_class = _text(seed, "SOPClassUID")
    if sop_class not in RADIOLOGY_SOP_CLASS_UIDS:
        modality = _text(seed, "Modality") or "unknown"
        raise ImageValidationError(
            f"Unsupported DICOM SOP Class {sop_class or 'missing'} (modality {modality}). "
            "Accepted inputs are pathology WSI, CT, MR, CR, and secondary capture."
        )
    series_uid = _text(seed, "SeriesInstanceUID")
    study_uid = _text(seed, "StudyInstanceUID")
    if not series_uid or not study_uid:
        raise ImageValidationError("DICOM series is missing Study or Series UID")
    instances = []
    seen = set()
    for path in files:
        dataset = _dataset(path)
        if _text(dataset, "SeriesInstanceUID") != series_uid:
            continue
        if _text(dataset, "SOPClassUID") != sop_class:
            continue
        sop_instance = _text(dataset, "SOPInstanceUID")
        if not sop_instance:
            raise ImageValidationError(f"{path.name} is missing a SOP Instance UID")
        if sop_instance in seen:
            raise ImageValidationError(f"Duplicate SOP Instance UID {sop_instance}")
        seen.add(sop_instance)
        rows = int(getattr(dataset, "Rows", 0) or 0)
        columns = int(getattr(dataset, "Columns", 0) or 0)
        if rows < 1 or columns < 1:
            raise ImageValidationError(f"{path.name} has no image dimensions")
        instances.append({
            "path": str(path),
            "sop_instance_uid": sop_instance,
            "instance_number": int(getattr(dataset, "InstanceNumber", 0) or 0),
            "position": _position(dataset),
            "rows": rows,
            "columns": columns,
            "file_size": path.stat().st_size,
            "transfer_syntax_uid": _text(dataset.file_meta, "TransferSyntaxUID") if getattr(dataset, "file_meta", None) else None,
        })
    if not instances:
        raise ImageValidationError("DICOM series contains no readable instances")
    if all(item["position"] for item in instances):
        normal = (0.0, 0.0, 1.0)
        orientation = getattr(seed, "ImageOrientationPatient", None)
        if orientation and len(orientation) == 6:
            row = tuple(float(part) for part in orientation[:3])
            column = tuple(float(part) for part in orientation[3:])
            normal = (
                row[1] * column[2] - row[2] * column[1],
                row[2] * column[0] - row[0] * column[2],
                row[0] * column[1] - row[1] * column[0],
            )
        instances.sort(key=lambda item: sum(a * b for a, b in zip(item["position"], normal)))
    else:
        instances.sort(key=lambda item: item["instance_number"])
    modality = _text(seed, "Modality") or RADIOLOGY_SOP_CLASS_UIDS[sop_class]
    spacing = _float(seed, "PixelSpacing")
    digest = sha256()
    total_size = 0
    for instance in instances:
        digest.update(instance["sop_instance_uid"].encode("utf-8"))
        total_size += instance["file_size"]
    return {
        "file_type": "dicom-stack",
        "format": f"{modality} DICOM",
        "width": instances[0]["columns"],
        "height": instances[0]["rows"],
        "level_count": len(instances),
        "study_instance_uid": study_uid,
        "series_instance_uid": series_uid,
        "sop_class_uid": sop_class,
        "anchor": instances[0]["path"],
        "instances": instances,
        "file_size": total_size,
        "checksum_sha256": digest.hexdigest(),
        "normalized_metadata": {"slide_label": _text(seed, "SeriesDescription") or modality, "barcode": None, "scanner_vendor": _text(seed, "Manufacturer"), "scanner_model": _text(seed, "ManufacturerModelName"), "magnification": None, "acquisition_time": _text(seed, "StudyDate")},
        "physical_calibration": {"available": bool(spacing), "pixel_size_microns_x": (spacing or 0) * 1000 or None, "pixel_size_microns_y": (spacing or 0) * 1000 or None, "source": "PixelSpacing" if spacing else None},
        "renderer_type": "dicom-stack",
        "modality": modality,
    }


def discover_series(anchor: Path) -> dict[str, Any]:
    """Read the WSI series that contains anchor. Sibling files of another series are ignored."""
    anchor = Path(anchor)
    if anchor.is_dir():
        candidates = sorted(path for path in anchor.iterdir() if path.is_file() and is_dicom_file(path))
        if not candidates:
            raise ImageValidationError("Directory contains no DICOM files")
        anchor = candidates[0]
    if not anchor.is_file():
        raise ImageValidationError("DICOM source does not exist")
    if _text(_dataset(anchor), "SOPClassUID") != WSI_SOP_CLASS_UID:
        return _radiology_series(anchor)

    seed = read_instance(anchor)
    folder = anchor.parent
    instances = []
    seen: set[str] = set()
    for path in sorted(folder.iterdir()):
        if not path.is_file() or not is_dicom_file(path):
            continue
        try:
            instance = read_instance(path)
        except ImageValidationError:
            if path.resolve() == anchor.resolve():
                raise
            continue
        if instance["series_instance_uid"] != seed["series_instance_uid"]:
            continue
        if instance["sop_instance_uid"] in seen:
            raise ImageValidationError(f"Duplicate SOP Instance UID {instance['sop_instance_uid']}")
        seen.add(instance["sop_instance_uid"])
        instances.append(instance)
    if not instances:
        raise ImageValidationError("DICOM series contains no readable WSI instances")

    digest = sha256()
    total_size = 0
    for instance in instances:
        digest.update(instance["sop_instance_uid"].encode("utf-8"))
        digest.update(str(instance["file_size"]).encode("utf-8"))
        total_size += instance["file_size"]
    largest = max(instances, key=lambda item: item["rows"] * item["columns"])
    spacing = largest["pixel_spacing_row"]
    return {
        "file_type": "dicom-wsi",
        "format": "DICOM WSI",
        "width": largest["columns"],
        "height": largest["rows"],
        "level_count": len(instances),
        "study_instance_uid": seed["study_instance_uid"],
        "series_instance_uid": seed["series_instance_uid"],
        "sop_class_uid": WSI_SOP_CLASS_UID,
        "anchor": largest["path"],
        "instances": instances,
        "file_size": total_size,
        "checksum_sha256": digest.hexdigest(),
        "normalized_metadata": {
            "slide_label": largest["specimen_id"],
            "barcode": largest["container_id"] or largest["specimen_id"],
            "scanner_vendor": largest["manufacturer"],
            "scanner_model": largest["manufacturer_model"],
            "magnification": None,
            "acquisition_time": None,
        },
        "physical_calibration": {
            "available": bool(spacing),
            "pixel_size_microns_x": spacing,
            "pixel_size_microns_y": spacing,
            "source": "PixelSpacing" if spacing else None,
        },
        "dicom": {
            "study_instance_uid": seed["study_instance_uid"],
            "series_instance_uid": seed["series_instance_uid"],
            "sop_instance_uids": [item["sop_instance_uid"] for item in instances],
            "transfer_syntax_uid": largest["transfer_syntax_uid"],
        },
    }


def stage_series(series: dict[str, Any], uploads_dir: Path) -> str:
    """Copy original instances into uploads/<id>/ and return the stored anchor path."""
    folder_name = uuid4().hex
    destination = Path(uploads_dir) / folder_name
    destination.mkdir(parents=True, exist_ok=False)
    anchor_name = None
    anchor_source = str(Path(series["anchor"]).resolve())
    for instance in series["instances"]:
        source = Path(instance["path"])
        target_name = source.name
        copy2(source, destination / target_name)
        if str(source.resolve()) == anchor_source:
            anchor_name = target_name
    if anchor_name is None:
        raise ImageValidationError("DICOM series anchor was not copied")
    return f"{folder_name}/{anchor_name}"
