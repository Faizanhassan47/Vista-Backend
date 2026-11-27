"""TotalSegmentator configuration, geometry, and measurements.

Inference runs in the separate tested environment. This module does not
download weights and does not change NeuroMRI or pathology models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FEATURE_CODE = "TOTALSEGMENTATOR"
MODEL_NAME = "TotalSegmentator"
MODEL_VERSION = "2.18.0"
TASK_FOR_MODALITY = {"CT": "total", "MR": "total_mr", "MRI": "total_mr"}
_CATALOG_PATH = Path(__file__).with_name("totalsegmentator_catalog.json")
_FIXED_COLORS = {
    "liver": [176, 48, 48],
    "spleen": [150, 48, 140],
    "kidney_left": [42, 92, 196],
    "kidney_right": [36, 150, 96],
}


def _load_catalog():
    raw = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    by_task = {}
    for task, items in raw.items():
        by_task[task] = {item["id"]: item for item in items}
    return by_task


CATALOG = _load_catalog()


def _hsv_color(name):
    hue = (sum((index + 1) * ord(char) for index, char in enumerate(name)) * 37) % 360
    saturation, value = 0.62, 0.86
    sector = hue / 60.0
    chroma = value * saturation
    mix = chroma * (1 - abs((sector % 2) - 1))
    base = value - chroma
    red, green, blue = {
        0: (chroma, mix, 0),
        1: (mix, chroma, 0),
        2: (0, chroma, mix),
        3: (0, mix, chroma),
        4: (mix, 0, chroma),
    }.get(int(sector), (chroma, 0, mix))
    return [int(round((channel + base) * 255)) for channel in (red, green, blue)]


def color_for(roi):
    return list(_FIXED_COLORS.get(roi) or _hsv_color(roi))


def label_for(roi):
    for task in ("total", "total_mr"):
        item = CATALOG[task].get(roi)
        if item:
            return item["label"]
    return roi.replace("_", " ").title()


def class_ids(task):
    if task not in CATALOG:
        raise TotalSegmentatorError("Unsupported TotalSegmentator task", "unsupported_task")
    return list(CATALOG[task])


def structures_for_modality(modality):
    task = task_for_modality(modality)
    items = []
    for roi, item in CATALOG[task].items():
        items.append({
            "id": roi,
            "label": item["label"],
            "category": item["category"],
            "modality": item["modality"],
            "task": task,
            "supported": True,
            "color": color_for(roi),
        })
    categories = []
    for item in items:
        if item["category"] not in categories:
            categories.append(item["category"])
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "modality": "CT" if task == "total" else "MR",
        "task": task,
        "device": "cpu",
        "fast": True,
        "count": len(items),
        "categories": categories,
        "structures": items,
        "research_use_only": True,
    }


ROIS = {
    roi: {"label": label_for(roi), "color": color_for(roi)}
    for roi in list(CATALOG["total"]) + [name for name in CATALOG["total_mr"] if name not in CATALOG["total"]]
}


class TotalSegmentatorError(ValueError):
    def __init__(self, message, category="invalid_request"):
        super().__init__(message)
        self.category = category


def _env(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.is_file():
        return default
    prefix = name + "="
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return default


def project_root():
    return Path(__file__).resolve().parents[2]


def weights_dir():
    """nnUNet results directory. Relative config is resolved from the project root."""
    configured = _env("TOTALSEG_WEIGHTS_PATH").strip()
    if not configured:
        path = project_root() / "Ai models" / "totalsegmentator" / "models"
    else:
        path = Path(configured)
        if not path.is_absolute():
            path = project_root() / path
    return path.resolve()


REQUIRED_WEIGHTS = (
    "Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
    "Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/plans.json",
    "Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/dataset.json",
    "Dataset298_TotalSegmentator_total_6mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
    "Dataset298_TotalSegmentator_total_6mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/plans.json",
    "Dataset298_TotalSegmentator_total_6mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/dataset.json",
    "Dataset852_TotalSegMRI_total_3mm_1088subj/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
    "Dataset852_TotalSegMRI_total_3mm_1088subj/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres/plans.json",
    "Dataset852_TotalSegMRI_total_3mm_1088subj/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres/dataset.json",
)


def missing_weights():
    root = weights_dir()
    return [item for item in REQUIRED_WEIGHTS if not (root / item).is_file() or (root / item).stat().st_size < 1024]


def settings():
    root = Path(__file__).resolve().parents[1]
    work = _env("TOTALSEG_WORK_DIR") or str(root / "storage" / "totalsegmentator")
    return {
        "python": _env("TOTALSEG_PYTHON").strip(),
        "work_dir": work,
        "weights_dir": str(weights_dir()),
        "threads": max(1, int(_env("TOTALSEG_THREADS", "4") or 4)),
        # CPU TotalSegmentator (especially total_mr) routinely needs >15 minutes.
        "timeout_seconds": max(60, int(_env("TOTALSEG_TIMEOUT_SECONDS", "3600") or 3600)),
        "fast": _env("TOTALSEG_FAST", "1") != "0",
        "model_version": MODEL_VERSION,
    }


def worker_env(base=None):
    """Environment for the TotalSegmentator process. Checkpoints come from the project tree."""
    env = dict(os.environ if base is None else base)
    config = settings()
    weights = config["weights_dir"]
    work = Path(config["work_dir"])
    home = str(work / "runtime-home")
    mpl_config = work / "mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    Path(home).mkdir(parents=True, exist_ok=True)
    env["TOTALSEG_WEIGHTS_PATH"] = weights
    env["TOTALSEG_HOME_DIR"] = home
    env["nnUNet_results"] = weights
    env["nnUNet_raw"] = weights
    env["nnUNet_preprocessed"] = weights
    env["OMP_NUM_THREADS"] = str(config["threads"])
    # Avoid first-run matplotlib font-cache builds eating the whole job timeout.
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(mpl_config)
    env["HOME"] = home
    return env


def apply_modality_verification(metadata, modality, username, verified_at):
    """Record an authorized CT or MRI confirmation. Does not read pixels or filenames."""
    chosen = str(modality or "").upper()
    if chosen not in {"CT", "MR"}:
        raise TotalSegmentatorError("Verification must be CT or MR", "unsupported_modality")
    updated = dict(metadata or {})
    current = str(updated.get("modality") or "").upper()
    if current in {"MRI"}:
        current = "MR"
    if current in {"CT", "MR"} and current != chosen:
        raise TotalSegmentatorError("This study is already verified as a different modality", "unsupported_modality")
    if current == chosen:
        return updated
    updated["modality"] = chosen
    updated["modality_verification"] = {
        "status": "verified",
        "modality": chosen,
        "source": "authorized_user",
        "verified_by": username,
        "verified_at": verified_at,
        "note": "Authorized user confirmed this NIfTI modality. Not inferred from the filename or intensities.",
    }
    return updated


def task_for_modality(modality):
    task = TASK_FOR_MODALITY.get(str(modality or "").upper())
    if not task:
        raise TotalSegmentatorError(
            "TotalSegmentator needs a verified CT or MRI study. The file format alone is not enough.",
            "unsupported_modality",
        )
    return task


def normalize_rois(requested, task="total"):
    allowed = class_ids(task)
    names = []
    for item in requested or []:
        name = str(item).strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise TotalSegmentatorError("Select at least one anatomical structure", "empty_selection")
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise TotalSegmentatorError("Unsupported structure: " + ", ".join(unknown), "unsupported_roi")
    return names


def command(python, source, output_dir, task, rois, fast=True):
    if not python:
        raise TotalSegmentatorError("TotalSegmentator Python is not configured", "not_configured")
    missing = missing_weights()
    if missing:
        raise TotalSegmentatorError(
            "Project TotalSegmentator weights are incomplete: " + ", ".join(missing[:3]),
            "missing_weights",
        )
    argv = [
        python, "-m", "totalsegmentator.bin.TotalSegmentator",
        "-i", str(source),
        "-o", str(output_dir),
        "-ta", task,
        "-d", "cpu",
        "-nr", str(settings()["threads"]),
    ]
    if set(rois) != set(class_ids(task)):
        argv.extend(["-rs", *rois])
    if fast:
        argv.append("-f")
    return argv


def voxel_volume_ml(affine):
    matrix = np.asarray(affine, dtype=np.float64)[:3, :3]
    return abs(float(np.linalg.det(matrix))) / 1000.0


def touches_boundary(mask):
    array = np.asarray(mask) > 0
    if not array.any():
        return False
    faces = [array[0], array[-1], array[:, 0], array[:, -1], array[:, :, 0], array[:, :, -1]]
    return any(bool(np.any(face)) for face in faces)


def measure_mask(image, mask, roi, report_hu):
    data = np.asarray(image, dtype=np.float32)
    labels = np.asarray(mask) > 0
    if data.shape != labels.shape:
        raise TotalSegmentatorError("Segmentation mask does not match the source volume", "misaligned_mask")
    count = int(labels.sum())
    ml = count * voxel_volume_ml(np.eye(4) if getattr(image, "affine", None) is None else image.affine)
    truncated = touches_boundary(labels)
    result = _measurement(roi, count, ml, truncated, labels, image.affine if getattr(image, "affine", None) is not None else np.eye(4))
    if report_hu:
        if count == 0:
            result.update({"hu_mean": None, "hu_min": None, "hu_max": None, "hu_std": None})
        else:
            values = data[labels]
            values = values[np.isfinite(values)]
            result.update(_hu(values))
    return result


def measure_nifti(image_path, mask_path, roi, report_hu):
    import nibabel as nib

    image = nib.load(str(image_path))
    mask = nib.load(str(mask_path))
    if not np.allclose(np.asarray(image.affine), np.asarray(mask.affine), atol=1e-2):
        raise TotalSegmentatorError("Segmentation mask affine does not match the source", "misaligned_mask")
    if tuple(image.shape) != tuple(mask.shape):
        raise TotalSegmentatorError("Segmentation mask shape does not match the source", "misaligned_mask")
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    labels = np.asanyarray(mask.dataobj)
    count = int((labels > 0).sum())
    ml = count * voxel_volume_ml(image.affine)
    truncated = touches_boundary(labels > 0)
    result = _measurement(roi, count, ml, truncated, labels > 0, image.affine)
    if report_hu and count:
        values = data[labels > 0]
        values = values[np.isfinite(values)]
        result.update(_hu(values))
    elif report_hu:
        result.update({"hu_mean": None, "hu_min": None, "hu_max": None, "hu_std": None})
    return result


def empty_measurement(roi, report_hu):
    result = _measurement(roi, 0, 0.0, False, None, np.eye(4))
    if report_hu:
        result.update({"hu_mean": None, "hu_min": None, "hu_max": None, "hu_std": None})
    return result


def _hu(values):
    if values.size == 0:
        return {"hu_mean": None, "hu_min": None, "hu_max": None, "hu_std": None}
    return {
        "hu_mean": round(float(values.mean()), 2),
        "hu_min": round(float(values.min()), 2),
        "hu_max": round(float(values.max()), 2),
        "hu_std": round(float(values.std()), 2),
    }


def _measurement(roi, count, ml, truncated, labels, affine):
    bbox = None
    extent_mm = None
    if labels is not None and count:
        coords = np.argwhere(np.asarray(labels) > 0)
        low = coords.min(axis=0)
        high = coords.max(axis=0)
        voxels = (high - low + 1).astype(np.float64)
        spacing = np.sqrt((np.asarray(affine, dtype=np.float64)[:3, :3] ** 2).sum(axis=0))
        bbox = {"min": [int(v) for v in low], "max": [int(v) for v in high], "size_voxels": [int(v) for v in voxels]}
        extent_mm = [round(float(v), 2) for v in voxels * spacing]
    warning = (
        "Potentially truncated — segmentation reaches scan boundary; reported volume may represent only the visible portion."
        if truncated and count > 0 else None
    )
    return {
        "roi": roi,
        "label": label_for(roi),
        "color": color_for(roi),
        "voxel_count": count,
        "volume_ml": round(ml, 2),
        "bounding_box": bbox,
        "extent_mm": extent_mm,
        "empty": count == 0,
        "potentially_truncated": bool(truncated and count > 0),
        "coverage_warning": warning,
    }


def init_totalsegmentator_catalog(connect_fn):
    from contextlib import closing
    with closing(connect_fn()) as db:
        db.execute(
            """IF NOT EXISTS (SELECT 1 FROM ai_features WHERE feature_code = ?)
            INSERT INTO ai_features (
                feature_code, feature_name, description, is_implemented, is_active,
                is_catalog_visible, catalog_category, catalog_summary, catalog_purpose,
                catalog_value, catalog_title, catalog_display_order, catalog_outputs_json
            ) VALUES (
                ?, 'TotalSegmentator',
                'Research-only anatomical segmentation for verified CT and MRI volumes.',
                1, 1, 1,
                'Radiology',
                'Segments supported CT and MRI anatomical structures on a verified volume.',
                'CPU anatomical segmentation. Not a diagnosis.',
                'Masks are predicted anatomy, not confirmed clinical findings.',
                'TotalSegmentator',
                6,
                ?
            )""",
            (FEATURE_CODE, FEATURE_CODE, '["Organ masks","Volumes","Research use only"]'),
        )
        db.commit()
