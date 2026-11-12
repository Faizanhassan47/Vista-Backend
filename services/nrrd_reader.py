"""Read NRRD volumes into an in-memory NIfTI image for the existing viewer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from services.image_validation import ImageValidationError

MAX_NRRD_VOXELS = 200_000_000


def looks_like_nrrd(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".nrrd") or name.endswith(".nhdr"):
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"NRRD"
    except OSError:
        return False


def _reject_detached_paths(path: Path) -> None:
    if path.suffix.lower() != ".nhdr":
        return
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ImageValidationError("NRRD header could not be read") from exc
    for line in lines:
        if not line.lower().startswith("data file:"):
            continue
        reference = line.split(":", 1)[1].strip().strip('"')
        if reference.upper().startswith("LIST"):
            raise ImageValidationError("NRRD data file lists are not supported. Upload a single .nrrd file.")
        candidate = Path(reference)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ImageValidationError("NRRD data file path is not allowed")
        resolved = (path.parent / candidate).resolve()
        if path.parent.resolve() not in resolved.parents and resolved != path.parent.resolve():
            raise ImageValidationError("NRRD data file must sit next to the header")


def open_scalar_volume(path: Path):
    """Return a nibabel image. NIfTI is loaded as before. NRRD is wrapped without rescaling."""
    import nibabel as nib

    path = Path(path)
    if not looks_like_nrrd(path):
        return nib.load(str(path))
    _reject_detached_paths(path)
    try:
        import nrrd
    except ImportError as exc:
        raise ImageValidationError("NRRD support requires pynrrd") from exc
    try:
        data, header = nrrd.read(str(path))
    except Exception as exc:
        raise ImageValidationError(f"{path.name} could not be read as NRRD") from exc
    values = np.asanyarray(data)
    if values.ndim != 3:
        raise ImageValidationError(f"{path.name} has {values.ndim} dimensions. Only 3D scalar NRRD volumes are supported.")
    if int(np.prod(values.shape)) > MAX_NRRD_VOXELS:
        raise ImageValidationError("NRRD volume is too large to load")
    directions = header.get("space directions")
    origin_header = header.get("space origin")
    origin = np.zeros(3, dtype=np.float64) if origin_header is None else np.array(origin_header, dtype=np.float64)
    affine = np.eye(4, dtype=np.float64)
    if directions is not None:
        for index, vector in enumerate(list(directions)[:3]):
            if vector is None:
                continue
            affine[:3, index] = np.array(vector, dtype=np.float64)
    affine[:3, 3] = origin[:3]
    image = nib.Nifti1Image(np.ascontiguousarray(values), affine)
    image.header.set_slope_inter(1.0, 0.0)
    return image
