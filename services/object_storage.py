"""Upload case and slide files to iDrive e2 under a patient-CNIC prefix."""

import logging
import re
import threading

from fastapi import HTTPException

from core.config import (
    IDRIVE_E2_ACCESS_KEY_ID,
    IDRIVE_E2_BUCKET,
    IDRIVE_E2_ENABLED,
    IDRIVE_E2_ENDPOINT,
    IDRIVE_E2_REGION,
    IDRIVE_E2_SECRET_ACCESS_KEY,
)

UNLINKED_PREFIX = "unlinked"
logger = logging.getLogger("pathology.storage")


def patient_folder(cnic: str | None, patient_name: str | None = None) -> str:
    digits = re.sub(r"\D", "", cnic or "")
    safe_name = re.sub(r"[\\/:*?\"<>|]+", " ", patient_name or "")
    safe_name = re.sub(r"\s+", " ", safe_name).strip()[:80]
    parts = []
    if digits:
        parts.append(digits)
    if safe_name:
        parts.append(safe_name)
    if not parts:
        return UNLINKED_PREFIX
    return " ".join(parts)


def cnic_folder(cnic: str | None) -> str:
    return patient_folder(cnic)


def object_key(cnic: str | None, filename: str, patient_name: str | None = None) -> str:
    safe_name = filename.replace("\\", "/").split("/")[-1]
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(400, "Invalid stored filename")
    return f"{patient_folder(cnic, patient_name)}/{safe_name}"


def _client():
    if not IDRIVE_E2_ENABLED:
        return None
    if not IDRIVE_E2_ACCESS_KEY_ID or not IDRIVE_E2_SECRET_ACCESS_KEY:
        raise HTTPException(
            503,
            "iDrive e2 is enabled but IDRIVE_E2_ACCESS_KEY_ID and IDRIVE_E2_SECRET_ACCESS_KEY are empty",
        )
    try:
        import boto3
        from botocore.client import Config
    except ModuleNotFoundError as exc:
        raise HTTPException(503, "iDrive e2 is enabled but boto3 is not installed") from exc

    return boto3.client(
        "s3",
        endpoint_url=IDRIVE_E2_ENDPOINT,
        aws_access_key_id=IDRIVE_E2_ACCESS_KEY_ID,
        aws_secret_access_key=IDRIVE_E2_SECRET_ACCESS_KEY,
        region_name=IDRIVE_E2_REGION,
        config=Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 2, "mode": "standard"},
            s3={"addressing_style": "virtual"},
        ),
    )


def _upload_case_file(local_path, cnic, filename, patient_name):
    client = _client()
    if client is None:
        return None
    key = object_key(cnic, filename, patient_name)
    client.upload_file(str(local_path), IDRIVE_E2_BUCKET, key)
    legacy_key = object_key(None, filename)
    if legacy_key != key:
        client.delete_object(Bucket=IDRIVE_E2_BUCKET, Key=legacy_key)
    return key


def upload_case_file(local_path, cnic: str | None, filename: str, patient_name: str | None = None) -> str | None:
    """Copy a validated slide into bucket/{CNIC patient name}/.

    The copy runs after the local file is already saved. A slow or failed
    remote copy must not keep the upload request open or delete the local slide.
    """
    if not IDRIVE_E2_ENABLED:
        return None

    def send():
        try:
            _upload_case_file(local_path, cnic, filename, patient_name)
        except Exception:
            logger.exception("iDrive e2 copy failed for %s; the local slide was kept", filename)

    threading.Thread(target=send, name="idrive-e2-upload", daemon=True).start()
    return None


def relocate_case_file(filename: str, cnic: str | None, patient_name: str | None = None) -> str | None:
    """Move a slide that was stored before the patient was known into the patient folder."""
    client = _client()
    if client is None or not filename:
        return None
    destination = object_key(cnic, filename, patient_name)
    source_name = filename.replace("\\", "/").split("/")[-1]
    source = f"{UNLINKED_PREFIX}/{source_name}"
    if source == destination:
        return destination
    try:
        client.copy_object(
            Bucket=IDRIVE_E2_BUCKET,
            CopySource={"Bucket": IDRIVE_E2_BUCKET, "Key": source},
            Key=destination,
        )
        client.delete_object(Bucket=IDRIVE_E2_BUCKET, Key=source)
    except Exception:
        return None
    return destination


def delete_case_file(cnic: str | None, filename: str, patient_name: str | None = None) -> None:
    client = _client()
    if client is None or not filename:
        return
    keys = {object_key(cnic, filename, patient_name), object_key(None, filename)}
    for key in keys:
        try:
            client.delete_object(Bucket=IDRIVE_E2_BUCKET, Key=key)
        except Exception:
            continue
