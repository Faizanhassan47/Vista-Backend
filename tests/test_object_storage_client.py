import builtins

import pytest
from fastapi import HTTPException

import services.object_storage as storage


def test_missing_boto3_is_a_structured_error(monkeypatch):
    monkeypatch.setattr(storage, "IDRIVE_E2_ENABLED", True)
    monkeypatch.setattr(storage, "IDRIVE_E2_ACCESS_KEY_ID", "key")
    monkeypatch.setattr(storage, "IDRIVE_E2_SECRET_ACCESS_KEY", "secret")
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "boto3":
            raise ModuleNotFoundError("boto3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(HTTPException) as caught:
        storage.relocate_case_file("instance-0009.dcm", "123", "Patient")
    assert caught.value.status_code == 503
    assert "boto3" in str(caught.value.detail)
