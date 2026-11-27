"""Unit tests for rate limiting and in-memory job queue helpers."""

from collections import deque

from middleware.rate_limit import RateLimitMiddleware, _bucket_for, _is_exempt
from services.job_queue import register_handler, worker_id


def test_rate_limit_buckets():
    assert _bucket_for("/api/auth/login", "POST") == "auth"
    assert _bucket_for("/api/ai/neuromri/predict", "POST") == "ai"
    assert _bucket_for("/api/cases", "POST") == "write"
    assert _bucket_for("/api/cases", "GET") == "default"


def test_viewer_tile_paths_are_rate_limit_exempt():
    assert _is_exempt("/api/slides/13064/slide_files/12/7_7.jpeg")
    assert _is_exempt("/api/slides/13064/slide.dzi")
    assert _is_exempt("/api/slides/9/radiology/frame")
    assert _is_exempt("/api/slides/9/thumbnail")
    assert _is_exempt("/api/ai/jobs/job-70fe9813e1ca")
    assert _is_exempt("/api/ai/totalsegmentator/jobs/abc")
    assert not _is_exempt("/api/cases")
    assert not _is_exempt("/api/ai/neuromri/predict")


def test_rate_limit_allow_then_block(monkeypatch):
    middleware = RateLimitMiddleware(app=None)
    monkeypatch.setattr("middleware.rate_limit.RATE_LIMIT_ENABLED", True)
    key = "default:test-client"
    limit, window = 3, 60
    middleware._limits["default"] = (limit, window)
    middleware._hits[key] = deque()

    for _ in range(limit):
        allowed, remaining, _ = middleware._allow(key, limit, window)
        assert allowed is True
        assert remaining >= 0

    allowed, remaining, retry = middleware._allow(key, limit, window)
    assert allowed is False
    assert remaining == 0
    assert retry >= 1


def test_worker_id_and_handler_registry():
    assert worker_id()
    seen = {}

    def handler(payload):
        seen["payload"] = payload
        return {"ok": True}

    register_handler("unit.test", handler)
    from services.job_queue import _HANDLERS

    assert "unit.test" in _HANDLERS
    assert _HANDLERS["unit.test"]({"a": 1}) == {"ok": True}
    assert seen["payload"] == {"a": 1}
