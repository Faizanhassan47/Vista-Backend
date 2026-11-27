"""
Global HTTP rate limiting middleware.

Uses a sliding window per client IP (and optional route class). Designed to sit
behind a load balancer when TRUST_PROXY_HEADERS=1 so X-Forwarded-For / X-Real-IP
are honored. In-memory counters are per process; nginx also rate-limits at the edge.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from core.config import (
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    RATE_LIMIT_AI_REQUESTS,
    RATE_LIMIT_AUTH_REQUESTS,
    env_bool,
)


def _client_ip(request: Request) -> str:
    if env_bool("TRUST_PROXY_HEADERS", False):
        forwarded = request.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if forwarded or real_ip:
            return forwarded or real_ip
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _bucket_for(path: str, method: str) -> str:
    lower = path.lower()
    if lower.startswith("/api/auth/login") or lower.startswith("/api/auth/logout"):
        return "auth"
    if lower.startswith("/api/ai/") or "/ai/" in lower:
        return "ai"
    if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        return "write"
    return "default"


def _is_exempt(path: str) -> bool:
    """High-frequency viewer assets must not compete with API rate buckets."""
    if path in {"/api/status", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}:
        return True
    lower = path.lower()
    # OpenSeadragon Deep Zoom tiles / descriptor (many GETs per zoom)
    if "/slide_files/" in lower or lower.endswith(".dzi") or "/slide.dzi" in lower:
        return True
    # Radiology slice / mask / volume GETs during MPR/pan
    if "/radiology/frame" in lower or "/radiology/hu-mask" in lower or "/radiology/volume" in lower:
        return True
    # Thumbnails in the slide tray
    if "/thumbnail" in lower and lower.startswith("/api/slides/"):
        return True
    # AI / TotalSeg job status polling (frequent GETs while a job runs)
    if lower.startswith("/api/ai/jobs/") or "/totalsegmentator/jobs/" in lower:
        return True
    return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rejects abusive clients with HTTP 429 and Retry-After."""

    def __init__(self, app):
        super().__init__(app)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._limits = {
            "auth": (RATE_LIMIT_AUTH_REQUESTS, RATE_LIMIT_WINDOW_SECONDS),
            "ai": (RATE_LIMIT_AI_REQUESTS, RATE_LIMIT_WINDOW_SECONDS),
            "write": (max(RATE_LIMIT_REQUESTS // 2, 30), RATE_LIMIT_WINDOW_SECONDS),
            "default": (RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS),
        }

    def _allow(self, key: str, limit: int, window: int) -> tuple[bool, int, int]:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] > window:
                bucket.popleft()
            if len(bucket) >= limit:
                retry = max(1, int(window - (now - bucket[0])) + 1)
                return False, 0, retry
            bucket.append(now)
            remaining = max(0, limit - len(bucket))
            return True, remaining, window

    async def dispatch(self, request: Request, call_next) -> Response:
        if not RATE_LIMIT_ENABLED:
            return await call_next(request)

        path = request.url.path or "/"
        if _is_exempt(path):
            return await call_next(request)

        bucket = _bucket_for(path, request.method)
        limit, window = self._limits[bucket]
        ip = _client_ip(request)
        allowed, remaining, retry_after = self._allow(f"{bucket}:{ip}", limit, window)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again shortly."},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Bucket": bucket,
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Bucket"] = bucket
        return response
