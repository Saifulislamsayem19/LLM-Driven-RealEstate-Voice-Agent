"""
Security helpers — admin authentication, per-client rate limiting, and HTTP security headers.
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

# ── Admin authentication ──────────────────────────────────────────────────────

_basic = HTTPBasic(auto_error=False, realm="HomeFind AI admin")


def require_admin(credentials: HTTPBasicCredentials | None = Depends(_basic)) -> None:
    """HTTP Basic auth against ADMIN_PASSWORD. Admin routes are hidden (404) until it is set."""
    if not settings.ADMIN_PASSWORD:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    valid = credentials is not None and secrets.compare_digest(
        credentials.password.encode(), settings.ADMIN_PASSWORD.encode()
    ) and secrets.compare_digest(credentials.username.encode(), settings.ADMIN_USERNAME.encode())
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": 'Basic realm="HomeFind AI admin"'},
        )


def require_same_origin(request: Request) -> None:
    """Block cross-site form posts (browsers resend Basic auth credentials automatically)."""
    source = request.headers.get("origin") or request.headers.get("referer")
    if not source or urlparse(source).netloc != request.headers.get("host"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-site request blocked")


# ── Rate limiting ─────────────────────────────────────────────────────────────

class RateLimiter:
    """In-memory sliding-window limiter, keyed by client IP. Fine for a single instance."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def __call__(self, request: Request) -> None:
        if self.limit <= 0:
            return
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = int(self.window - (now - hits[0])) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests — please wait a moment and try again.",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)
            if len(self._hits) > 10_000:  # drop idle clients so memory stays bounded
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]:
                    del self._hits[k]


chat_rate_limit = RateLimiter(settings.RATE_LIMIT_CHAT_PER_MINUTE)
audio_rate_limit = RateLimiter(settings.RATE_LIMIT_AUDIO_PER_MINUTE)


# ── Daily quotas and jailbreak lockout ────────────────────────────────────────

class SlidingCounter:
    """Timestamps per key within a sliding window (in memory, thread-safe)."""

    def __init__(self, window_seconds: float):
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        return hits

    def count(self, key: str) -> int:
        with self._lock:
            return len(self._prune(key, time.monotonic()))

    def add(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune(key, now).append(now)
            if len(self._hits) > 10_000:  # drop idle keys so memory stays bounded
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]:
                    del self._hits[k]

    def retry_after(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            hits = self._prune(key, now)
            return int(self.window - (now - hits[0])) + 1 if hits else 0

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


def _too_many(detail: str, retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail,
        headers={"Retry-After": str(max(retry_after, 1))},
    )


class AbuseGuard:
    """Cost protection for the LLM-backed chat endpoint.

    - Per-visitor and global daily caps on requests that actually reach the LLM
      (refused off-topic / jailbreak messages never count, they cost nothing).
    - Visitors who keep sending jailbreak attempts are locked out for a while.
    Limits are read from settings on every call. 0 disables a limit.
    """

    DAY = 24 * 3600.0
    GLOBAL_KEY = "*"

    def __init__(self):
        self._daily = SlidingCounter(self.DAY)
        self._strikes = SlidingCounter(0)  # window is set per call from settings

    @staticmethod
    def client_key(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def check_not_locked(self, client: str) -> None:
        limit = settings.JAILBREAK_STRIKE_LIMIT
        self._strikes.window = settings.JAILBREAK_LOCKOUT_MINUTES * 60.0
        if limit > 0 and self._strikes.count(client) >= limit:
            raise _too_many(
                "Too many blocked requests. Please try again later.", self._strikes.retry_after(client)
            )

    def record_strike(self, client: str) -> None:
        self._strikes.window = settings.JAILBREAK_LOCKOUT_MINUTES * 60.0
        self._strikes.add(client)

    def check_quota(self, client: str) -> None:
        per_ip, overall = settings.DAILY_CHAT_LIMIT_PER_IP, settings.DAILY_CHAT_LIMIT_GLOBAL
        if per_ip > 0 and self._daily.count(client) >= per_ip:
            raise _too_many("You've reached today's chat limit. Please come back tomorrow.", self._daily.retry_after(client))
        if overall > 0 and self._daily.count(self.GLOBAL_KEY) >= overall:
            raise _too_many(
                "The assistant is very busy today. Please try again later.", self._daily.retry_after(self.GLOBAL_KEY)
            )

    def record_llm_call(self, client: str) -> None:
        self._daily.add(client)
        self._daily.add(self.GLOBAL_KEY)

    def reset(self) -> None:
        self._daily.clear()
        self._strikes.clear()


abuse_guard = AbuseGuard()


# ── Security headers ──────────────────────────────────────────────────────────

_CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self' https://cdnjs.cloudflare.com",
    "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com",
    "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com",
    "img-src 'self' data:",
    "media-src 'self' data: blob:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    # Hugging Face Spaces shows the app inside an iframe on huggingface.co
    "frame-ancestors 'self' https://huggingface.co",
])

# Swagger / ReDoc pages load their own CDN scripts and inline code
_CSP_EXEMPT_PREFIXES = ("/docs", "/redoc")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "microphone=(self), camera=(), geolocation=()")
        if not request.url.path.startswith(_CSP_EXEMPT_PREFIXES):
            response.headers.setdefault("Content-Security-Policy", _CSP)
        return response
