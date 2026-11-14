"""
Integration tests for admin auth, request limits, rate limiting and security headers.
"""
import base64

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from unittest.mock import patch

    from app.services.agent import agent

    with patch.object(agent, "initialize", return_value=True), \
         patch.object(agent, "setup_chain", return_value=True):
        from app.core.application import create_app
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            yield c


@pytest.fixture
def admin_password(monkeypatch):
    from app.core.config import settings

    # Don't depend on the developer's own .env credentials
    monkeypatch.setattr(settings, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(settings, "ADMIN_PASSWORD", "s3cret-pass")
    return "s3cret-pass"


def _basic(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def test_admin_routes_hidden_without_admin_password(client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ADMIN_PASSWORD", "")
    assert client.get("/api-@dmin").status_code == 404
    assert client.get("/requirements").status_code == 404


def test_admin_routes_require_credentials(client, admin_password):
    response = client.get("/requirements")
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Basic")
    assert client.get("/requirements", headers=_basic("admin", "wrong")).status_code == 401
    assert client.get("/api-@dmin", headers=_basic("admin", admin_password)).status_code == 200


def test_admin_key_update_rejects_cross_site_and_env_injection(client, admin_password, tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    auth = _basic("admin", admin_password)

    cross_site = client.post("/api-@dmin", data={"api_key": "sk-" + "a" * 40},
                             headers={**auth, "Origin": "https://evil.example"})
    assert cross_site.status_code == 403

    injected = client.post("/api-@dmin", data={"api_key": "sk-" + "a" * 40 + "\nADMIN_PASSWORD=x"},
                           headers={**auth, "Origin": "http://testserver"})
    assert injected.status_code == 200
    assert "valid OpenAI API key" in injected.text
    assert not (tmp_path / ".env").exists()


def test_ask_rejects_oversized_and_malformed_input(client):
    from app.core.config import settings

    too_long = client.post("/ask", json={"query": "x" * (settings.MAX_QUERY_CHARS + 1)})
    assert too_long.status_code == 413
    assert client.post("/ask", json={"query": 123}).status_code == 400
    assert client.post("/ask", content=b"not json", headers={"Content-Type": "application/json"}).status_code == 400


def test_errors_do_not_leak_internals(client, monkeypatch):
    from app.api.routes import chat

    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail sk-proj-abc")

    monkeypatch.setattr(chat, "process_query", boom)
    response = client.post("/ask", json={"query": "hello"})
    assert response.status_code == 500
    assert "secret" not in response.text


def test_transcribe_rejects_non_audio_and_large_files(client, monkeypatch):
    from app.core.config import settings

    not_audio = client.post("/audio/transcribe", files={"audio_file": ("x.html", b"<html>", "text/html")})
    assert not_audio.status_code == 415

    monkeypatch.setattr(settings, "MAX_AUDIO_UPLOAD_MB", 0)
    too_big = client.post("/audio/transcribe", files={"audio_file": ("a.wav", b"RIFF1234", "audio/wav")})
    assert too_big.status_code == 413


def test_rate_limiter_blocks_after_limit():
    from fastapi import HTTPException
    from starlette.requests import Request

    from app.core.security import RateLimiter

    limiter = RateLimiter(limit=2, window_seconds=60)
    request = Request({"type": "http", "client": ("1.2.3.4", 1234), "headers": []})
    limiter(request)
    limiter(request)
    with pytest.raises(HTTPException) as exc:
        limiter(request)
    assert exc.value.status_code == 429
    limiter(Request({"type": "http", "client": ("5.6.7.8", 1234), "headers": []}))  # other clients unaffected


def test_security_headers_present(client):
    headers = client.get("/").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert "script-src 'self' https://cdnjs.cloudflare.com" in headers["content-security-policy"]
    assert "frame-ancestors 'self' https://huggingface.co" in headers["content-security-policy"]
    assert "content-security-policy" not in client.get("/docs").headers


def test_no_cors_by_default(client):
    response = client.get("/status", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers

