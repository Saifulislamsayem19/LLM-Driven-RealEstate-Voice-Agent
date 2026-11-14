"""
Unit tests for provider selection (OpenAI → Groq / local fallback).
Run with: pytest tests/
"""
import io
import wave

import pytest


@pytest.fixture
def providers(monkeypatch):
    from app.core.config import settings
    from app.services import providers

    probe = providers.openai_available  # tests may monkeypatch it away
    probe.cache_clear()
    yield providers, settings, monkeypatch
    probe.cache_clear()


def test_no_openai_key_uses_groq_and_local_embeddings(providers):
    providers, settings, monkeypatch = providers
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-test")

    assert providers.active_providers() == {"chat": "groq", "embeddings": "local", "speech": "groq"}
    assert type(providers.get_chat_model()).__name__ == "ChatGroq"


def test_working_openai_key_falls_back_to_groq_per_request(providers):
    providers, settings, monkeypatch = providers
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(providers, "openai_available", lambda: True)

    llm = providers.get_chat_model()
    assert type(llm.runnable).__name__ == "ChatOpenAI"
    assert [type(f).__name__ for f in llm.fallbacks] == ["ChatGroq"]


def test_no_keys_raises(providers):
    providers, settings, monkeypatch = providers
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")

    with pytest.raises(RuntimeError, match="No working LLM provider"):
        providers.get_chat_model()


def test_speech_falls_back_when_openai_fails(providers):
    providers, settings, monkeypatch = providers
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(providers, "openai_available", lambda: True)

    def openai_tts(*args, **kwargs):
        raise RuntimeError("insufficient_quota")

    monkeypatch.setattr(providers._openai_client().audio.speech, "create", openai_tts)
    monkeypatch.setattr(providers, "_groq_tts", lambda text: b"RIFF-groq")

    assert providers.synthesize("Hello") == (b"RIFF-groq", "audio/wav")


def test_split_for_tts_respects_limit_and_keeps_text():
    from app.services.providers import split_for_tts

    text = "Short one. " + "This is a much longer sentence with plenty of words in it " * 8 + "! End?"
    chunks = split_for_tts(text, limit=200)

    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks).split() == text.split()
    assert split_for_tts("x" * 450, limit=200) == ["x" * 200, "x" * 200, "x" * 50]


def test_concat_wav_joins_frames():
    from app.services.providers import concat_wav

    def clip(n_frames):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x01\x00" * n_frames)
        return buf.getvalue()

    with wave.open(io.BytesIO(concat_wav([clip(100), clip(50)])), "rb") as r:
        assert r.getnframes() == 150
        assert r.getframerate() == 24000
