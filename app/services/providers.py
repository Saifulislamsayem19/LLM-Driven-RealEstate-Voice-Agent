"""
LLM / embedding / audio provider selection — OpenAI first, Groq as fallback.

OpenAI is used while its key works. If the key is missing, invalid or out of
quota (checked once at startup), chat and audio switch to Groq and embeddings
switch to a local FastEmbed model, since Groq has no embeddings API.
"""
from __future__ import annotations

import io
import re
import wave
from functools import cache
from typing import Any, Callable

import openai
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.core.config import settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_TTS_MAX_CHARS = 200  # Orpheus limit per request


# ── Availability ──────────────────────────────────────────────────────────────

@cache
def openai_available() -> bool:
    """True if the OpenAI key is set and can make a (tiny) billable request."""
    if not settings.OPENAI_API_KEY:
        return False
    try:
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY, timeout=15, max_retries=0)
        client.embeddings.create(model=settings.OPENAI_EMBEDDING_MODEL, input="ping")
        return True
    except Exception as exc:
        print(f"⚠️ OpenAI unavailable ({type(exc).__name__}: {exc}). Falling back to Groq / local embeddings.")
        return False


def groq_available() -> bool:
    return bool(settings.GROQ_API_KEY)


def active_providers() -> dict[str, str]:
    """Which provider serves each capability — reported by /status."""
    use_openai = openai_available()
    fallback = "groq" if groq_available() else "none"
    return {
        "chat": "openai" if use_openai else fallback,
        "embeddings": "openai" if use_openai else "local",
        "speech": "openai" if use_openai else fallback,
    }


# ── Chat model ────────────────────────────────────────────────────────────────

def _openai_chat(max_retries: int) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if settings.OPENAI_REASONING_EFFORT:
        sampling = {"reasoning_effort": settings.OPENAI_REASONING_EFFORT}
    else:
        sampling = {"temperature": settings.LLM_TEMPERATURE}
    return ChatOpenAI(
        model=settings.OPENAI_CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        max_retries=max_retries,
        **sampling,
    )


def _groq_chat() -> BaseChatModel:
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.GROQ_CHAT_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        api_key=settings.GROQ_API_KEY,
        reasoning_effort=settings.GROQ_REASONING_EFFORT,
    )


def get_chat_model() -> Runnable:
    """OpenAI with automatic per-request fallback to Groq (e.g. if the key expires mid-run)."""
    if openai_available():
        if groq_available():
            # Don't burn time retrying OpenAI when Groq can answer instead
            return _openai_chat(max_retries=0).with_fallbacks([_groq_chat()])
        return _openai_chat(max_retries=2)
    if groq_available():
        print(f"Using Groq chat model: {settings.GROQ_CHAT_MODEL}")
        return _groq_chat()
    raise RuntimeError("No working LLM provider: set a valid OPENAI_API_KEY or GROQ_API_KEY.")


# ── Embeddings ────────────────────────────────────────────────────────────────

def _index_tag(provider: str, model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", f"{provider}-{model}").strip("-").lower()


def get_embeddings() -> tuple[Embeddings, str]:
    """Return (embeddings, index tag). The tag keeps indexes built by different models apart —
    vectors from different models are incompatible even when their dimensions match."""
    if openai_available():
        from langchain_openai import OpenAIEmbeddings

        return (
            OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY),
            _index_tag("openai", settings.OPENAI_EMBEDDING_MODEL),
        )

    # On Windows with an outdated VC++ runtime, onnxruntime (used by FastEmbed) crashes on import
    # unless faiss has already loaded its newer bundled msvcp140.dll
    import faiss  # noqa: F401
    from langchain_community.embeddings import FastEmbedEmbeddings

    print(f"Using local embedding model: {settings.LOCAL_EMBEDDING_MODEL}")
    return (
        FastEmbedEmbeddings(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            cache_dir=str(settings.VECTOR_DB_DIR / "models"),
        ),
        _index_tag("local", settings.LOCAL_EMBEDDING_MODEL),
    )


# ── Audio (speech-to-text / text-to-speech) ───────────────────────────────────

@cache
def _openai_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=settings.OPENAI_API_KEY)


@cache
def _groq_client() -> openai.OpenAI:
    # Groq's audio endpoints are OpenAI-compatible
    return openai.OpenAI(api_key=settings.GROQ_API_KEY, base_url=GROQ_BASE_URL)


def _with_fallback(attempts: list[tuple[str, Callable[[], Any]]]) -> Any:
    errors = []
    for name, attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            print(f"⚠️ {name} failed: {exc}")
            errors.append(f"{name}: {exc}")
    if not errors:
        raise RuntimeError("No working speech provider: set a valid OPENAI_API_KEY or GROQ_API_KEY.")
    raise RuntimeError("; ".join(errors))


def transcribe(filename: str, contents: bytes) -> str:
    attempts = []
    if openai_available():
        attempts.append(("OpenAI STT", lambda: _openai_client().audio.transcriptions.create(
            model=settings.OPENAI_STT_MODEL, file=(filename, contents)).text))
    if groq_available():
        attempts.append(("Groq STT", lambda: _groq_client().audio.transcriptions.create(
            model=settings.GROQ_STT_MODEL, file=(filename, contents)).text))
    return _with_fallback(attempts)


def synthesize(text: str) -> tuple[bytes, str]:
    """Return (audio bytes, MIME type)."""
    attempts = []
    if openai_available():
        attempts.append(("OpenAI TTS", lambda: (_openai_client().audio.speech.create(
            model=settings.OPENAI_TTS_MODEL, voice=settings.OPENAI_TTS_VOICE, input=text).content, "audio/mpeg")))
    if groq_available():
        attempts.append(("Groq TTS", lambda: (_groq_tts(text), "audio/wav")))
    return _with_fallback(attempts)


def _groq_tts(text: str) -> bytes:
    clips = [
        _groq_client().audio.speech.create(
            model=settings.GROQ_TTS_MODEL,
            voice=settings.GROQ_TTS_VOICE,
            input=chunk,
            response_format="wav",
        ).content
        for chunk in split_for_tts(text)
    ]
    return concat_wav(clips)


def split_for_tts(text: str, limit: int = GROQ_TTS_MAX_CHARS) -> list[str]:
    """Split text into chunks of at most `limit` chars, preferring sentence then word boundaries."""
    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        pieces = [sentence]
        if len(sentence) > limit:
            pieces, line = [], ""
            for word in sentence.split(" "):
                while len(word) > limit:  # pathological unbroken string
                    pieces.append(word[:limit])
                    word = word[limit:]
                if line and len(line) + 1 + len(word) > limit:
                    pieces.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                pieces.append(line)
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > limit:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def concat_wav(clips: list[bytes]) -> bytes:
    """Join WAV clips that share the same format into one WAV file."""
    if len(clips) == 1:
        return clips[0]
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        for i, clip in enumerate(clips):
            with wave.open(io.BytesIO(clip), "rb") as reader:
                if i == 0:
                    writer.setparams(reader.getparams())
                writer.writeframes(reader.readframes(reader.getnframes()))
    return out.getvalue()
