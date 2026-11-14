"""
Audio endpoints — speech-to-text and text-to-speech.
Logic preserved exactly from the original audio.py.
"""
from __future__ import annotations

import base64
import os

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import audio_rate_limit
from app.services.providers import synthesize, transcribe

router = APIRouter(tags=["audio"])


@router.post("/transcribe", dependencies=[Depends(audio_rate_limit)])
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Receive audio file, transcribe using Whisper (OpenAI, falling back to Groq), return text."""
    max_bytes = settings.MAX_AUDIO_UPLOAD_MB * 1024 * 1024
    if audio_file.content_type and not audio_file.content_type.startswith(("audio/", "video/webm")):
        raise HTTPException(status_code=415, detail="Please upload an audio file.")
    contents = await audio_file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Audio file is larger than {settings.MAX_AUDIO_UPLOAD_MB} MB.")

    try:
        # Only the extension matters to the provider; never trust the client's filename
        ext = os.path.splitext(audio_file.filename or "")[1].lower()
        filename = f"audio{ext if ext.isascii() and len(ext) <= 6 else '.webm'}"
        return {"text": await run_in_threadpool(transcribe, filename, contents)}

    except Exception as exc:
        print(f"/audio/transcribe error: {exc!r}")
        raise HTTPException(status_code=502, detail="Speech recognition is unavailable right now.")


@router.post("/tts", dependencies=[Depends(audio_rate_limit)])
async def text_to_speech(request: Request):
    """Receive text, generate speech audio, return base64-encoded audio and its MIME type."""
    try:
        data = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request")
    text = data.get("text", "") if isinstance(data, dict) else ""

    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="No text provided")
    text = text[: settings.MAX_TTS_CHARS]

    try:
        audio, mime_type = await run_in_threadpool(synthesize, text)
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        return {"audio": audio_base64, "mime_type": mime_type}

    except Exception as exc:
        print(f"/audio/tts error: {exc!r}")
        raise HTTPException(status_code=502, detail="Text-to-speech is unavailable right now.")
