"""
Audio endpoints — speech-to-text and text-to-speech.
"""
from __future__ import annotations

import base64

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.services.providers import synthesize, transcribe

router = APIRouter(tags=["audio"])


@router.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Receive audio file, transcribe using Whisper (OpenAI, falling back to Groq), return text."""
    try:
        contents = await audio_file.read()
        return {"text": transcribe(audio_file.filename or "audio.webm", contents)}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/tts")
async def text_to_speech(request: Request):
    """Receive text, generate speech audio, return base64-encoded audio and its MIME type."""
    data = await request.json()
    text: str = data.get("text", "")

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        audio, mime_type = synthesize(text)
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        return {"audio": audio_base64, "mime_type": mime_type}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
