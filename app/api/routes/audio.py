"""
Audio endpoints — speech-to-text and text-to-speech.
Logic preserved exactly from the original audio.py.
"""
from __future__ import annotations

import base64
import os

import openai
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.config import settings

router = APIRouter(tags=["audio"])

_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)


@router.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Receive audio file, transcribe using Whisper, return text."""
    try:
        contents = await audio_file.read()
        temp_filename = f"temp_{audio_file.filename}"

        with open(temp_filename, "wb") as f:
            f.write(contents)

        with open(temp_filename, "rb") as f:
            transcript = _client.audio.transcriptions.create(
                model=settings.OPENAI_STT_MODEL,
                file=f,
            )

        os.remove(temp_filename)
        return {"text": transcript.text}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/tts")
async def text_to_speech(request: Request):
    """Receive text, generate speech audio, return base64-encoded audio."""
    data = await request.json()
    text: str = data.get("text", "")

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        response = _client.audio.speech.create(
            model=settings.OPENAI_TTS_MODEL,
            input=text,
            voice=settings.OPENAI_TTS_VOICE,
        )
        audio_base64 = base64.b64encode(response.content).decode("utf-8")
        return {"audio": audio_base64}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
