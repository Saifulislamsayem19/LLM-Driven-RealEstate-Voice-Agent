import os
import openai
import base64
from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from dotenv import load_dotenv

load_dotenv()

# Create router
audio_router = APIRouter(prefix="/audio", tags=["audio"])

# Load API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize the OpenAI client 
client = openai.OpenAI(api_key=api_key)

@audio_router.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Receive audio file, transcribe using Whisper, return text."""
    try:
        # Read the file content
        contents = await audio_file.read()
        
        # Save to a temporary file (OpenAI API requires a file-like object)
        temp_filename = f"temp_{audio_file.filename}"
        with open(temp_filename, "wb") as f:
            f.write(contents)
        
        # Open the file for transcription
        with open(temp_filename, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        
        # Clean up temporary file
        os.remove(temp_filename)
        
        return {"text": transcript.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@audio_router.post("/tts")
async def text_to_speech(request: Request):
    """Receive text input, generate speech audio, return audio bytes (base64)."""
    data = await request.json()
    text = data.get("text", "")
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            input=text,
            voice="alloy"  
        )
        
        audio_content = response.content 
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        return {"audio": audio_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))