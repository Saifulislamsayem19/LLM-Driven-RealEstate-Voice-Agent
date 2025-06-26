import os
import openai
import base64
from flask import Blueprint, request, jsonify

audio_bp = Blueprint("audio", __name__)

# Load API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize the OpenAI client
client = openai.OpenAI(api_key=api_key)

@audio_bp.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """Receive audio file, transcribe using Whisper, return text."""
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files["audio_file"]

    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return jsonify({"text": transcript.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@audio_bp.route("/tts", methods=["POST"])
def text_to_speech():
    """Receive text input, generate speech audio, return audio bytes (base64)."""
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        response = client.audio.speech.create(
            model="tts-1",
            input=text,
            voice="alloy"
        )
        
        # Read the audio content and encode to base64
        audio_content = response.content 
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        return jsonify({"audio": audio_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
