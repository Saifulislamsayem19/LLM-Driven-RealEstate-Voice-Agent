"""
Centralised configuration — reads from environment / .env file.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()  


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Project meta ────────────────────────────────────────────────────────
    PROJECT_NAME: str = "HomeFind AI"
    VERSION: str = "1.0.0"

    # ── Paths (resolved relative to project root) ───────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parents[2]  # project root

    @property
    def STATIC_DIR(self) -> Path:
        return self.BASE_DIR / "static"

    @property
    def TEMPLATES_DIR(self) -> Path:
        return self.BASE_DIR / "templates"

    @property
    def DATASET_PATH(self) -> Path:
        return self.BASE_DIR / "dataset" / "USA Housing Dataset.csv"

    @property
    def VECTOR_DB_DIR(self) -> Path:
        return self.BASE_DIR / "vector_db"

    @property
    def USER_DATA_DIR(self) -> Path:
        return self.BASE_DIR / "user_data"

    @property
    def USER_REQUIREMENTS_FILE(self) -> Path:
        return self.USER_DATA_DIR / "user_requirements.csv"

    # ── External services ────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-6-luna"
    # gpt-6-luna is a reasoning model (no temperature). Set empty for a non-reasoning model to use LLM_TEMPERATURE.
    OPENAI_REASONING_EFFORT: str = "low"  # none | low | medium | high
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_TTS_MODEL: str = "gpt-4o-mini-tts"  # OpenAI shuts down all /audio/speech models on 2027-01-06; Groq TTS takes over
    OPENAI_TTS_VOICE: str = "marin"
    OPENAI_STT_MODEL: str = "gpt-transcribe"

    # Fallback when the OpenAI key is missing / expired (key: https://console.groq.com/keys)
    GROQ_API_KEY: str = ""
    GROQ_CHAT_MODEL: str = "openai/gpt-oss-120b"
    GROQ_REASONING_EFFORT: str = "low"  # low | medium | high
    GROQ_STT_MODEL: str = "whisper-large-v3-turbo"
    GROQ_TTS_MODEL: str = "canopylabs/orpheus-v1-english"
    GROQ_TTS_VOICE: str = "hannah"  # autumn, diana, hannah, austin, daniel, troy

    # Groq has no embeddings API, so the fallback embeds locally (downloaded on first use)
    LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 7860
    # The web UI is served from the same origin, so no cross-origin access is needed by default
    ALLOWED_ORIGINS: List[str] = []

    # ── Security ─────────────────────────────────────────────────────────────
    # Admin pages (/api-@dmin, /requirements) are disabled until ADMIN_PASSWORD is set
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = ""
    RATE_LIMIT_CHAT_PER_MINUTE: int = 20  # per client IP; 0 disables
    RATE_LIMIT_AUDIO_PER_MINUTE: int = 20
    # Daily caps on messages that reach the LLM (refused messages are free); 0 disables
    DAILY_CHAT_LIMIT_PER_IP: int = 150
    DAILY_CHAT_LIMIT_GLOBAL: int = 3000
    # Visitors who send this many jailbreak attempts are locked out for the given time
    JAILBREAK_STRIKE_LIMIT: int = 5
    JAILBREAK_LOCKOUT_MINUTES: int = 15
    MAX_QUERY_CHARS: int = 2000
    MAX_HISTORY_MESSAGES: int = 60
    MAX_TTS_CHARS: int = 4000
    MAX_AUDIO_UPLOAD_MB: int = 25

    # ── Agent behaviour ──────────────────────────────────────────────────────
    LLM_TEMPERATURE: float = 0.3
    MAX_PROPERTIES_SHOWN: int = 3
    VECTOR_BATCH_SIZE: int = 100
    EXACT_PRICE_TOLERANCE: float = 5000.0
    SIMILAR_PRICE_TOLERANCE_PCT: float = 0.15


settings = Settings()

# Ensure runtime directories exist
settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
settings.USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
