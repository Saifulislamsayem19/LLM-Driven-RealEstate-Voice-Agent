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
    OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo-16k"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_TTS_MODEL: str = "tts-1"
    OPENAI_TTS_VOICE: str = "alloy"
    OPENAI_STT_MODEL: str = "whisper-1"

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 7860
    ALLOWED_ORIGINS: List[str] = ["*"]

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
