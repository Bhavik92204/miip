from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Empty-string defaults so imports never crash; agents validate at call time.
    groq_api_key: str = ""
    hf_token: str = ""
    pagerduty_api_key: str = ""
    database_url: str = ""

    log_level: str = "INFO"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vision_model: str = "llava-hf/llava-1.5-7b-hf"                        # local HF (future)
    vision_llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"   # Groq vision API
    whisper_model: str = "openai/whisper-base"


settings = Settings()
