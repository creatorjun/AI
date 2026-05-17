# app/config.py
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    llm_backend: Literal["vllm", "mlx", "ollama"] = "vllm"

    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_api_key: str = "EMPTY"

    mlx_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    mlx_max_tokens: int = 32

    ollama_base_url: str = "http://ollama:11434/v1"
    ollama_model: str = "gemma4:e4b-it-q4_K_M"
    ollama_api_key: str = "ollama"

    hf_token: str = ""
    app_env: str = "development"
    reranker_enabled: bool = True
    hybrid_alpha: float = 0.5
    watch_folder: str = ""
    semantic_chunker_threshold: float = 0.85


settings = Settings()
