"""Central configuration, loaded from environment / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    embedding_model: str = "voyage-3"
    embedding_dim: int = 1024
    generation_model: str = "claude-opus-4-8"

    retrieval_top_k: int = 8
    rrf_k: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
