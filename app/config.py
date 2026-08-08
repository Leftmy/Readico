from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    APP_HOST: str = "0.0.0.0"
    APP_ENV: str = "development"
    APP_NAME: str = "Readico API"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Storage & Processing
    STORAGE_PROVIDER: str = "local"
    STORAGE_LOCATION: str = "./data/uploads"
    STORAGE_BUCKET: str = "readico-documents"

    # RAG & Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    BATCH_SIZE: int = 200

    # OpenAI API
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # Vector DB
    VECTOR_DB_TYPE: str = "qdrant"
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()