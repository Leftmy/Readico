from pathlib import Path
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Readico"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Storage (Storage-agnostic naming)
    STORAGE_PROVIDER: str = "local"
    STORAGE_LOCATION: str = "./data/uploads"
    STORAGE_BUCKET: str = "readico-documents"
    MAX_FILE_SIZE_MB: int = 3000

    # RAG
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    BATCH_SIZE: int = 200

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Granular Vector DB
    VECTOR_DB_TYPE: str = "faiss"
    VECTOR_DB_STORAGE_DIR: Path = Path("./data/faiss_index")
    VECTOR_DB_HOST: str = "localhost"
    VECTOR_DB_PORT: int = 6333
    VECTOR_DB_COLLECTION: str = "readico_docs"

    # Granular Relational DB
    DB_DRIVER: str = "postgresql"
    DB_USER: str = "readico_user"
    DB_PASSWORD: str = "readico_password"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "readico_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @computed_field
    @property
    def local_storage_path(self) -> Path:
        """Returns Path object if local provider is used"""
        return Path(self.STORAGE_LOCATION)

    @computed_field
    @property
    def database_url(self) -> str:
        if self.DB_DRIVER.startswith("sqlite"):
            return f"{self.DB_DRIVER}:///{self.DB_NAME}.db"
        return f"{self.DB_DRIVER}://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @computed_field
    @property
    def vector_db_url(self) -> str:
        if self.VECTOR_DB_TYPE.lower() == "faiss":
            return str(self.VECTOR_DB_STORAGE_DIR)
        return f"http://{self.VECTOR_DB_HOST}:{self.VECTOR_DB_PORT}"


settings = Settings()