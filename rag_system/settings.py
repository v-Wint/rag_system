from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    DATABASE_HOST: str = "mongodb://rag_system:rag_system@localhost:27017"
    DATABASE_NAME: str = "rag_system"

    TEXT_EMBEDDING_MODEL_ID: str = "intfloat/multilingual-e5-base"
    MAX_CHUNK_SIZE_TOKENS: int = 2048

    QDRANT_DATABASE_HOST: str = "127.0.0.1"
    QDRANT_DATABASE_PORT: int = 6333

    HUGGINGFACE_ACCESS_TOKEN: str | None = None 
    TEXT_EMBEDDING_DEVICE: str = "cpu"


    MONGO_DOCUMENT_MODELS: list[str] = [
        "rag_system.domain.documents.Document",
    ]

    @classmethod
    def load_settings(cls) -> "Settings":
        return cls()


settings = Settings.load_settings()
