from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    DATABASE_HOST: str = "mongodb://rag_system:rag_system@localhost:27017"
    DATABASE_NAME: str = "rag_system"

    MONGO_DOCUMENT_MODELS: list[str] = [
        "rag_system.domain.documents.Document",
    ]

    @classmethod
    def load_settings(cls) -> "Settings":
        return cls()


settings = Settings.load_settings()
