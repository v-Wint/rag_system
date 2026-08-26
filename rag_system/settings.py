from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    DATABASE_HOST: str = "mongodb://rag_system:rag_system@localhost:27017"
    DATABASE_NAME: str = "rag_system"

    TEXT_EMBEDDING_MODEL_ID: str = "intfloat/multilingual-e5-base"
    MAX_CHUNK_SIZE_TOKENS: int = 2048
    MAX_SCHEMA_TOKENS: int = 1024

    CROSS_ENCODER_MODEL_ID: str = "BAAI/bge-reranker-v2-m3"

    QDRANT_DATABASE_HOST: str = "127.0.0.1"
    QDRANT_DATABASE_PORT: int = 6333

    HUGGINGFACE_ACCESS_TOKEN: str | None = None 
    TEXT_EMBEDDING_DEVICE: str = "cpu"

    GROQ_API_KEY: str | None = None
    DEEPSEEK_API_KEY: str | None = None

    PREPROCESSING_MODEL_ID: str = "deepseek-chat"
    LLM_MODEL_ID: str = "deepseek-chat"
    LLM_AS_JUDGE_ID: str = "deepseek-chat"

    MONGO_DOCUMENT_MODELS: list[str] = [
        "rag_system.domain.documents.Document",
        "rag_system.domain.schema.DocumentTree",
        "rag_system.domain.schema.SchemaString",
        "rag_system.domain.eval_prediction.EvalPrediction",
    ]

    @classmethod
    def load_settings(cls) -> "Settings":
        return cls()


settings = Settings.load_settings()
