from pydantic import BaseModel, model_validator

from rag_system.settings import settings


class RAGConfig(BaseModel):
    collection_name: str
    preprocess_template_file: str = "templates/preprocess.txt"
    preprocess_model: str = settings.PREPROCESSING_MODEL_ID
    preprocess_model_temperature: float = 0.1

    embedding_model: str = settings.TEXT_EMBEDDING_MODEL_ID
    cross_encoder_model: str = settings.CROSS_ENCODER_MODEL_ID
    vector_retrieval_size_fact: int = 10
    reranking_size_fact: int = 5
    vector_retrieval_size_schema: int = 10
    reranking_size_schema: int = 3

    llm_model: str = settings.LLM_MODEL_ID
    llm_model_temperature: float = 0.1
    llm_fact_template: str = "templates/fact.txt"
    llm_schema_template: str = "templates/schema.txt"
    llm_general_template: str = "templates/general.txt"
