from zenml import pipeline
from typing import Optional
from pydantic import BaseModel, model_validator

from rag_system.settings import settings
from rag_system.domain import Document
from rag_system.infrastructure import Embedder
from rag_system.application.features import CleaningMethod, ChunkingMethod

from steps.features import (
    get_changed_step, clean_documents_step, 
    chunk_documents_step, embed_load_chunks_step)


class FeaturePipelineConfig(BaseModel):
    cleaning_method: CleaningMethod = CleaningMethod.REMNOTE_V1
    chunking_method: ChunkingMethod = ChunkingMethod.HIERARCHICAL_V1
    embedding_model: str = settings.TEXT_EMBEDDING_MODEL_ID
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def resolve(self):
        embedder = Embedder.from_pretrained(self.embedding_model)
        if embedder.max_tokens:
            upper_limit_tokens = min(embedder.max_tokens, settings.MAX_CHUNK_SIZE_TOKENS)
        else:
            upper_limit_tokens = settings.MAX_CHUNK_SIZE_TOKENS

        if not self.max_tokens:
            self.max_tokens = upper_limit_tokens

        if self.max_tokens > upper_limit_tokens:
            raise ValueError(
                f"requested max_tokens={self.max_tokens} exceeds the upper limit "
                f"of {upper_limit_tokens} for embedding_model={self.embedding_model!r} "
                f"(embedder max_seq_length-derived limit, capped by settings.MAX_CHUNK_SIZE_TOKENS)"
            )
        return self

    @property
    def safe_model_slug(self) -> str:
        """Sanitizes model names for safe file paths or db collections."""
        assert self.embedding_model is not None
        return self.embedding_model.replace("/", "_").replace("-", "_")

    def get_collection_name(self, prefix: str = "chunks") -> str:
        parts = [prefix, self.cleaning_method, self.chunking_method, self.safe_model_slug, str(self.max_tokens)]
        return "__".join(parts)


@pipeline
def feature_pipeline(
    documents: list[Document] | None = None,
    deleted_rel_paths: list[str] | None = None,
    config: FeaturePipelineConfig = FeaturePipelineConfig()
):
    raw_documents, to_delete_rel = get_changed_step(
        documents, deleted_rel_paths, config.embedding_model, config.get_collection_name()
    )

    cleaned_documents = clean_documents_step(
        raw_documents, config.cleaning_method
    )

    chunks, schema = chunk_documents_step(
        cleaned_documents, config.chunking_method, config.embedding_model, config.max_tokens
    )

    embed_load_chunks_step(
        chunks, to_delete_rel, config.embedding_model, config.get_collection_name()
    )


if __name__ == '__main__':
    feature_pipeline()
