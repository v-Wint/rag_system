from typing import Optional, Literal, Union, Annotated
from pydantic import BaseModel, model_validator, Field
from enum import Enum

from rag_system.settings import settings


class ChunkingMethod(str, Enum):
    HIERARCHICAL = "hierarchical"


class HierarchicalV1Config(BaseModel):
    method: ChunkingMethod = ChunkingMethod.HIERARCHICAL
    version: Literal["1.0"] = "1.0"

    embedding_model: str = settings.TEXT_EMBEDDING_MODEL_ID
    max_chunk_size: Optional[int] = None
    max_schema_size: Optional[int] = None

    @model_validator(mode="after")
    def resolve(self):
        from rag_system.infrastructure import Embedder
        embedder = Embedder.from_pretrained(self.embedding_model)
        if embedder.max_tokens:
            # if the embedder is quite good there should be a guardrail against very big chunks
            upper_limit_tokens = min(embedder.max_tokens, settings.MAX_CHUNK_SIZE_TOKENS)
        else:
            upper_limit_tokens = settings.MAX_CHUNK_SIZE_TOKENS

        if not self.max_chunk_size:
            self.max_chunk_size = upper_limit_tokens

        if self.max_chunk_size > upper_limit_tokens:
            raise ValueError(
                f"requested max_chunk_size={self.max_chunk_size} exceeds the upper limit "
                f"of {upper_limit_tokens} for embedding_model={self.embedding_model!r} "
                f"(embedder max_seq_length-derived limit, capped by settings.MAX_CHUNK_SIZE_TOKENS)"
            )

        if not self.max_schema_size:
            self.max_schema_size = self.max_chunk_size

        return self

    @property
    def safe_model_slug(self) -> str:
        """Sanitizes model names for safe file paths or db collections."""
        assert self.embedding_model is not None
        return self.embedding_model.replace("/", "_").replace("-", "_")

    def get_collection_name(self, prefix: str = "chunks") -> str:
        parts = [prefix, self.method, self.safe_model_slug, str(self.max_chunk_size)]
        if self.max_schema_size:
            parts.append(str(self.max_schema_size))
        return "__".join(parts)


# form a hierarchy, boilerplate for now 
HierarchicalConfig = Annotated[
    Union[HierarchicalV1Config],
    Field(discriminator="version")
]

ChunkingConfig = Annotated[
    Union[HierarchicalConfig],
    Field(discriminator="method")
]
