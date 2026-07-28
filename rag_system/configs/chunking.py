from typing import Optional, Literal, Union, Annotated
from pydantic import BaseModel, model_validator, Field
from enum import Enum

from rag_system.settings import settings


class ChunkingMethod(str, Enum):
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"


class BaseChunkingConfig(BaseModel):
    method: ChunkingMethod
    version: str
    embedding_model: str = settings.TEXT_EMBEDDING_MODEL_ID
    max_chunk_size: Optional[int] = None

    def _resolve_max_chunk_size(self) -> int:
        from rag_system.infrastructure import Embedder

        embedder = Embedder.from_pretrained(self.embedding_model)

        upper_limit = (
            min(embedder.max_tokens, settings.MAX_CHUNK_SIZE_TOKENS)
            if embedder.max_tokens
            else settings.MAX_CHUNK_SIZE_TOKENS
        )

        if self.max_chunk_size is None:
            self.max_chunk_size = upper_limit

        if self.max_chunk_size > upper_limit:
            raise ValueError(...)

        return self.max_chunk_size

    @model_validator(mode="after")
    def resolve(self):
        self._resolve_max_chunk_size()
        return self

    @property
    def safe_model_slug(self) -> str:
        """Sanitizes model names for safe file paths or db collections."""
        assert self.embedding_model is not None
        return self.embedding_model.replace("/", "_").replace("-", "_")

    @property
    def slug_parts(self) -> list[str]:
        """Override/extend in subclasses to add extra slug components."""
        return ["chunks", self.method, self.version, self.safe_model_slug, str(self.max_chunk_size)]

    @property
    def slug(self) -> str:
        return "__".join(self.slug_parts)


class RecursiveV1Config(BaseChunkingConfig):
    method: Literal[ChunkingMethod.RECURSIVE] = ChunkingMethod.RECURSIVE # type: ignore
    version: Literal["1.0"] = "1.0" # type: ignore
    overlap_size: Optional[int] = None

    @model_validator(mode="after")
    def resolve_overlap(self):
        max_chunk_size = self._resolve_max_chunk_size()

        if self.overlap_size is None:
            self.overlap_size = int(max_chunk_size * 0.1)

        return self


    @property
    def slug_parts(self) -> list[str]:
        """Override/extend in subclasses to add extra slug components."""
        existing = super().slug_parts
        existing.append(str(self.overlap_size))
        return existing


class HierarchicalV1Config(BaseChunkingConfig):
    method: Literal[ChunkingMethod.HIERARCHICAL] = ChunkingMethod.HIERARCHICAL # type: ignore
    version: Literal["1.0"] = "1.0" # type: ignore
    max_schema_size: Optional[int] = None

    @model_validator(mode="after")
    def resolve_max_schema_size(self):
        max_chunk_size = self._resolve_max_chunk_size()

        if not self.max_schema_size:
            self.max_schema_size = max_chunk_size
        return self


RecursiveConfig = Annotated[
    Union[RecursiveV1Config],
    Field(discriminator="version")
]

HierarchicalConfig = Annotated[
    Union[HierarchicalV1Config],
    Field(discriminator="version")
]

ChunkingConfig = Annotated[
    Union[HierarchicalConfig],
    Field(discriminator="method")
]
