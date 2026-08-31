from typing import Optional, Literal, Union, Annotated, TypeVar, Generic, Any, Callable
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator, TypeAdapter
import yaml

from rag_system.settings import settings
from .enums import ChunkingMethod, SizeMetric


MethodT = TypeVar("MethodT", bound=ChunkingMethod)
VersionT = TypeVar("VersionT", bound=str)


class BaseChunkingConfig(BaseModel, ABC, Generic[MethodT, VersionT]):
    method: MethodT
    version: VersionT
    size_metric: SizeMetric = SizeMetric.TOKENS

    @model_validator(mode="after")
    def _mark_discriminators_set(self) -> "BaseChunkingConfig":
        # Force these into model_fields_set regardless of how the
        # instance was constructed, so exclude_unset never drops them.
        self.__pydantic_fields_set__ |= {"strategy", "version"}
        return self

    @abstractmethod
    def resolve(self) -> "BaseChunkingConfig":
        ...

    @abstractmethod
    def make_get_size(self) -> Callable[[str], int]:
        """Return a size function for the config's size_metric."""
        ...

    def get_params(self) -> dict[str, Any]:
        return {
            'method': self.method,
            'version': self.version,
            'size_metric': self.size_metric,
        }


class EmbeddingChunkingConfig(BaseChunkingConfig[MethodT, VersionT], Generic[MethodT, VersionT]):
    """Base for chunking configs whose max_chunk_size is resolved against the embedding model."""
    embedding_model: str = settings.TEXT_EMBEDDING_MODEL_ID
    raw_max_chunk_size: Optional[int] = None

    resolved_max_chunk_size: Optional[int] = None

    @property
    def max_chunk_size(self) -> int:
        if self.resolved_max_chunk_size is not None:
            return self.resolved_max_chunk_size

        from rag_system.infrastructure import Embedder
        embedder = Embedder.from_pretrained(self.embedding_model)
        max_tokens = embedder.max_tokens or settings.MAX_CHUNK_SIZE_TOKENS
        upper_limit = min(max_tokens, settings.MAX_CHUNK_SIZE_TOKENS)

        if self.raw_max_chunk_size is None:
            self.resolved_max_chunk_size = upper_limit
        else:
            if self.raw_max_chunk_size > upper_limit:
                raise ValueError(
                    f"requested raw_max_chunk_size={self.raw_max_chunk_size} exceeds the upper limit "
                    f"of {upper_limit} for embedding_model={self.embedding_model!r}"
                )
            self.resolved_max_chunk_size = self.raw_max_chunk_size

        return self.resolved_max_chunk_size

    def resolve(self) -> "EmbeddingChunkingConfig":
        _ = self.max_chunk_size
        return self

    def make_get_size(self) -> Callable[[str], int]:
        """Return a size function honoring size_metric (tokens vs raw chars)."""
        if self.size_metric == SizeMetric.TOKENS:
            from rag_system.infrastructure import Embedder
            embedder = Embedder.from_pretrained(self.embedding_model)
            return lambda s: embedder.get_token_count(s)
        return len

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

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({
            'embedding_model': self.embedding_model,
            'max_chunk_size': self.max_chunk_size,
        })
        return params


class RecursiveV1Config(EmbeddingChunkingConfig[Literal[ChunkingMethod.RECURSIVE], Literal["1.0"]]):
    method: Literal[ChunkingMethod.RECURSIVE] = ChunkingMethod.RECURSIVE
    version: Literal["1.0"] = "1.0"
    raw_overlap_size: Optional[int] = None

    resolved_overlap_size: Optional[int] = None

    @property
    def overlap_size(self) -> int:
        if self.resolved_overlap_size is not None:
            return self.resolved_overlap_size

        if self.raw_overlap_size is not None:
            self.resolved_overlap_size = self.raw_overlap_size
        else:
            self.resolved_overlap_size = int(self.max_chunk_size * 0.1)

        return self.resolved_overlap_size

    def resolve(self):
        super().resolve()
        _ = self.overlap_size
        return self

    @property
    def slug_parts(self) -> list[str]:
        return super().slug_parts + [str(self.overlap_size)]

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()

        params.update({
            'overlap_size': self.overlap_size,
        })
        return params


class HierarchicalV1Config(EmbeddingChunkingConfig[Literal[ChunkingMethod.HIERARCHICAL], Literal["1.0"]]):
    method: Literal[ChunkingMethod.HIERARCHICAL] = ChunkingMethod.HIERARCHICAL
    version: Literal["1.0"] = "1.0"
    raw_max_schema_size: Optional[int] = None

    resolved_max_schema_size: Optional[int] = None

    @property
    def max_schema_size(self) -> int:
        if self.resolved_max_schema_size is not None:
            return self.resolved_max_schema_size

        # schema is not being embedded, so there is not constraint
        if self.raw_max_schema_size is not None:
            self.resolved_max_schema_size = self.raw_max_schema_size
        else:
            self.resolved_max_schema_size = self.max_chunk_size

        return self.resolved_max_schema_size

    def resolve(self):
        super().resolve()
        _ = self.max_schema_size
        return self

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()

        params.update({
            'max_schema_size': self.max_schema_size,
        })
        return params


class AgenticV1Config(BaseChunkingConfig[Literal[ChunkingMethod.AGENTIC], Literal["1.0"]]):
    method: Literal[ChunkingMethod.AGENTIC] = ChunkingMethod.AGENTIC
    version: Literal["1.0"] = "1.0"
    size_metric: SizeMetric = SizeMetric.CHARS
    max_chunk_size: int = 8000

    def resolve(self) -> "AgenticV1Config":
        return self

    def make_get_size(self) -> Callable[[str], int]:
        """Agentic splits on raw characters — no embedder involved."""
        return len

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({
            'max_chunk_size': self.max_chunk_size,
        })
        return params


RecursiveConfig = Annotated[
    Union[RecursiveV1Config],
    Field(discriminator="version")
]

HierarchicalConfig = Annotated[
    Union[HierarchicalV1Config],
    Field(discriminator="version")
]

AgenticConfig = Annotated[
    Union[AgenticV1Config],
    Field(discriminator="version")
]

ChunkingConfig = Annotated[
    Union[RecursiveConfig, HierarchicalConfig, AgenticConfig],
    Field(discriminator="method")
]


def load_from_yaml(yaml_path: str) -> ChunkingConfig:
    with open(yaml_path, "r") as f:
        raw_data = yaml.safe_load(f)
    return TypeAdapter(ChunkingConfig).validate_python(raw_data)
