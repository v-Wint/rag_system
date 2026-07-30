from typing import Literal, Union, Annotated, TypeVar, Generic, Optional, Callable
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator

from .chunking import ChunkingConfig, RecursiveV1Config, HierarchicalV1Config, HierarchicalConfig, BaseChunkingConfig
from .enums import InferenceStrategy

from rag_system.settings import settings
from rag_system.infrastructure import PromptStore


StrategyT = TypeVar("StrategyT", bound=InferenceStrategy)
VersionT = TypeVar("VersionT", bound=str)
ChunkingT = TypeVar("ChunkingT", bound=BaseChunkingConfig)


class BaseInferenceConfig(BaseModel, ABC, Generic[StrategyT, VersionT, ChunkingT]):
    strategy: StrategyT
    version: VersionT

    chunking: ChunkingT

    @abstractmethod
    def resolve(self):
        return self

    def get_template_resolver(self):
        return lambda name: PromptStore().load_one(
            strategy=self.strategy,
            version=self.version,
            name=name
        )


class BaseTemplateConfig(BaseModel, ABC):
    """Base mixin providing private resolver storage."""
    _resolver: Optional[Callable[[str], str]] = None

    def _resolve_field(self, field_name: str, template_name: str) -> str:
        """Helper to resolve and cache template text into resolved_* fields."""
        current_val = getattr(self, field_name)
        if current_val is not None:
            return current_val
        if self._resolver is None:
            raise RuntimeError(
                f"Resolver not bound to {self.__class__.__name__}. "
                "Ensure parent config ran model_validator."
            )
        resolved_val = self._resolver(template_name)
        setattr(self, field_name, resolved_val)
        return resolved_val


class RecursiveV1InferenceConfig(
    BaseInferenceConfig[
        Literal[InferenceStrategy.RECURSIVE],
        Literal["1.0"],
        ChunkingConfig
    ]
):
    strategy: Literal[InferenceStrategy.RECURSIVE] = InferenceStrategy.RECURSIVE
    version: Literal["1.0"] = "1.0"
    # theoretically naive retrieval can use hierarchical retrieval chunks 
    chunking: ChunkingConfig = RecursiveV1Config()

    class RetrievalConfig(BaseModel):
        k: int = 6

    retrieval: RetrievalConfig = RetrievalConfig()

    class GenerationConfig(BaseTemplateConfig):
        template_name: str = 'template'
        model_name: str = settings.LLM_MODEL_ID
        model_temperature: float = 0.3

        resolved_template_text: Optional[str] = None

        @property
        def template_text(self) -> str:
            return self._resolve_field("resolved_template_text", self.template_name)

    generation: GenerationConfig = GenerationConfig()

    @model_validator(mode="after")
    def attach_context(self) -> "RecursiveV1InferenceConfig":
        self.generation._resolver = self.get_template_resolver()
        return self

    def resolve(self):
        super().resolve()
        _ = self.generation.template_text
        return self


class HierarchicalV1InferenceConfig(
    BaseInferenceConfig[
        Literal[InferenceStrategy.HIERARCHICAL],
        Literal["1.0"],
        HierarchicalConfig,
    ]
):
    strategy: Literal[InferenceStrategy.HIERARCHICAL] = InferenceStrategy.HIERARCHICAL
    version: Literal["1.0"] = "1.0"
    chunking: HierarchicalConfig = HierarchicalV1Config()

    class PreprocessConfig(BaseTemplateConfig):
        template_name: str = 'preprocess'
        model_name: str = settings.PREPROCESSING_MODEL_ID
        model_temperature: float = 0.1

        resolved_template_text: Optional[str] = None

        @property
        def template_text(self) -> str:
            return self._resolve_field("resolved_template_text", self.template_name)

    preprocess: PreprocessConfig = PreprocessConfig()

    class RetrievalConfig(BaseModel):
        class VectorSearchConfig(BaseModel):
            fact_k: int = 20
            schema_k: int = 10

        vector: VectorSearchConfig = VectorSearchConfig()

        class RerankerConfig(BaseModel):
            model_name: str = settings.CROSS_ENCODER_MODEL_ID
            fact_k: int = 6
            schema_k: int = 4

        reranker: RerankerConfig = RerankerConfig()

    retrieval: RetrievalConfig = RetrievalConfig()

    class GenerationConfig(BaseTemplateConfig):
        model_name: str = settings.LLM_MODEL_ID
        model_temperature: float = 0.3

        fact_template_name: str = 'fact'
        schema_template_name: str = 'schema'
        general_template_name: str = 'general'

        resolved_fact_template_text: Optional[str] = None
        resolved_schema_template_text: Optional[str] = None
        resolved_general_template_text: Optional[str] = None

        @property
        def fact_template_text(self) -> str:
            return self._resolve_field("resolved_fact_template_text", self.fact_template_name)

        @property
        def schema_template_text(self) -> str:
            return self._resolve_field("resolved_schema_template_text", self.schema_template_name)

        @property
        def general_template_text(self) -> str:
            return self._resolve_field("resolved_general_template_text", self.general_template_name)

    generation: GenerationConfig = GenerationConfig()

    @model_validator(mode="after")
    def attach_context(self):
        self.preprocess._resolver = self.get_template_resolver()
        self.generation._resolver = self.get_template_resolver()
        return self

    def resolve(self):
        super().resolve()
        _ = self.preprocess.template_text
        _ = self.generation.schema_template_text
        _ = self.generation.fact_template_text
        _ = self.generation.general_template_text
        return self


RecursiveInferenceConfig = Annotated[
    Union[RecursiveV1InferenceConfig],
    Field(discriminator="version")
]

HierarchicalInferenceConfig = Annotated[
    Union[HierarchicalV1InferenceConfig],
    Field(discriminator="version")
]

InferenceConfig = Annotated[
    Union[RecursiveInferenceConfig, HierarchicalInferenceConfig],
    Field(discriminator="strategy")
]
