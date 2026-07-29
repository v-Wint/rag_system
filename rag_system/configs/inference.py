from typing import Optional, Literal, Union, Annotated, Protocol, TypeVar, Generic
from typing import Protocol, runtime_checkable, Mapping
from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator, Field

from rag_system.settings import settings
from rag_system.infrastructure import PromptStore

from .chunking import ChunkingConfig, RecursiveV1Config, HierarchicalV1Config, HierarchicalConfig, BaseChunkingConfig
from .enums import InferenceStrategy


StrategyT = TypeVar("StrategyT", bound=InferenceStrategy)
VersionT = TypeVar("VersionT", bound=str)
ChunkingT = TypeVar("ChunkingT", bound=BaseChunkingConfig)


@runtime_checkable
class HasTemplates(Protocol):
    def template_names(self) -> dict[str, str]: ...


class BaseInferenceConfig(BaseModel, Generic[StrategyT, VersionT, ChunkingT]):
    strategy: StrategyT
    version: VersionT

    chunking: ChunkingT

    def load_prompts(self, store: PromptStore) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if isinstance(value, HasTemplates):
                for key, tmpl_name in value.template_names().items():
                    resolved[f"{field_name}.{key}"] = store.load_one(self.strategy, self.version, tmpl_name)
        return resolved


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

    class GenerationConfig(BaseModel):
        template_name: str = 'template'
        model_name: str = settings.LLM_MODEL_ID
        model_temperature: float = 0.3

        def template_names(self) -> dict[str, str]:
            return {'template': self.template_name}

    generation: GenerationConfig = GenerationConfig()


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

    class PreprocessConfig(BaseModel):
        template_name: str = 'preprocess'
        model_name: str = settings.PREPROCESSING_MODEL_ID
        model_temperature: float = 0.1

        def template_names(self) -> dict[str, str]:
            return {'template': self.template_name}

    preprocess: PreprocessConfig = PreprocessConfig()

    class RetrievalConfig(BaseModel):
        class VectorSearchConfig(BaseModel):
            fact_k: int = 20
            schema_k: int = 10

        vector: VectorSearchConfig = VectorSearchConfig()

        class RerankerConfig(BaseModel):
            model_name: str = settings.CROSS_ENCODER_MODEL_ID
            fact_k: int = 6
            schema_k: int = 2

        reranker: RerankerConfig = RerankerConfig()

    retrieval: RetrievalConfig = RetrievalConfig()

    class GenerationConfig(BaseModel):
        model_name: str = settings.LLM_MODEL_ID
        model_temperature: float = 0.3

        fact_template_name: str = 'fact'
        schema_template_name: str = 'schema'
        general_template_name: str = 'general'

        def template_names(self) -> dict[str, str]:
            return {
                'fact': self.fact_template_name,
                'schema': self.schema_template_name,
                'general': self.general_template_name
            }

    generation: GenerationConfig = GenerationConfig()


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
