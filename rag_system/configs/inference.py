from typing import Literal, Union, Annotated, TypeVar, Generic, Optional, Callable, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator, TypeAdapter
import yaml


from .chunking import ChunkingConfig, RecursiveV1Config, HierarchicalV1Config, HierarchicalConfig, AgenticConfig, AgenticV1Config, BaseChunkingConfig
from .enums import InferenceStrategy

from rag_system.settings import settings


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
        from rag_system.infrastructure import PromptStore
        return lambda name: PromptStore().load_one(
            strategy=self.strategy,
            version=self.version,
            name=name
        )

    @model_validator(mode="after")
    def _mark_discriminators_set(self) -> "BaseInferenceConfig":
        # Force these into model_fields_set regardless of how the
        # instance was constructed, so exclude_unset never drops them.
        self.__pydantic_fields_set__ |= {"strategy", "version"}
        return self

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        params = {
            'strategy': self.strategy,
            'version': self.version,
        }
        chunking_params = self.chunking.get_params()
        params.update({'chunking.' + k: v for k, v in chunking_params.items()})
        return params


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
        k: int = 8

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

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({
            'retrieval.k': self.retrieval.k
        })
        params.update(
            {'generation.' + k: v for k, v in self.generation.model_dump().items() if not k.endswith('template_text')}
        )
        return params


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
            fact_k: int = 8
            schema_k: int = 5

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
        self.chunking.resolve()
        _ = self.preprocess.template_text
        _ = self.generation.schema_template_text
        _ = self.generation.fact_template_text
        _ = self.generation.general_template_text
        return self

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {'preprocess.' + k: v for k, v in self.preprocess.model_dump().items() if not k.endswith('template_text')}
        )
        params.update(
            {'retrieval.vector' + k: v for k, v in self.retrieval.vector.model_dump().items()}
        )
        params.update(
            {'retrieval.reranker' + k: v for k, v in self.retrieval.reranker.model_dump().items()}
        )
        params.update(
            {'generation.' + k: v for k, v in self.generation.model_dump().items() if not k.endswith('template_text')}
        )
        return params


class AgenticV1InferenceConfig(
    BaseInferenceConfig[
        Literal[InferenceStrategy.AGENTIC],
        Literal["1.0"],
        AgenticConfig,
    ]
):
    strategy: Literal[InferenceStrategy.AGENTIC] = InferenceStrategy.AGENTIC
    version: Literal["1.0"] = "1.0"
    chunking: AgenticConfig = AgenticV1Config()

    class GenerationConfig(BaseTemplateConfig):
        template_name: str = 'agent_template'
        model_name: str = settings.LLM_MODEL_ID
        model_temperature: float = 0.3

        resolved_template_text: Optional[str] = None

        @property
        def template_text(self) -> str:
            return self._resolve_field("resolved_template_text", self.template_name)

    generation: GenerationConfig = GenerationConfig()

    @model_validator(mode="after")
    def attach_context(self) -> "AgenticV1InferenceConfig":
        self.generation._resolver = self.get_template_resolver()
        return self

    def resolve(self):
        super().resolve()
        self.chunking.resolve()
        _ = self.generation.template_text
        return self

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {'generation.' + k: v for k, v in self.generation.model_dump().items() if not k.endswith('template_text')}
        )
        return params


RecursiveInferenceConfig = Annotated[
    Union[RecursiveV1InferenceConfig],
    Field(discriminator="version")
]

HierarchicalInferenceConfig = Annotated[
    Union[HierarchicalV1InferenceConfig],
    Field(discriminator="version")
]

AgenticInferenceConfig = Annotated[
    Union[AgenticV1InferenceConfig],
    Field(discriminator="version")
]

InferenceConfig = Annotated[
    Union[RecursiveInferenceConfig, HierarchicalInferenceConfig, AgenticInferenceConfig],
    Field(discriminator="strategy")
]


def load_from_yaml(yaml_path: str) -> InferenceConfig:
    with open(yaml_path, "r") as f:
        raw_data = yaml.safe_load(f)
    return TypeAdapter(InferenceConfig).validate_python(raw_data)


def from_argument():
    import sys
    if len(sys.argv) >= 2:
        return load_from_yaml(sys.argv[1])
    return None
