from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

from rag_system.configs.inference import (
    InferenceConfig,
    HierarchicalV1InferenceConfig,
    InferenceStrategy,
    RecursiveV1InferenceConfig,
    AgenticV1InferenceConfig,
)
from rag_system.domain import InferenceResult

class InferenceRunner(ABC):
    def __init__(self, config: InferenceConfig):
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def predict(self, query: str) -> InferenceResult:
        ...


class HierarchicalV1InferenceRunner(InferenceRunner):
    def __init__(self, config: HierarchicalV1InferenceConfig):
        self.config = config
        self._graph = None

    def setup(self):
        if not self._graph:
            from .strategies.hierarchical.v1.graph import build_graph
            self._graph = build_graph(self.config)

    def predict(self, query: str) -> InferenceResult:
        if self._graph is None:
            self.setup()
        result = self._graph.invoke({'query': query}) # type: ignore
        return InferenceResult(
            query=query,
            retrieved_chunks=result.get('retrieved_chunks'),
            answer=result['answer'],
            metadata={'question_type': result['question_type']}
        )

class RecursiveV1InferenceRunner(InferenceRunner):
    def __init__(self, config: RecursiveV1InferenceConfig):
        self.config = config
        self._chain = None

    def setup(self):
        if not self._chain:
            from .strategies.recursive.v1.chain import build_chain
            self._chain = build_chain(self.config)

    def predict(self, query: str) -> InferenceResult:
        if self._chain is None:
            self.setup()
        result = self._chain.invoke(query) # type: ignore
        return InferenceResult(
            query=query,
            retrieved_chunks=result.get('retrieved_chunks'),
            answer=result['answer'],
        )


class AgenticV1InferenceRunner(InferenceRunner):
    def __init__(self, config: AgenticV1InferenceConfig):
        self.config = config
        self._graph = None

    def setup(self):
        if not self._graph:
            from .strategies.agentic.v1.graph import build_graph
            self._graph = build_graph(self.config)

    def predict(self, query: str) -> InferenceResult:
        if self._graph is None:
            self.setup()
        result = self._graph.invoke({  # type: ignore
            'query': query,
            'messages': [
                SystemMessage(self.config.generation.template_text),
                HumanMessage(query),
            ],
            'iteration': 0,
        })
        return InferenceResult(
            query=query,
            retrieved_chunks=None,
            answer=result['answer'],
            metadata={
                'expanded_ids': [
                    id
                    for entry in result.get('debug', [])
                    for id in entry.get('expanded_ids', [])
                ],
            },
        )


def get_runner(config: InferenceConfig) -> InferenceRunner:
    if config.strategy == InferenceStrategy.HIERARCHICAL and config.version == "1.0":
        return HierarchicalV1InferenceRunner(config)

    if config.strategy == InferenceStrategy.RECURSIVE and config.version == "1.0":
        return RecursiveV1InferenceRunner(config)

    if config.strategy == InferenceStrategy.AGENTIC and config.version == "1.0":
        return AgenticV1InferenceRunner(config)

    raise NotImplementedError
