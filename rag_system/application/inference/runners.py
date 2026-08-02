from abc import ABC, abstractmethod

from rag_system.configs.inference import InferenceConfig, HierarchicalV1InferenceConfig, InferenceStrategy
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


def get_runner(config: InferenceConfig) -> InferenceRunner:
    if config.strategy == InferenceStrategy.HIERARCHICAL and config.version == "1.0":
        return HierarchicalV1InferenceRunner(config)

    raise NotImplementedError
