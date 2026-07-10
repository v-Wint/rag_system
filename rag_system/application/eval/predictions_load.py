from rag_system.utils import get_hash

from rag_system.domain import EvalPrediction, RAGConfig


def load_predictions(dataset_name: str, config: RAGConfig) -> list[EvalPrediction]:
    return EvalPrediction.find(
        EvalPrediction.dataset_name == dataset_name,
        EvalPrediction.config_hash == get_hash(config.model_dump_json())
    ).to_list()
