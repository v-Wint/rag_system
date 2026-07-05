from zenml import pipeline, step
from zenml.logger import get_logger

from rag_system.application.inference import build_graph, RAGState, RAGConfig

from steps.inference import inference_step

@pipeline(enable_cache=False)
def inference_pipeline(query: str, config: RAGConfig) -> str:
    return inference_step(config, query)


if __name__ == "__main__":
    run = inference_pipeline(
        query="Top 10 most usefull docker commands", 
        config=RAGConfig(collection_name="chunks__remnote_v1__hierarchical_v1__intfloat_multilingual_e5_base__510__2048")
    )
