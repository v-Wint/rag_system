from zenml import pipeline, step
from zenml.logger import get_logger

from rag_system.application.inference import build_graph, RAGState, RAGConfig
from rag_system.infrastructure import VectorStore, Embedder, CrossEncoder

from steps.inference import inference_step

@pipeline(enable_cache=False)
def inference_pipeline(query: str, config: RAGConfig) -> str:
    return inference_step(config, query)


def warmup(config: RAGConfig):
    embedder = Embedder.from_pretrained(config.embedding_model)
    store = VectorStore.from_collection_name(config.collection_name, embedder, create_if_missing=False)
    cross_encoder = CrossEncoder.from_pretrained(config.cross_encoder_model)
    embedder.embed_query("warmup")


if __name__ == "__main__":
    config=RAGConfig(collection_name="chunks__remnote_v1__hierarchical_v1__intfloat_multilingual_e5_base__510__2048")
    warmup(config)

    run = inference_pipeline(
        query="What is encapsulation", 
        config=config
    )

    run = inference_pipeline(
        query="How to write sliding window in SQL?", 
        config=config
    )
