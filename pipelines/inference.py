from zenml import pipeline

from rag_system.domain import RAGConfig
from rag_system.infrastructure import VectorStore, Embedder, CrossEncoder

from steps.inference import inference_step

@pipeline(enable_cache=False)
def inference_pipeline(queries: list[str], config: RAGConfig):
    for query in queries:
        inference_step(config, query)


def warmup(config: RAGConfig):
    embedder = Embedder.from_pretrained(config.embedding_model)
    VectorStore.from_collection_name(config.collection_name, embedder, create_if_missing=False)
    CrossEncoder.from_pretrained(config.cross_encoder_model)
    embedder.embed_query("warmup")


if __name__ == "__main__":
    config=RAGConfig(
        collection_name="chunks__remnote_v1__hierarchical_v1__intfloat_multilingual_e5_base__510__2048",
    )
    
    warmup(config)

    inference_pipeline(
        ["what is encapsulation?", "S Q L ?"],
        config=config
    )
