from rag_system.infrastructure import VectorStore, Embedder, CrossEncoder
from rag_system.configs.chunking import ChunkingConfig
from rag_system.settings import settings


def make_retrieve_node(
    chunking_config: ChunkingConfig,
    vector_k: int,
    reranker_model_name: str,
    reranker_k: int,
    query_key: str = "query",
):
    embedder = Embedder.from_pretrained(chunking_config.embedding_model)
    store = VectorStore.for_retrieval(chunking_config.slug, embedder)
    base_retriever = store.as_retriever(search_kwargs={"k": vector_k})
    cross_encoder = CrossEncoder.from_pretrained(reranker_model_name, device=settings.TEXT_EMBEDDING_DEVICE, token=settings.HUGGINGFACE_ACCESS_TOKEN)

    def retrieve_node(state) -> dict:
        candidates = base_retriever.invoke(state[query_key])

        pairs = [(state[query_key], doc.page_content) for doc in candidates]
        scores = cross_encoder.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top = ranked[:reranker_k]

        debug = {
            "reranker_candidates": [
                {
                    "id": doc.metadata.get('_id'),
                    "score": float(score),
                    "preview": doc.page_content[:512],
                }
                for doc, score in ranked
            ],
            "config_slug": chunking_config.slug,
        }

        return {
            "retrieved_chunks": [doc.page_content for doc, _ in top],
            "debug": debug,
        }

    return retrieve_node
