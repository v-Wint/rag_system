from sentence_transformers import CrossEncoder
from rag_system.infrastructure import VectorStore, Embedder
from ..state import RAGState


def make_retrieve_node(
    embedding_model: str,
    collection_name: str,
    vector_retrieval_size: int,
    cross_encoder_model: str,
    reranking_size: int,
):
    embedder = Embedder(embedding_model)
    store = VectorStore.from_collection_name(collection_name, embedder, False)
    base_retriever = store.as_retriever(search_kwargs={"k": vector_retrieval_size})
    cross_encoder = CrossEncoder(cross_encoder_model)

    def retrieve_node(state: RAGState) -> dict:
        candidates = base_retriever.invoke(state.improved_query or state.query)

        pairs = [(state.query, doc.page_content) for doc in candidates]
        scores = cross_encoder.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top = ranked[:reranking_size]

        retrieval_debug = {
            "candidates": [
                {
                    "id": doc.metadata.get("id"),
                    "score": float(score),
                    "preview": doc.page_content[:120],
                }
                for doc, score in ranked
            ],
            "selected_ids": [doc.metadata.get("id") for doc, _ in top],
        }

        return {
            "retrieved_chunks": [doc.page_content for doc, _ in top],
            "retrieval_debug": retrieval_debug,
        }

    return retrieve_node
