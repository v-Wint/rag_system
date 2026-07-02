from zenml import step, log_metadata
from loguru import logger
from tqdm import tqdm

from langchain_core.utils.iter import batch_iterate

from rag_system.domain import Chunk
from rag_system.infrastructure import Embedder, VectorStore
    
@step
def embed_load_chunks_step(
    chunks: list[Chunk],
    embedding_model: str,
    collection_name: str,
    batch_size: int = 200
) -> None:
    embedder = Embedder.from_pretrained(embedding_model)
    store = VectorStore(collection_name=collection_name, embedding=embedder)

    logger.info(
        f"Saving {len(chunks)} chunks with embedding_model={embedding_model}, collection_name={collection_name}"
    )

    for batch in tqdm(list(batch_iterate(batch_size, chunks))):
        store.add_chunks(batch)

    logger.info(f"Saved {len(chunks)} chunks to collection '{collection_name}'.")

    log_metadata(
        metadata={
            "embedding_model": embedding_model,
            "collection_name": collection_name,
            "num_chunks": len(chunks),
        },
    )

