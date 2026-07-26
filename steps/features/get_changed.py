from typing import Optional
from loguru import logger
from typing_extensions import Annotated
from zenml import step, log_metadata

from rag_system.infrastructure import mongo_init, VectorStore, Embedder
from rag_system.domain import Document
from rag_system.application.features import reconcile


@step(enable_cache=False)
def get_changed_step(
    embedding_model: str,
    collection_name: str
) -> tuple[
    Annotated[list[Document], "raw_documents"],
    Annotated[Optional[list[str]], "deleted_paths_relative"]
]:
    mongo_init()

    logger.info("Reconciling warehouse and feature store.")
    documents = Document.find_all().to_list()
    logger.info(f"Loaded {len(documents)} documents from warehouse")

    store_hashes = VectorStore.from_collection_name(collection_name, Embedder.from_pretrained(embedding_model)).get_all_path_hash_pairs()
    logger.info(f"Loaded {len(store_hashes)} documents from feature store")

    new_docs, changed_docs, unchanged_docs, to_delete = reconcile(store_hashes, documents)

    logger.info(f"New documents: {len(new_docs)}, modified: {len(changed_docs)}, unchanged: {len(unchanged_docs)}, deleted: {len(to_delete)}")


    log_metadata({
        "total_in_warehouse": len(documents),
        "total_in_store": len(store_hashes),
        "new": len(new_docs),
        "modified": len(changed_docs),
        "unchanged": len(unchanged_docs),
        "deleted": len(to_delete),
        "new_paths": [doc.relative_path for doc in new_docs],
        "deleted_paths": to_delete
    })

    return new_docs + changed_docs, to_delete
