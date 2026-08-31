from loguru import logger
from typing_extensions import Annotated
from zenml import step, log_metadata

from rag_system.infrastructure import mongo_init, VectorStore
from rag_system.domain import Document
from rag_system.application.features import reconcile


@step(enable_cache=False)
def reconcile_against_vector_store_step(
    collection_name: str
) -> tuple[
    Annotated[list[Document], "raw_documents"],
    Annotated[list[str], "deleted_paths_relative"]
]:
    """Reconcile warehouse documents against the collection's path/hash map.

    Returns docs needing (re)chunking and the stale relative paths to delete.
    Recursive baseline only (no KBTree involved).
    """
    mongo_init()

    logger.info("Reconciling warehouse and feature store.")
    documents = Document.find_all().to_list()
    logger.info(f"Loaded {len(documents)} documents from warehouse")

    if VectorStore.collection_exists(collection_name):
        store_hashes = VectorStore.get_all_path_hash_pairs(collection_name)
    else:
        logger.info("No hashes in the feature store")
        return documents, []

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
