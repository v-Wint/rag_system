from loguru import logger
from typing_extensions import Annotated
from zenml import step, log_metadata

from rag_system.infrastructure import mongo_init
from rag_system.domain import Document, KBTree
from rag_system.application.features import reconcile
from rag_system.configs.enums import SizeMetric


@step(enable_cache=False)
def reconcile_against_tree_step(
    size_metric: SizeMetric,
    max_size: int
) -> tuple[
    Annotated[list[Document], "documents_to_build"],
    Annotated[list[str], "deleted_paths"]
]:
    """Reconcile warehouse documents against the KBTree's stored doc hashes.

    Returns docs whose subtrees must be (re)built and the doc paths to drop
    from the tree (present in the tree but gone from the warehouse).
    """
    mongo_init()

    logger.info("Reconciling warehouse and KB tree.")
    documents = Document.find_all().to_list()
    logger.info(f"Loaded {len(documents)} documents from warehouse")

    tree = KBTree.load(size_metric, max_size)
    tree_hashes = tree.get_doc_hash_map() if tree else {}
    logger.info(f"Loaded {len(tree_hashes)} documents from KB tree")

    new_docs, changed_docs, unchanged_docs, to_delete = reconcile(tree_hashes, documents)

    logger.info(
        f"New documents: {len(new_docs)}, modified: {len(changed_docs)}, "
        f"unchanged: {len(unchanged_docs)}, deleted: {len(to_delete)}"
    )

    log_metadata({
        "total_in_warehouse": len(documents),
        "total_in_tree": len(tree_hashes),
        "new": len(new_docs),
        "modified": len(changed_docs),
        "unchanged": len(unchanged_docs),
        "deleted": len(to_delete),
        "new_paths": [doc.relative_path for doc in new_docs],
        "deleted_paths": to_delete
    })

    return new_docs + changed_docs, to_delete
