from zenml import step, log_metadata
from loguru import logger
from tqdm import tqdm

from langchain_core.utils.iter import batch_iterate

from rag_system.infrastructure import mongo_init, Embedder, VectorStore
from rag_system.domain import KBTree
from rag_system.application.features.chunking.hierarchical import flatten_leaves
from rag_system.configs.enums import SizeMetric


@step(enable_cache=False)
def embed_hierarchical_chunks_step(
    size_metric: SizeMetric,
    max_size: int,
    embedding_model: str,
    collection_name: str,
    structure_changed: bool,
    batch_size: int = 200
) -> None:
    """Embed the hierarchical leaf chunks from the KBTree into a collection.

    If the collection's path/hash map differs from the tree's (any doc change,
    fresh collection, or a new config reusing the tree), clears the whole
    collection and re-embeds every leaf. Hierarchical-specific.
    """
    mongo_init()

    tree = KBTree.load(size_metric, max_size)
    if tree is None:
        logger.info(
            f"No KB tree for size_metric={size_metric}, max_size={max_size}; nothing to embed."
        )
        return

    tree_hashes = tree.get_doc_hash_map()
    store_hashes = VectorStore.get_all_path_hash_pairs(collection_name)

    if store_hashes == tree_hashes:
        logger.info(f"Collection '{collection_name}' is up to date; skipping.")
        return

    if not structure_changed:
        logger.info(
            f"Collection '{collection_name}' differs from KB tree despite no structural "
            f"change (e.g. fresh config sharing an existing tree); clearing and re-embedding."
        )

    logger.info(
        f"Collection '{collection_name}' differs from KB tree; clearing and re-embedding "
        f"({len(tree_hashes)} docs)"
    )

    embedder = Embedder.from_pretrained(embedding_model)
    store = VectorStore.for_indexing(collection_name=collection_name, embedding=embedder)
    store.clear()

    documents = flatten_leaves(tree.root.children)
    for batch in tqdm(list(batch_iterate(batch_size, documents))):
        store.add_documents(batch)

    logger.info(f"Embedded {len(documents)} chunks into collection '{collection_name}'.")

    log_metadata({
        "collection_name": collection_name,
        "num_documents": len(tree_hashes),
        "num_chunks": len(documents),
        "clear_and_reembed": True,
    })
