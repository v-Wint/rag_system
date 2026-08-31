from typing import Annotated

from zenml import step, log_metadata
from loguru import logger

from rag_system.application.features.tree import prune_tree
from rag_system.domain import KBTree, DocNode, SchemaString
from rag_system.infrastructure import Embedder, mongo_init
from rag_system.configs.enums import SizeMetric


@step(enable_cache=False)
def prune_save_schema_step(
    size_metric: SizeMetric,
    max_size: int,
    max_schema_size: int,
    embedding_model: str,
    structure_changed: bool
) -> Annotated[str, "truncated_schema"]:
    """Unite the KBTree by title, prune to the schema token budget, persist.

    Short-circuits when structure is unchanged and a SchemaString already exists
    for (size_metric, max_size, max_schema_size). Hierarchical-specific.
    """
    mongo_init()

    if not structure_changed and SchemaString.load(size_metric, max_size, max_schema_size):
        return ''

    tree = KBTree.load(size_metric, max_size)
    if tree is None:
        return ''

    united = DocNode.unite(tree.root.children)
    embedder = Embedder.from_pretrained(embedding_model)

    pruned = prune_tree(united, lambda s: embedder.get_token_count(s), max_schema_size)
    pruned_str = str(pruned)

    depth = pruned.height()
    tokens = embedder.get_token_count(pruned_str)

    logger.info(
        f"Truncated tree to depth={depth}, "
        f"{tokens}/{max_schema_size} tokens"
    )

    schema_string = SchemaString(
        text=pruned_str,
        size_metric=size_metric,
        max_size=max_size,
        max_schema_size=max_schema_size,
    )
    schema_string.upsert()
    logger.info(
        f"Saved schema for size_metric={size_metric}, max_size={max_size}"
    )

    log_metadata(
        metadata={
            "size_metric": size_metric,
            "max_size": max_size,
            "max_schema_size": max_schema_size,
            "max_depth": depth,
            "token_count": tokens,
            "total_children": len(pruned.children),
        },
        infer_artifact=True
    )
    return pruned_str
