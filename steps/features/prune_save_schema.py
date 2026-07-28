from typing import Annotated

from zenml import step, log_metadata
from loguru import logger

from rag_system.application.features import prune_tree
from rag_system.domain import DocumentTree, DocumentNode, SchemaString
from rag_system.infrastructure import Embedder, mongo_init


@step(enable_cache=False)
def prune_save_schema_step(
    config_slug: str,
    embedding_model: str,
    max_size: int,
    structure_changed: bool
) -> Annotated[str, "truncated_schema"]:
    mongo_init()

    if not structure_changed and SchemaString.load(config_slug, max_size):
        return ''

    trees = DocumentTree.load(config_slug)

    logger.info(f"Loaded {len(trees)} trees")

    if not trees:
        return ''

    united = DocumentNode.unite([tree.root for tree in trees])
    embedder = Embedder.from_pretrained(embedding_model)

    pruned = prune_tree(united, lambda s: embedder.get_token_count(s), max_size)
    pruned_str = str(pruned)

    depth = pruned.height()
    tokens = embedder.get_token_count(pruned_str)

    logger.info(
        f"Truncated tree to depth={depth}, "
        f"{tokens}/{max_size} tokens"
    )

    schema_string = SchemaString(text=pruned_str, max_size=max_size, config_slug=config_slug)
    schema_string.upsert()
    logger.info(f"Saved schema for config '{config_slug}'")

    log_metadata(
        metadata={
            "slug": config_slug,
            "max_depth": depth,
            "token_count": tokens,
            "max_size": max_size,
            "total_children": len(pruned.children),
        },
        infer_artifact=True
    )
    return pruned_str
