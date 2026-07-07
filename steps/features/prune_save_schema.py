from typing import Annotated

from zenml import step, log_metadata
from loguru import logger


from rag_system.application.features import reconcile_schema
from rag_system.domain import SchemaNode, SchemaText, Document
from rag_system.domain.schema import unite_schemas
from rag_system.infrastructure import Embedder, mongo_init

@step(enable_cache=False)
def prune_save_schema_step(
    schemas: list[SchemaNode],
    to_delete_rel: list[str],
    embedding_model: str,
    max_schema_tokens: int | None,
    collection_name: str
) -> Annotated[str, "truncated_schema"]:
    if not schemas:
        logger.info(f"No changed schemas for collection '{collection_name}'; skipping.")
        return ''

    mongo_init()

    if not max_schema_tokens:
        raise ValueError("max_schema_tokens is required to save schema")

    logger.info(
        f"Reconciling schema for collection '{collection_name}': "
        f"{len(schemas)} changed docs, {len(to_delete_rel)} deleted paths"
    )

    old_schema = SchemaText.find_one(SchemaText.collection == collection_name).run()
    schema_node, n_added, n_replaced, n_deleted = reconcile_schema(
        old_root=old_schema.root if old_schema else None,
        schemas=schemas,
        deleted_rel_paths=to_delete_rel,
    )
    logger.info(f"Reconciled: {n_added} added, {n_replaced} replaced, {n_deleted} deleted")


    embedder = Embedder.from_pretrained(embedding_model)
    depth = 0
    while embedder.get_token_count(str(schema_node.truncated(depth))) < max_schema_tokens:
        depth += 1
    if depth == 0:
        raise ValueError("Schema exceeds max_schema_tokens even at depth 0; truncating to root only")
    

    schema_truncated = schema_node.truncated(depth - 1)
    token_count = embedder.get_token_count(str(schema_truncated))
    logger.info(
        f"Truncated schema to depth={depth - 1}, "
        f"{token_count}/{max_schema_tokens} tokens"
    )

    schema_text = SchemaText(
        root=schema_node,
        text=str(schema_truncated),
        collection=collection_name
    )
    schema_text.upsert()
    logger.info(f"Saved schema for collection '{collection_name}'")


    log_metadata(
        metadata={
            "collection": collection_name,
            "max_depth": depth - 1,
            "truncated_token_count": token_count,
            "max_schema_tokens": max_schema_tokens,
            "docs_added": n_added,
            "docs_replaced": n_replaced,
            "docs_deleted": n_deleted,
            "total_children": len(schema_node.children),
        },
        infer_artifact=True
    )
    return schema_text.text
