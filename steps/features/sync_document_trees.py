from typing import Annotated

from zenml import step, log_metadata
from loguru import logger

from rag_system.domain import DocumentTree
from rag_system.infrastructure import mongo_init

@step(enable_cache=False)
def sync_document_trees_step(
    trees: list[DocumentTree],
    to_delete_rel: list[str],
    config_slug: str
) -> Annotated[bool, "structure_changed"]:
    if not trees and not to_delete_rel:
        logger.info(f"No changes in document trees for config '{config_slug}'; skipping.")
        return False

    mongo_init()

    modified = created = deleted = 0

    if trees:
        logger.info(f"Upserting {len(trees)} trees")
        upsert_result = DocumentTree.bulk_upsert(trees)
        if upsert_result:
            modified = upsert_result.modified_count
            created = upsert_result.upserted_count

    if to_delete_rel:
        logger.info(f"Deleting {len(to_delete_rel)} trees")
        delete_result = DocumentTree.delete_by_paths(config_slug, to_delete_rel)

        if delete_result:
            deleted = delete_result.deleted_count

    log_metadata({
        "modified": modified,
        "created": created,
        "deleted": deleted
    })

    logger.info(f"Modified: {modified}, created: {created}, deleted: {deleted}")

    return bool(modified + created + deleted)
    