from typing import Annotated

from zenml import step, log_metadata
from loguru import logger

from rag_system.domain import Document

@step(enable_cache=False)
def delete_docs_by_path_step(
    paths_to_delete: list[str]
) -> Annotated[int, "deleted_count"]:
    if not paths_to_delete:
        return 0
    logger.info(f"Deleting {len(paths_to_delete)} paths")
    
    result = Document.bulk_delete_by_paths(paths_to_delete)
    deleted = result.deleted_count if result else 0

    log_metadata({
        "deleted": deleted
    })
    logger.info(f"Deleted: {deleted}")

    return deleted
