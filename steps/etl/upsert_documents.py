from typing import Annotated

from zenml import step, log_metadata
from loguru import logger

from rag_system.domain import Document

@step(enable_cache=False)
def upsert_documents_step(
    documents: list[Document]
) -> tuple[
    Annotated[int, "modified_num"],
    Annotated[int, "created_num"],
]:
    result = Document.bulk_upsert(documents)
    modified = created = 0
    if result:
        modified = result.modified_count
        created = result.upserted_count

    logger.info(f"Upserting {len(documents)} documents")
    log_metadata({
        "modified": modified,
        "created": created
    })
    logger.info(f"Modified: {modified}, created: {created}")

    return modified, created
