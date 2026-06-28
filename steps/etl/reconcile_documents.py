from loguru import logger
from zenml import step, log_metadata
from typing_extensions import Annotated

from rag_system.infrastructure.db import mongo_init
from rag_system.application.etl import get_document_hashes, compute_files_hashes
from rag_system.domain.documents import Document


@step
def reconcile_documents_step(paths: list[str]) -> Annotated[list[str], "paths_to_upsert"]:
    mongo_init()

    logger.info(f"Reconciling {len(paths)} files against warehouse")

    warehouse_hashes = get_document_hashes()
    logger.info(f"Loaded {len(warehouse_hashes)} documents from warehouse")

    disk_hashes = compute_files_hashes(paths)

    to_delete = set(warehouse_hashes.keys()) - set(disk_hashes.keys())
    if to_delete:
        logger.info(f"Deleting {len(to_delete)} orphaned documents")
        Document.find({"absolute_path": {"$in": list(to_delete)}}).delete()

    to_upsert = [
        path for path, hash_ in disk_hashes.items()
        if warehouse_hashes.get(path) != hash_
    ]
    unchanged = len(disk_hashes) - len(to_upsert)

    logger.info(f"To upsert: {len(to_upsert)}, unchanged: {unchanged}")

    log_metadata({
        "total_on_disk": len(disk_hashes),
        "total_in_warehouse": len(warehouse_hashes),
        "deleted": len(to_delete),
        "to_upsert": len(to_upsert),
        "unchanged": unchanged,
    })

    return to_upsert
