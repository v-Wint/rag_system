from pathlib import Path
from typing_extensions import Annotated
from loguru import logger
from zenml import step, log_metadata

from rag_system.domain import Document
from rag_system.infrastructure.db import mongo_init
from rag_system.application.etl import reconcile


@step(enable_cache=False)
def sync_warehouse_step(data_dir: Path | str, absolute_paths: list[str]) -> Annotated[list[Document], "upserted_documents"]:
    logger.info(f"Reconciling {len(absolute_paths)} files against warehouse")

    mongo_init()

    warehouse_hashes = Document.get_all_path_hash_pairs()
    logger.info(f"Loaded {len(warehouse_hashes)} documents from warehouse")

    new_files, changed_files, unchanged_files, to_delete = reconcile(warehouse_hashes, absolute_paths, data_dir)

    to_upsert = new_files + changed_files
    Document.bulk_delete_by_paths(to_delete)

    upserted_result = Document.bulk_upsert(to_upsert)
    modified = upserted_result.modified_count if upserted_result else 0
    created = upserted_result.upserted_count if upserted_result else 0

    logger.info(f"Added/Updated: {len(to_upsert)}, unchanged: {len(unchanged_files)}, deleted: {len(to_delete)}")

    log_metadata({
        "total_on_disk": len(absolute_paths),
        "total_in_warehouse": len(warehouse_hashes),
        "deleted": len(to_delete),
        "upserted": len(to_upsert),
        "modified": modified,
        "created": created,
        "unchanged": len(unchanged_files),
        "new_paths": [d.path for d in new_files],
        "changed_paths": [d.path for d in changed_files],
        "deleted_paths": to_delete,
    })

    return to_upsert
