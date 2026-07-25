from pathlib import Path

from zenml import step, log_metadata
from typing_extensions import Annotated
from loguru import logger

from rag_system.domain import Document
from rag_system.infrastructure import mongo_init
from rag_system.application.etl import reconcile


@step(enable_cache=False)
def reconcile_against_warehouse_step(
    data_dir: Path | str,
    absolute_paths: list[str]
) -> tuple[
    Annotated[list[Document], "docs_to_upsert"],
    Annotated[list[str], "abs_paths_to_delete"]
]:
    logger.info(f"Reconciling {len(absolute_paths)} files against warehouse")
    mongo_init()
    data_dir = Path(data_dir).resolve()

    warehouse_hashes = Document.get_all_path_hash_pairs()
    logger.info(f"Loaded {len(warehouse_hashes)} documents from warehouse")

    new_files, changed_files, unchanged_files, deleted_files = reconcile(warehouse_hashes, absolute_paths, data_dir)

    logger.info(f"Found new: {len(new_files)}, changed: {len(changed_files)}, unchanged: {len(unchanged_files)}, deleted: {len(deleted_files)}")

    log_metadata({
        "total_on_disk": len(absolute_paths),
        "total_in_warehouse": len(warehouse_hashes),
        "new_files": len(new_files),
        "changed_files": len(changed_files),
        "unchanged_files": len(unchanged_files),
        "to_delete": len(deleted_files),

        "new_paths": [d.relative_path for d in new_files],
        "changed_paths": [d.relative_path for d in changed_files],
        "paths_to_delete": deleted_files,
    })

    return new_files + changed_files, deleted_files
