from pathlib import Path
from typing import Annotated
from zenml import pipeline

from rag_system.domain import Document

from steps.etl import (
    crawl_document_paths_step, reconcile_against_warehouse_step,
    clean_documents_step, upsert_documents_step,
    delete_docs_by_path_step
)

@pipeline
def etl_pipeline(data_dir: str | Path) -> tuple[
    Annotated[list[Document], "modified_documets"],
    Annotated[list[str], "deleted_paths"]
]:
    abosulte_file_paths = crawl_document_paths_step(data_dir)
    docs_to_upsert, paths_to_delete = reconcile_against_warehouse_step(data_dir, abosulte_file_paths)

    docs_to_upsert = clean_documents_step(docs_to_upsert)
    upsert_documents_step(docs_to_upsert)

    delete_docs_by_path_step(paths_to_delete)

    return docs_to_upsert, paths_to_delete


if __name__ == '__main__':
    etl_pipeline('data')
