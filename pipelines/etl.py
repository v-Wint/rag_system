from pathlib import Path
from typing import Annotated
from zenml import pipeline

from rag_system.domain import Document

from steps.etl import crawl_document_paths_step, sync_warehouse_step

@pipeline
def etl_pipeline(data_dir: str | Path) -> tuple[
    Annotated[list[Document], "upserted_documents"],
    Annotated[list[str], "deleted_paths_relative"]
]:
    absolute_paths = crawl_document_paths_step(data_dir)
    return sync_warehouse_step(data_dir, absolute_paths)

if __name__ == '__main__':
    etl_pipeline('data')
