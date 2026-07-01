from pathlib import Path
from zenml import pipeline
from steps.etl import crawl_document_paths_step, sync_warehouse_step

@pipeline
def etl_pipeline(data_dir: str | Path):
    absolute_paths = crawl_document_paths_step(data_dir)
    sync_warehouse_step(data_dir, absolute_paths)

if __name__ == '__main__':
    etl_pipeline('data')
