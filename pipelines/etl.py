from pathlib import Path
from zenml import pipeline
from steps.etl import crawl_document_paths_step, extract_documents_step

@pipeline(enable_cache=False)
def etl_pipeline(data_dir: str | Path):
    paths = crawl_document_paths_step(data_dir)
    extract_documents_step(data_dir, paths)

if __name__ == '__main__':
    etl_pipeline('data')
