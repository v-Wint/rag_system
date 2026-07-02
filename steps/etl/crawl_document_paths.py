from collections import Counter
from pathlib import Path
from typing_extensions import Annotated
from loguru import logger
from zenml import step, log_metadata
from rag_system.application.etl import crawl_document_paths

@step(enable_cache=False)
def crawl_document_paths_step(data_dir: str) -> Annotated[list[str], "document_paths"]:
    """Crawl the given directory for relevant documents and return their paths."""
    logger.info(f"Crawling documents in: {data_dir}")
    paths = crawl_document_paths(data_dir)
    metadata = {
        "file_count": len(paths),
        "root_directory": str(Path(data_dir).resolve()),
        "extensions_found": dict(Counter(p.suffix for p in paths)),
    }
    log_metadata(metadata=metadata, infer_artifact=True)
    logger.info(f"Found {len(paths)} document(s).")
    return [p.as_posix() for p in paths]
