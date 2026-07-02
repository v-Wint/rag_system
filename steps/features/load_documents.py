from typing import Optional
from loguru import logger
from typing_extensions import Annotated
from zenml import step, log_metadata

from rag_system.infrastructure.db import mongo_init
from rag_system.application.features import load_documents
from rag_system.domain import Document


@step(enable_cache=False)
def load_documents_step(document_ids: Optional[list[str]] = None) -> Annotated[list[Document], "raw_documents"]:
    mongo_init()
    logger.info(f"Loading documents{f' by {len(document_ids)} ids' if document_ids else ' from database'}.")
    documents = load_documents(document_ids)
    log_metadata(metadata={"document_count": len(documents)}, infer_artifact=True)
    logger.info(f"Loaded {len(documents)} document(s).")
    return documents


if __name__ == '__main__':
    load_documents_step()
