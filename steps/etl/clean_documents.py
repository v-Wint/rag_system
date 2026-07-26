from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import step
from rag_system.domain import Document
from rag_system.application.etl import clean_text


@step
def clean_documents_step(
    documents: list[Document],
) -> Annotated[list[Document], "cleaned_documents"]:
    if not documents:
        return []
    logger.info(f"Cleaning {len(documents)} document(s)")
    cleaned = [doc.model_copy(update={"text": clean_text(doc.text)}) for doc in tqdm(documents)]
    logger.info(f"Cleaned {len(cleaned)} document(s).")
    return cleaned
