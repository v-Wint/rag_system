from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import step, log_metadata
from rag_system.domain import Document
from rag_system.application.features import clean_text, CleaningMethod
from rag_system.infrastructure import mongo_init


@step
def clean_documents_step(
    documents: list[Document],
    cleaning_method: CleaningMethod,
) -> Annotated[list[Document], "cleaned_documents"]:
    if not documents:
        return []
    logger.info(f"Cleaning {len(documents)} document(s) with method '{cleaning_method}'.")
    cleaned = [doc.model_copy(update={"text": clean_text(doc.text, cleaning_method)}) for doc in tqdm(documents)]
    log_metadata(
        metadata={
            "cleaning_method": cleaning_method.value,
            "document_count": len(cleaned),
        },
        infer_artifact=True,
    )
    logger.info(f"Cleaned {len(cleaned)} document(s).")
    return cleaned
