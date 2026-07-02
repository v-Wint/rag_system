from typing import Optional
from rag_system.domain import Document


def load_documents(document_ids: Optional[list[str]] = None) -> list[Document]:
    if document_ids:
        return Document.find({"_id": {"$in": document_ids}}).to_list()
    return Document.find_all().to_list()
