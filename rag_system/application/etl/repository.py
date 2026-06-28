from rag_system.domain import Document

def get_document_hashes() -> dict[str, str]:
    docs = Document.find_all().run()
    return {doc.absolute_path: doc.hash for doc in docs}