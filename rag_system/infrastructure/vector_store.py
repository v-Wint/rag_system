from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from rag_system.settings import settings
from rag_system.domain import Chunk


_qdrant_client: QdrantClient | None = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=f"http://{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}"
        )
    return _qdrant_client


class VectorStore(QdrantVectorStore):
    def __init__(self, collection_name, embedding):
        client = get_qdrant_client()
        
        if not client.collection_exists(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding.vector_size,
                    distance=Distance.COSINE
                ),
            )
            
        super().__init__(client=client, collection_name=collection_name, embedding=embedding)


    def chunk_to_doc(self, chunk: Chunk):
        return Document(
            page_content=chunk.embedding_text,
            metadata={
                "content": chunk.content,
                "embedding_text": chunk.embedding_text,
                "title": chunk.title,
                "doc_path": "/".join(chunk.doc_path),
                "abs_path": "/".join(chunk.abs_path),
                "rel_path": "/".join(chunk.rel_path)
            }
        )

    def add_chunks(self, chunks: list[Chunk], **kwargs):
        documents = [self.chunk_to_doc(chunk) for chunk in chunks]
        return self.add_documents(documents, **kwargs)
