import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from rag_system.settings import settings
from rag_system.domain import ChunkDocument
from rag_system.utils import get_hash


_qdrant_client: QdrantClient | None = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=f"http://{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}"
        )
    return _qdrant_client


class VectorStore(QdrantVectorStore):
    _instances: dict[str, "VectorStore"] = {}

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

    @classmethod
    def from_collection_name(cls, collection_name: str, embedding) -> "VectorStore":
        if collection_name not in cls._instances:
            cls._instances[collection_name] = cls(collection_name, embedding)
        return cls._instances[collection_name]

    def get_all_path_hash_pairs(self) -> dict[str, str]:
        if not self.client.collection_exists(collection_name=self.collection_name):
            return {}
        
        path_hash_pairs = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=1000,
                offset=offset,
            )
            if not points:
                break
            for point in points:
                payload = point.payload
                if not payload:
                    continue
                metadata = payload.get('metadata')
                if metadata and "doc_path" in metadata and "doc_hash" in metadata:
                    path_hash_pairs[metadata["doc_path"]] = metadata["doc_hash"]
            if offset is None:
                break
        return path_hash_pairs

    def chunk_to_doc(self, cd: ChunkDocument):
        doc_hash = cd.doc_hash
        chunk = cd.chunk
        chunk_id = str(uuid.UUID(get_hash(chunk.embedding_text + chunk.title + doc_hash)))

        return Document(
            page_content=chunk.embedding_text,
            id=chunk_id,
            metadata={
                "content": chunk.content,
                "title": chunk.title,
                "doc_hash": doc_hash,
                "doc_path": "/".join(chunk.doc_path),
                "abs_path": "/".join(chunk.abs_path),
                "rel_path": "/".join(chunk.rel_path),
            }
        )

    def add_chunks(self, cds: list[ChunkDocument], **kwargs):
        documents = [self.chunk_to_doc(cd) for cd in cds]
        return self.add_documents(documents, **kwargs)

    def delete_by_relative_paths(self, relative_paths: list[str]) -> None:
        if not relative_paths:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(
                            key="metadata.doc_path",
                            match=models.MatchAny(any=relative_paths),
                        )])
            ),
        )

    def upsert_chunks(self, cds: list[ChunkDocument], **kwargs):
        if not cds:
            return

        paths_to_sweep = list({"/".join(cd.chunk.doc_path) for cd in cds})
        self.delete_by_relative_paths(paths_to_sweep)
        return self.add_chunks(cds, **kwargs)
