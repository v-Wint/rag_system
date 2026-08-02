from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from rag_system.settings import settings
from rag_system.domain import ChunkDocument
from rag_system.utils import get_hash
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

_qdrant_client: QdrantClient | None = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=f"http://{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}"
        )
    return _qdrant_client


class VectorStoreError(Exception):
    pass


class CollectionNotFoundError(VectorStoreError):
    pass


class VectorStore(QdrantVectorStore):
    _instances: dict[str, "VectorStore"] = {}

    def __init__(self, collection_name: str, embedding, create_if_missing: bool = False):
        client = get_qdrant_client()
        exists = client.collection_exists(collection_name=collection_name)

        if not exists:
            if create_if_missing:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
            else:
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' does not exist"
                )

        super().__init__(client=client, collection_name=collection_name, embedding=embedding)

    @classmethod
    def _get_or_create_instance(
        cls, collection_name: str, embedding, create_if_missing: bool
    ) -> "VectorStore":
        if collection_name not in cls._instances:
            cls._instances[collection_name] = cls(
                collection_name, embedding, create_if_missing=create_if_missing
            )
        return cls._instances[collection_name]

    @classmethod
    def for_indexing(cls, collection_name: str, embedding) -> "VectorStore":
        return cls._get_or_create_instance(collection_name, embedding, create_if_missing=True)

    @classmethod
    def for_retrieval(cls, collection_name: str, embedding) -> "VectorStore":
        return cls._get_or_create_instance(collection_name, embedding, create_if_missing=False)

    @classmethod
    def collection_exists(cls, collection_name: str) -> bool:
        client = get_qdrant_client()
        return client.collection_exists(collection_name=collection_name)

    @staticmethod
    def get_all_path_hash_pairs(collection_name) -> dict[str, str]:
        client = get_qdrant_client()
        if not client.collection_exists(collection_name=collection_name):
            return {}
        
        path_hash_pairs = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
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

    def add_chunks(self, cds: list[ChunkDocument], **kwargs):
        documents = [cd.to_document() for cd in cds]
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
