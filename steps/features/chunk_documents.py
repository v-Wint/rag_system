from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import step, log_metadata
from rag_system.domain import Document, SchemaNode, ChunkDocument
from rag_system.domain.schema import unite_schemas
from rag_system.application.features import chunk_document, ChunkingMethod
from rag_system.infrastructure import Embedder


@step
def chunk_documents_step(
    documents: list[Document],
    chunking_method: ChunkingMethod,
    embedding_model: str,
    max_tokens: int
) -> tuple[
    Annotated[list[ChunkDocument], "chunks"],
    Annotated[SchemaNode | None, "document_schema"]
]:
    if not documents:
        return [], None

    embedder = Embedder.from_pretrained(embedding_model)

    logger.info(
        f"Chunking {len(documents)} documents with method={chunking_method}, "
        f"embedding_model={embedding_model}, max_tokens={max_tokens}"
    )

    chunk_list: list[ChunkDocument] = []
    schema_list: list[SchemaNode] = []

    for document in tqdm(documents, desc="Chunking documents"):
        chunks, schema = chunk_document(
            document.text,
            document.relative_path,
            chunking_method,
            get_token_count=lambda s: embedder.get_token_count(s),
            max_tokens=max_tokens
        )
        cds = [ChunkDocument(chunk=chunk, doc_hash=document.hash) for chunk in chunks]
        chunk_list += cds
        if schema:
            schema_list.append(schema)

    logger.info(f"Produced {len(chunk_list)} chunks from {len(documents)} documents")

    log_metadata(
        metadata={
            "chunking_method": chunking_method,
            "embedding_model": embedding_model,
            "max_tokens": max_tokens,
            "num_documents": len(documents),
            "num_chunks": len(chunk_list),
        },
        artifact_name="chunks",
        infer_artifact=True
    )

    return chunk_list, unite_schemas(schema_list)
