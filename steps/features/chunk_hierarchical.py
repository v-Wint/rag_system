from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import step, log_metadata
from rag_system.domain import Document, SchemaNode, ChunkDocument
from rag_system.application.features.chunking.hierarchical import chunk_document_hierarchical
from rag_system.infrastructure import Embedder
from rag_system.configs.chunking import HierarchicalConfig


@step
def chunk_hierarchical_step(
    documents: list[Document],
    config: HierarchicalConfig
) -> tuple[
    Annotated[list[ChunkDocument], "chunks"],
    Annotated[list[SchemaNode], "document_schemas"]
]:
    if not documents:
        return [], []

    embedder = Embedder.from_pretrained(config.embedding_model)

    logger.info(
        f"Chunking {len(documents)} documents with {config.model_dump()}"
    )

    chunk_list: list[ChunkDocument] = []
    schema_list: list[SchemaNode] = []

    for document in tqdm(documents, desc="Chunking documents"):
        chunks, schema = chunk_document_hierarchical(
            document, config, lambda s: embedder.get_token_count(s),
        )
        cds = [ChunkDocument(chunk=chunk, doc_hash=document.hash) for chunk in chunks]
        chunk_list += cds
        if schema:
            schema_list.append(schema)

    logger.info(f"Produced {len(chunk_list)} chunks from {len(documents)} documents")

    metadata = config.model_dump()
    metadata.update({
        "num_documents": len(documents),
        "num_chunks": len(chunk_list),
    })

    log_metadata(
        metadata=metadata,
        artifact_name="chunks",
        infer_artifact=True
    )
    return chunk_list, schema_list
