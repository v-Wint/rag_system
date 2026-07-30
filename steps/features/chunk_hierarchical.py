from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import step, log_metadata
from rag_system.domain import Document, DocumentTree, ChunkDocument
from rag_system.application.features.chunking.hierarchical import chunk_document_hierarchical
from rag_system.infrastructure import Embedder
from rag_system.configs.chunking import HierarchicalConfig


@step
def chunk_hierarchical_step(
    documents: list[Document],
    config: HierarchicalConfig
) -> tuple[
    Annotated[list[ChunkDocument], "chunks"],
    Annotated[list[DocumentTree], "document_trees"]
]:
    if not documents:
        return [], []

    embedder = Embedder.from_pretrained(config.embedding_model)

    logger.info(
        f"Chunking {len(documents)} documents with {config.model_dump()}"
    )

    chunk_list: list[ChunkDocument] = []
    tree_list: list[DocumentTree] = []

    for document in tqdm(documents, desc="Chunking documents"):
        chunks, tree = chunk_document_hierarchical(
            document, config, lambda s: embedder.get_token_count(s),
        )
        cds = [ChunkDocument(chunk=chunk, doc_hash=document.hash) for chunk in chunks]
        chunk_list += cds
        tree_list.append(DocumentTree(root=tree, doc_path=document.relative_path, config_slug=config.slug))

    logger.info(f"Produced {len(chunk_list)} chunks from {len(documents)} documents")

    metadata = config.model_dump(exclude_none=True)
    metadata.update({
        "num_documents": len(documents),
        "num_chunks": len(chunk_list),
    })

    log_metadata(
        metadata=metadata,
        artifact_name="chunks",
        infer_artifact=True
    )
    return chunk_list, tree_list
