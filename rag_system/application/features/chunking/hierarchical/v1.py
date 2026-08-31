from langchain_core.documents import Document

from rag_system.domain import DocNode
from rag_system.domain.documents import Document as WarehouseDocument
from rag_system.application.features.tree import build_doc_subtree
from rag_system.configs.chunking import HierarchicalConfig


def flatten_leaves(doc_subtrees: list[DocNode]) -> list[Document]:
    """Translate leaf DocNodes of each doc subtree into LangChain Documents."""
    documents: list[Document] = []
    for subtree in doc_subtrees:
        doc_hash = subtree.doc_hash
        if doc_hash is None:
            continue
        for leaf in subtree.walk_leaves():
            if not leaf.rel_path:
                continue
            documents.append(leaf.to_langchain_document(doc_hash))
    return documents


def chunk_document_hierarchical(
    doc: WarehouseDocument,
    config: HierarchicalConfig,
) -> list[Document]:
    """Build a doc subtree and translate its leaves into LangChain chunks."""
    subtree = build_doc_subtree(
        doc.text,
        doc.relative_path.split('/'),
        doc.hash,
        config.make_get_size(),
        config.max_chunk_size,
    )
    return flatten_leaves([subtree])
