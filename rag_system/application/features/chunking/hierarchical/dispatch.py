from typing import Callable, Optional, Sequence

from rag_system.domain import AnyChunk, Document, DocumentNode
from rag_system.configs.chunking import ChunkingMethod, HierarchicalConfig
from .v1 import hierarchical_v1


def chunk_document_hierarchical(
    doc: Document,
    config: HierarchicalConfig,
    get_size: Callable[[str], int] = len
) -> tuple[Sequence[AnyChunk], DocumentNode]:
    assert config.method == ChunkingMethod.HIERARCHICAL, f"Expected HierarchicalConfig, got method='{config.method}'"

    if config.version == '1.0':
        assert config.max_chunk_size is not None
        return hierarchical_v1(doc.text, doc.relative_path, get_size, config.max_chunk_size)

    raise NotImplementedError
    