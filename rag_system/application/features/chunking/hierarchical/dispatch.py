from typing import Callable, Optional, Sequence

from rag_system.domain import AnyChunk, Document, SchemaNode
from rag_system.configs.chunking import ChunkingMethod, HierarchicalConfig
from .v1 import hierarchical_v1


def chunk_document_hierarchical(
    doc: Document,
    config: HierarchicalConfig,
    get_token_count: Callable[[str], int] = len
) -> tuple[Sequence[AnyChunk], Optional[SchemaNode]]:
    assert config.method == ChunkingMethod.HIERARCHICAL, f"Expected HierarchicalConfig, got method='{config.method}'"

    if config.version == '1.0':
        assert config.max_chunk_size is not None
        return hierarchical_v1(doc.text, doc.relative_path, get_token_count, config.max_chunk_size)

    raise NotImplementedError
    