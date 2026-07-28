from typing import Callable, Sequence

from rag_system.domain import AnyChunk, Document, DocumentNode
from rag_system.configs.chunking import ChunkingMethod, RecursiveConfig
from .v1 import recursive_v1


def chunk_document_recursive(
    doc: Document,
    config: RecursiveConfig,
    get_size: Callable[[str], int] = len
) -> Sequence[AnyChunk]:
    assert config.method == ChunkingMethod.RECURSIVE, f"Expected RecursiveConfig, got method='{config.method}'"

    if config.version == '1.0':
        assert config.max_chunk_size is not None
        assert config.overlap_size is not None
        return recursive_v1(doc.text, doc.relative_path, get_size, config.max_chunk_size, config.overlap_size)

    raise NotImplementedError
    