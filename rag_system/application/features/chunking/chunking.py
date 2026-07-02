from enum import Enum
from typing import Callable, Optional

from rag_system.domain import Chunk, SchemaNode
from .hierarchical_v1 import _hierarchical_v1

class ChunkingMethod(str, Enum):
    HIERARCHICAL_V1 = "hierarchical_v1"

ChunkingFunction = Callable[[str, str | list, Callable[[str], int], int], tuple[list[Chunk], SchemaNode]]
CHUNKING_REGISTRY: dict[ChunkingMethod, ChunkingFunction] = {
    ChunkingMethod.HIERARCHICAL_V1: _hierarchical_v1,
}

def chunk_document(
        text: str,
        doc_path: str | list,
        method: ChunkingMethod = ChunkingMethod.HIERARCHICAL_V1,
        *,
        get_token_count: Callable[[str], int] = len,
        max_tokens: int = 1_000_000,
) -> tuple[list[Chunk], SchemaNode]:
    return CHUNKING_REGISTRY[method](text, doc_path, get_token_count, max_tokens)
