from typing import Callable

from rag_system.domain import DocumentNode, HierarchicalChunk

from .base import _split_chunks, add_doc_path_to_schema

def hierarchical_v1(
    text: str, 
    doc_path: str | list, 
    get_size: Callable[[str], int] = len, 
    max_size=1_00_000
) -> tuple[list[HierarchicalChunk], DocumentNode]:

    if isinstance(doc_path, str):
        doc_path = doc_path.split('/')

    leaf_node = DocumentNode()
    data_chunks = _split_chunks(text, doc_path, [], get_size, max_size, leaf_node)

    root = add_doc_path_to_schema(doc_path, leaf_node)
    
    return data_chunks, root
