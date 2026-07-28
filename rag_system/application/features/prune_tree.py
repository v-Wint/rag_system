from typing import Callable
from rag_system.domain import DocumentNode

def prune_tree(
    tree: DocumentNode,
    get_size: Callable[[str], int], 
    max_size: int
):
    depth = 0
    while get_size(str(tree.truncated(depth))) < max_size:
        depth += 1
    if depth == 0:
        raise ValueError("Schema exceeds max_schema_tokens even at depth 0; truncating to root only")

    return tree.truncated(depth-1)
