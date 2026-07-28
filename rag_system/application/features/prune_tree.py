from typing import Callable
from rag_system.domain import DocumentNode

def prune_tree(
    tree: DocumentNode,
    get_size: Callable[[str], int],
    max_size: int,
):
    if get_size(str(tree.truncated(0))) > max_size:
        raise ValueError(
            "Schema exceeds max_schema_tokens even at depth 0; truncating to root only"
        )

    max_depth = tree.height()
    depth = 0
    while depth < max_depth and get_size(str(tree.truncated(depth + 1))) <= max_size:
        depth += 1

    return tree.truncated(depth)
