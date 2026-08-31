from typing_extensions import Annotated

from zenml import step, log_metadata
from loguru import logger

from rag_system.domain import DocNode, KBTree
from rag_system.infrastructure import mongo_init
from rag_system.configs.enums import SizeMetric


@step(enable_cache=False)
def upsert_kb_tree_step(
    subtrees: list[DocNode],
    deleted_paths: list[str],
    size_metric: SizeMetric,
    max_size: int
) -> Annotated[bool, "structure_changed"]:
    """Merge built subtrees into the whole-KB KBTree, drop deleted docs, persist.

    Reindexes ids/depths and upserts the single KBTree doc keyed on
    (size_metric, max_size). Returns whether the structure changed, which drives
    the schema rebuild and (via DAG edge) the re-embedding.
    """
    if not subtrees and not deleted_paths:
        logger.info(
            f"No tree changes for size_metric={size_metric}, max_size={max_size}; skipping."
        )
        return False

    mongo_init()

    tree = KBTree.load(size_metric, max_size)
    if tree is None:
        root = DocNode(title='root', text='root')
        tree = KBTree(root=root, size_metric=size_metric, max_size=max_size)

    root = tree.root

    if deleted_paths:
        deleted = set(deleted_paths)
        root.children = [
            child for child in root.children
            if "/".join(child.doc_path) not in deleted
        ]

    for subtree in subtrees:
        path = "/".join(subtree.doc_path)
        replaced = False
        for i, child in enumerate(root.children):
            if "/".join(child.doc_path) == path:
                root.children[i] = subtree
                replaced = True
                break
        if not replaced:
            root.children.append(subtree)

    root.reindex()
    tree.upsert()

    logger.info(
        f"Synced {len(subtrees)} built + {len(deleted_paths)} deleted subtrees "
        f"into KB tree (total docs: {len(root.children)})"
    )

    log_metadata({
        "built": len(subtrees),
        "deleted": len(deleted_paths),
        "total_docs": len(root.children),
    })
    return True
