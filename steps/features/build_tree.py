from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import step, log_metadata

from rag_system.domain import Document, DocNode
from rag_system.application.features.tree import build_doc_subtree
from rag_system.configs.chunking import ChunkingConfig


@step
def build_tree_step(
    documents: list[Document],
    config: ChunkingConfig
) -> Annotated[list[DocNode], "doc_subtrees"]:
    """Build rich DocNode subtrees for the changed documents.

    Uses the config's size metric and max_chunk_size. Shared with the future
    agentic pipeline (it needs the same tree, minus embedding).
    """
    if not documents:
        return []

    get_size = config.make_get_size()

    logger.info(
        f"Building doc subtrees for {len(documents)} documents with {config.model_dump()}"
    )

    subtrees: list[DocNode] = []
    for document in tqdm(documents, desc="Building doc trees"):
        subtrees.append(build_doc_subtree(
            document.text,
            document.relative_path.split('/'),
            document.hash,
            get_size,
            config.max_chunk_size,
        ))

    logger.info(f"Built {len(subtrees)} doc subtrees")

    log_metadata(metadata={
        "num_documents": len(documents),
        "num_subtrees": len(subtrees),
    }, infer_artifact=True)
    return subtrees
