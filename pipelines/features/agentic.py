from zenml import pipeline

from rag_system.configs.chunking import AgenticV1Config

from steps.features import (
    reconcile_against_tree_step, build_tree_step, upsert_kb_tree_step)


@pipeline
def agentic_feature_pipeline(config):
    documents, deleted_paths = reconcile_against_tree_step(
        config.size_metric, config.max_chunk_size
    )

    subtrees = build_tree_step(documents, config)

    upsert_kb_tree_step(
        subtrees, deleted_paths, config.size_metric, config.max_chunk_size
    )


if __name__ == '__main__':
    agentic_feature_pipeline(AgenticV1Config().resolve())
