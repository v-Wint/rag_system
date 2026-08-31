from zenml import pipeline

from rag_system.configs.chunking import HierarchicalConfig, HierarchicalV1Config

from steps.features import (
    reconcile_against_tree_step, build_tree_step, upsert_kb_tree_step,
    embed_hierarchical_chunks_step, prune_save_schema_step)


@pipeline
def hierarchical_feature_pipeline(
    config: HierarchicalConfig
):
    documents, deleted_paths = reconcile_against_tree_step(
        config.size_metric, config.max_chunk_size
    )

    subtrees = build_tree_step(documents, config)

    structure_changed = upsert_kb_tree_step(
        subtrees, deleted_paths, config.size_metric, config.max_chunk_size
    )

    embed_hierarchical_chunks_step(
        config.size_metric, config.max_chunk_size, config.embedding_model, config.slug,
        structure_changed
    )

    prune_save_schema_step(
        config.size_metric, config.max_chunk_size, config.max_schema_size,
        config.embedding_model, structure_changed
    )


if __name__ == '__main__':
    hierarchical_feature_pipeline(HierarchicalV1Config(raw_max_chunk_size=500, raw_max_schema_size=1000).resolve())
