from zenml import pipeline

from rag_system.configs.chunking import HierarchicalConfig, HierarchicalV1Config

from steps.features import (
    get_changed_step, chunk_hierarchical_step, embed_load_chunks_step, 
    sync_document_trees_step, prune_save_schema_step)


@pipeline
def hierarchical_feature_pipeline(
    config: HierarchicalConfig
):
    documents, to_delete_rel = get_changed_step(config.slug)

    chunks, trees = chunk_hierarchical_step(documents, config)

    embed_load_chunks_step(
        chunks, to_delete_rel, config.embedding_model, config.slug
    )

    structure_changed = sync_document_trees_step(
        trees, to_delete_rel, config.slug
    )

    prune_save_schema_step(
        config.slug, config.embedding_model, config.max_schema_size, structure_changed
    )


if __name__ == '__main__':
    hierarchical_feature_pipeline(HierarchicalV1Config(max_schema_size=2048))
