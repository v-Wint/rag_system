from zenml import pipeline

from rag_system.configs.chunking import HierarchicalConfig, HierarchicalV1Config

from steps.features import (
    get_changed_step, chunk_hierarchical_step, 
    embed_load_chunks_step, prune_save_schema_step)


@pipeline
def hierarchical_feature_pipeline(
    config: HierarchicalConfig
):
    documents, to_delete_rel = get_changed_step(config.get_collection_name())

    chunks, schemas = chunk_hierarchical_step(documents, config)

    prune_save_schema_step(
        schemas, to_delete_rel, config.embedding_model, config.max_schema_size, config.get_collection_name()
    )

    embed_load_chunks_step(
        chunks, to_delete_rel, config.embedding_model, config.get_collection_name(), 
        after="prune_save_schema_step"
    )

if __name__ == '__main__':
    hierarchical_feature_pipeline(HierarchicalV1Config(max_schema_size=1024))
