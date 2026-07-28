from zenml import pipeline

from rag_system.configs.chunking import RecursiveConfig, RecursiveV1Config

from steps.features import (
    get_changed_step, chunk_recursive_step, embed_load_chunks_step)

@pipeline
def recursive_feature_pipeline(
    config: RecursiveConfig
):
    documents, to_delete_rel = get_changed_step(config.slug)

    chunks = chunk_recursive_step(documents, config)

    embed_load_chunks_step(
        chunks, to_delete_rel, config.embedding_model, config.slug
    )


if __name__ == '__main__':
    recursive_feature_pipeline(RecursiveV1Config())
