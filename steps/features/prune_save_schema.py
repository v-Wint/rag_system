from zenml import step, log_metadata
from typing import Annotated


from rag_system.domain import SchemaNode, SchemaText
from rag_system.infrastructure import Embedder, mongo_init

@step(enable_cache=False)
def prune_save_schema_step(
    schema: SchemaNode | None,
    embedding_model: str,
    max_schema_tokens: int | None,
    collection_name: str
) -> Annotated[str, "truncated_schema"]:
    mongo_init()

    if not schema:
        return ''
    if not max_schema_tokens:
        raise ValueError("max_schema_tokens is required to save schema")

    embedder = Embedder.from_pretrained(embedding_model)
    depth = 0
    while embedder.get_token_count(str(schema.truncated(depth))) < max_schema_tokens:
        depth += 1
    

    schema_truncated = schema.truncated(depth-1)

    SchemaText.from_schema_node(
        schema_truncated,
        collection_name
    ).upsert()


    log_metadata(metadata={
        "max_depth": depth-1
        },
        infer_artifact=True
    )

    return str(schema_truncated)
