from rag_system.domain.schema import SchemaNode, unite_schemas


def reconcile_schema(
    old_root: SchemaNode | None,
    schemas: list[SchemaNode],
    deleted_rel_paths: list[str],
) -> tuple[SchemaNode, int, int, int]:
    if old_root is None:
        return unite_schemas(schemas), len(schemas), 0, 0

    schema_node = old_root

    before_count = len(schema_node.children)
    schema_node.children = [
        n for n in schema_node.children if n.title not in deleted_rel_paths
    ]
    n_deleted = before_count - len(schema_node.children)

    existing_by_title = {n.title: i for i, n in enumerate(schema_node.children)}
    n_added = 0
    n_replaced = 0

    for node in schemas:
        if node.title in existing_by_title:
            idx = existing_by_title[node.title]
            schema_node.children[idx] = node
            n_replaced += 1
        else:
            schema_node.children.append(node)
            existing_by_title[node.title] = len(schema_node.children) - 1
            n_added += 1

    return schema_node, n_added, n_replaced, n_deleted
