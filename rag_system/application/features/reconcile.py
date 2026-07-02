from rag_system.domain import Document

def reconcile(
    store_hashes: dict[str, str],
    documents: list[Document]
) -> tuple[list[Document], list[Document], list[Document], list[str]]:

    new_docs: list[Document] = []
    changed_docs: list[Document] = []
    unchanged_docs: list[Document] = []

    for document in documents:
        if document.relative_path not in store_hashes:
            new_docs.append(document)
        else:
            store_hash = store_hashes[document.relative_path]
            if store_hash == document.hash:
                unchanged_docs.append(document)
            else:
                changed_docs.append(document)

    to_delete = set(store_hashes.keys()) - set([d.relative_path for d in documents])
    return new_docs, changed_docs, unchanged_docs, list(to_delete)
