from pathlib import Path

from rag_system.domain import Document
from .document_load import load_document


def reconcile(
    warehouse_hashes: dict[str, str],
    absolute_paths: list[str],
    data_dir: Path | str,
) -> tuple[list[Document], list[Document], list[Document], list[str]]:

    new_files: list[Document] = []
    changed_files: list[Document] = []
    unchanged_files: list[Document] = []
    for absolute_path in absolute_paths:
        if absolute_path not in warehouse_hashes:
            new_files.append(load_document(data_dir, absolute_path))
        else:
            warehouse_hash = warehouse_hashes[absolute_path]
            document = load_document(data_dir, absolute_path)
            if warehouse_hash == document.hash:
                unchanged_files.append(document)
            else:
                changed_files.append(document)

    to_delete = set(warehouse_hashes.keys()) - set(absolute_paths)
    return new_files, changed_files, unchanged_files, list(to_delete)
