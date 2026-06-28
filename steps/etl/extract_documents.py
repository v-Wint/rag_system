from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import step, log_metadata
from datetime import datetime, timezone

from rag_system.infrastructure.db import mongo_init
from rag_system.application.etl import extract_text, compute_file_hash
from rag_system.domain import Document


@step
def extract_documents_step(data_dir: str, document_paths: list[str]) -> Annotated[list[Path], "extracted_paths"]:
    mongo_init()
    logger.info(f"Starting to extract {len(document_paths)} document(s).")
    root = Path(data_dir).resolve()
    metadata = {}
    successful_paths = []

    for path_str in tqdm(document_paths):
        path = Path(path_str)
        successful, extension = _extract_document(path, root)
        if successful:
            successful_paths.append(path)
        metadata = _add_to_metadata(metadata, extension, bool(successful))

    log_metadata(metadata=metadata, infer_artifact=True)
    logger.info(f"Successfully extracted {len(successful_paths)} / {len(document_paths)} documents.")
    return [str(path.relative_to(root)) for path in successful_paths]


def _extract_document(path: Path, root: Path) -> tuple[bool, str]:
    try:
        text = extract_text(path)
        hash_ = compute_file_hash(str(path))
        now = datetime.now(timezone.utc)
        result = Document.find_one({"absolute_path": str(path)}).upsert(
            {"$set": {
                "root_dir": str(root),
                "absolute_path": str(path),
                "relative_path": str(path.relative_to(root)),
                "text": text,
                "hash": hash_,
                "updated_at": now,
            }},
            on_insert=Document(
                root_dir=str(root),
                absolute_path=str(path),
                relative_path=str(path.relative_to(root)),
                text=text,
                hash=hash_,
                created_at=now,
                updated_at=now,
            ),
        ).run()
        return True, path.suffix
    except Exception as e:
        logger.error(f"Failed to extract {path}: {e!s}")
        return False, path.suffix


def _add_to_metadata(metadata: dict, extension: str, successful: bool) -> dict:
    if extension not in metadata:
        metadata[extension] = {}
    metadata[extension]["successful"] = metadata[extension].get("successful", 0) + successful
    metadata[extension]["total"] = metadata[extension].get("total", 0) + 1
    return metadata