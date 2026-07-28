from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_system.domain import RecursiveChunk

def recursive_v1(
    text: str, 
    doc_path: str | list, 
    get_size: Callable[[str], int] = len, 
    max_size=1_00_000,
    overlap_size=100
) -> list[RecursiveChunk]:
    if isinstance(doc_path, str):
        doc_path = doc_path.split('/')

    splitter = RecursiveCharacterTextSplitter(
        separators=['\n-', '\n\n\n', '\n\n', '\n', ' ', ''],
        length_function=get_size,
        chunk_size=max_size,
        chunk_overlap=overlap_size,
        is_separator_regex=True
    )

    chunk_texts = splitter.split_text(text)

    return [
        RecursiveChunk(text=chunk_text, embedding_text=chunk_text, doc_path=doc_path) for chunk_text in chunk_texts
    ]
