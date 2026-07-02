from pydantic import BaseModel

class Chunk(BaseModel):
    title: str
    # relative path of document inside of directory
    doc_path: list[str]
    abs_path: list[str]
    rel_path: list[str]
    embedding_text: str
    content: str

    @classmethod
    def from_params(cls, title: str, doc_path: list[str], parent_path: list[str], text: str) -> 'Chunk':
        rel_path = parent_path + [title]
        abs_path = doc_path + rel_path

        embedding_text = "Document Location: " + " > ".join(abs_path) + "\n\n" + text

        return cls(
            title=title,
            doc_path=doc_path,
            abs_path=abs_path,
            rel_path=rel_path,
            embedding_text=embedding_text,
            content=text
        )
