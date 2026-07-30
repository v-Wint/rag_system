from typing import ClassVar
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import scan_cache_dir

from rag_system import settings

_PREFIXED_MODELS = {
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
}

def _is_cached(model_name: str) -> bool:
    try:
        cache_info = scan_cache_dir()
        return any(repo.repo_id == model_name for repo in cache_info.repos)
    except Exception:
        return False


class Embedder(HuggingFaceEmbeddings):
    _instances: ClassVar[dict[str, "Embedder"]] = {}
    prefixed: bool = False

    def __init__(self, model_name):
        super().__init__(
            model_name=model_name,
            model_kwargs={
                "token": settings.HUGGINGFACE_ACCESS_TOKEN,
                "device": settings.TEXT_EMBEDDING_DEVICE,
                "local_files_only": _is_cached(model_name)
            },
            encode_kwargs={"normalize_embeddings": True}

        )
        if self.model_name in _PREFIXED_MODELS:
            self.prefixed = True

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Embedder":
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.prefixed:
            texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.prefixed:
            text = f"query: {text}"
        return super().embed_query(text)

    def get_token_count(self, text: str) -> int:
        return len(self._client.tokenizer(text)["input_ids"])

    @property
    def vector_size(self) -> int | None:
        return self._client.get_embedding_dimension()

    @property
    def max_tokens(self) -> int | None:
        max_tokens = self._client.max_seq_length 
        if max_tokens and self.prefixed:
            max_tokens -= 2
        return max_tokens
