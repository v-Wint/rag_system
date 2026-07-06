from typing import ClassVar
from sentence_transformers import CrossEncoder as HFCrossEncoder


class CrossEncoder(HFCrossEncoder):

    """Wrapper to singleton cross encoders for warmup outside of run"""
    _instances: ClassVar[dict[str, "CrossEncoder"]] = {}

    @classmethod
    def from_pretrained(cls, model_name: str) -> "CrossEncoder":
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]
