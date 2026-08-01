from typing import Any, ClassVar
from sentence_transformers import CrossEncoder as HFCrossEncoder


class CrossEncoder(HFCrossEncoder):

    """Wrapper to singleton cross encoders for warmup outside of run"""
    _instances: ClassVar[dict[tuple, "CrossEncoder"]] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str | None = None,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> "CrossEncoder":
        cache_key = (model_name, device, token, frozenset(kwargs.items()))

        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(
                model_name_or_path=model_name,
                device=device,
                token=token,
                **kwargs,
            )

        return cls._instances[cache_key]
