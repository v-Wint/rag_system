from enum import Enum
from typing import Callable

from .remnote_v1 import _clean_remnote_v1

class CleaningMethod(str, Enum):
    REMNOTE_V1 = "remnote_v1"


CLEANING_REGISTRY: dict[CleaningMethod, Callable[[str], str]] = {
    CleaningMethod.REMNOTE_V1: _clean_remnote_v1,
}

def clean_text(text: str, method: CleaningMethod = CleaningMethod.REMNOTE_V1) -> str:
    return CLEANING_REGISTRY[method](text)
