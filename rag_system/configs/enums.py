from enum import Enum

class ChunkingMethod(str, Enum):
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"


class InferenceStrategy(str, Enum):
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"
