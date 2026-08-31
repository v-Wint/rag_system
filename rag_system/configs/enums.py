from enum import Enum

class ChunkingMethod(str, Enum):
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"
    AGENTIC = "agentic"


class InferenceStrategy(str, Enum):
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"
    AGENTIC = "agentic"


class SizeMetric(str, Enum):
    """Unit of the max_size threshold for tree/chunk granularity."""
    TOKENS = "tokens"
    CHARS = "chars"
