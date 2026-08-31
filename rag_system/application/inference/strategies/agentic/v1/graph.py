from operator import add
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END

from rag_system.domain import KBTree
from rag_system.infrastructure import mongo_init
from rag_system.configs.inference import AgenticV1InferenceConfig

from ..nodes import make_agent_node, make_finalize_node, make_tools_node
from ..tools import make_expand_tool
from ..tree_index import TreeIndex


class AgenticV1State(TypedDict):
    query: str
    messages: Annotated[list, add]
    iteration: int
    answer: str
    debug: Annotated[list[dict[str, Any]], add]


def load_index(config: AgenticV1InferenceConfig) -> TreeIndex:
    mongo_init()
    tree = KBTree.load(config.chunking.size_metric, config.chunking.max_chunk_size)
    if tree is None:
        raise RuntimeError(
            f"No KBTree for size_metric={config.chunking.size_metric}, "
            f"max_size={config.chunking.max_chunk_size}. "
            "Run the agentic feature pipeline first."
        )
    return TreeIndex(tree.root)


def build_graph(config: AgenticV1InferenceConfig, index: TreeIndex | None = None):
    if index is None:
        index = load_index(config)
    tool = make_expand_tool(
        index,
        config.tooling.max_expand_size,
        config.tooling.preview_length,
    )

    graph = StateGraph(AgenticV1State)

    graph.add_node(
        "agent",
        make_agent_node(
            config.generation.model_name,
            config.generation.model_temperature,
            config.generation.template_text,
            tool,
        ),
    )

    graph.add_node("tools", make_tools_node(tool))

    graph.add_node(
        "finalize",
        make_finalize_node(
            config.generation.model_name,
            config.generation.model_temperature,
            config.generation.template_text,
        ),
    )

    graph.add_edge(START, "agent")

    max_iterations = config.tooling.max_iterations

    def route_after_agent(state) -> Literal["tools", "finalize"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            if state["iteration"] >= max_iterations:
                return "finalize"
            return "tools"
        return "finalize"

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()
