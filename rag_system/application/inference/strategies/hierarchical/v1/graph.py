from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from ...hierarchical import nodes
from rag_system.configs.inference import HierarchicalV1InferenceConfig

from ..enums import QuestionType


class HierarchicalV1State(TypedDict):
    query: str
    answer: str
    generate: str
    schema: str
    question_type: QuestionType
    improved_query: str
    retrieval_query: str
    retrieved_chunks: list[str]

    debug: Annotated[dict[str, Any], lambda left, right: (left or {}) | (right or {})]


def build_graph(config: HierarchicalV1InferenceConfig):
    graph = StateGraph(HierarchicalV1State)

    graph.add_node(
        "preprocess", 
        nodes.make_preprocess_node(
            config.chunking, 
            config.preprocess.template_text,
            config.preprocess.model_name,
            config.preprocess.model_temperature))

    graph.add_node(
        "retrieve_fact",
        nodes.make_retrieve_node(
            config.chunking,
            config.retrieval.vector.fact_k,
            config.retrieval.reranker.model_name,
            config.retrieval.reranker.fact_k,
            query_key="retrieval_query"
        )
    )

    graph.add_node(
        "retrieve_schema",
        nodes.make_retrieve_node(
            config.chunking,
            config.retrieval.vector.schema_k,
            config.retrieval.reranker.model_name,
            config.retrieval.reranker.schema_k,
            query_key="retrieval_query"
        )
    )

    graph.add_node(
        "generate_fact",
        nodes.make_fact_generation_node(
            config.generation.fact_template_text,
            config.generation.model_name,
            config.generation.model_temperature,
            query_key="improved_query"
        )
    )

    graph.add_node(
        "generate_schema",
        nodes.make_schema_generation_node(
            config.generation.schema_template_text,
            config.generation.model_name,
            config.generation.model_temperature,
            query_key="improved_query"
        )
    )

    graph.add_node(
        "generate_general",
        nodes.make_general_generation_node(
            config.generation.general_template_text,
            config.generation.model_name,
            config.generation.model_temperature,
            query_key="improved_query"
        )
    )

    graph.set_entry_point("preprocess")
    graph.add_conditional_edges(
        "preprocess",
        lambda state: state['question_type'],
        {
            "fact-question": "retrieve_fact",
            "schema-question": "retrieve_schema",
            "general-question": "generate_general",
        }
    )
    graph.add_edge("retrieve_fact", "generate_fact")
    graph.add_edge("retrieve_schema", "generate_schema")

    graph.add_edge("generate_fact", END)
    graph.add_edge("generate_schema", END)
    graph.add_edge("generate_general", END)

    return graph.compile()
