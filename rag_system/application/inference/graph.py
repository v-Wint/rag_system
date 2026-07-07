from langgraph.graph import StateGraph, END

from . import nodes
from rag_system.domain import RAGState, RAGConfig


def build_graph(config: RAGConfig):
    graph = StateGraph(RAGState)

    graph.add_node(
        "preprocess", 
        nodes.make_preprocess_node(
            config.collection_name, 
            config.preprocess_template_file, 
            config.preprocess_model, 
            config.preprocess_model_temperature
            )
        )

    graph.add_node(
        "retrieve_fact",
        nodes.make_retrieve_node(
            config.embedding_model,
            config.collection_name,
            config.vector_retrieval_size_fact,
            config.cross_encoder_model,
            config.reranking_size_fact
        )
    )

    graph.add_node(
        "retrieve_schema",
        nodes.make_retrieve_node(
            config.embedding_model,
            config.collection_name,
            config.vector_retrieval_size_schema,
            config.cross_encoder_model,
            config.reranking_size_schema
        )
    )

    graph.add_node(
        "answer_fact",
        nodes.make_fact_answer(
            config.llm_model,
            config.llm_model_temperature,
            config.llm_fact_template
        )
    )

    graph.add_node(
        "answer_schema",
        nodes.make_schema_answer(
            config.llm_model,
            config.llm_model_temperature,
            config.llm_schema_template
        )
    )

    graph.add_node(
        "answer_general",
        nodes.make_general_answer(
            config.llm_model,
            config.llm_model_temperature,
            config.llm_general_template
        )
    )

    graph.set_entry_point("preprocess")
    graph.add_conditional_edges(
        "preprocess",
        lambda state: state.question_type.value,
        {
            "fact-question": "retrieve_fact",
            "schema-question": "retrieve_schema",
            "general-question": "answer_general",
        }
    )
    graph.add_edge("retrieve_fact", "answer_fact")
    graph.add_edge("retrieve_schema", "answer_schema")

    graph.add_edge("answer_fact", END)
    graph.add_edge("answer_schema", END)
    graph.add_edge("answer_general", END)

    return graph.compile()


instances = {}

def get_graph(config: RAGConfig):
    hash_value = hash(config)
    instance = instances.get(hash_value)

    if not instance:
        instances[hash_value] = build_graph(config)
        return instances[hash_value]

    return instance



if __name__ == '__main__':
    graph = build_graph(RAGConfig(collection_name="chunks__remnote_v1__hierarchical_v1__intfloat_multilingual_e5_base__510__2048"))

    final_state = graph.invoke(RAGState(query="Help me understand bayes formula"))
    print(final_state['question_type'])
    print(final_state['answer'])
    for chunk in final_state['retrieved_chunks']:
        print(chunk)
    