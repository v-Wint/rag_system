from pathlib import Path
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from rag_system.domain import RAGState
from rag_system.settings import settings


def _make_answer_node(
    model_name: str,
    temperature: float,
    template_file: Path | str,
    input_mapper: Callable[[RAGState], dict],
):
    llm = ChatGroq(model=model_name, temperature=temperature, api_key=settings.GROQ_API_KEY) # type: ignore
    with open(template_file, 'r') as f:
        template = ChatPromptTemplate.from_template(f.read())
    chain = template | llm | StrOutputParser()

    def answer_node(state: RAGState) -> dict:
        node_input = input_mapper(state)
        answer = chain.invoke(node_input)
        return {"answer": answer, "schema_used_in_prompt": bool(node_input.get('schema'))}

    return answer_node


def make_fact_answer(model_name: str, temperature: float, template_file: Path | str):
    return _make_answer_node(
        model_name, temperature, template_file,
        input_mapper= lambda state: {
            "query": state.improved_query,
            "context": "\n\n".join(state.retrieved_chunks), # type: ignore
        },
    )


def make_schema_answer(model_name: str, temperature: float, template_file: Path | str):
    return _make_answer_node(
        model_name, temperature, template_file,
        input_mapper=lambda state: {
            "query": state.improved_query,
            "schema": state.retrieved_schema,
            "context": "\n\n".join(state.retrieved_chunks), # type: ignore
        },
    )


def make_general_answer(model_name: str, temperature: float, template_file: Path | str):
    return _make_answer_node(
        model_name, temperature, template_file,
        input_mapper=lambda state: {"query": state.improved_query},
    )
