from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

from rag_system.settings import settings


def _build_chain(
    system_template_text: str, 
    human_template_text: str, 
    model_name: str, 
    model_temperature: float
):
    llm = ChatDeepSeek(model=model_name, temperature=model_temperature, api_key=settings.DEEPSEEK_API_KEY) # type: ignore
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template_text),
        ("human", human_template_text),
    ])
    return prompt | llm


def make_general_generation_node(
    template_text: str, 
    model_name: str, 
    model_temperature: float, 
    query_key: str = "query"
):
    chain = _build_chain(template_text, "[QUERY]\n{query}", model_name, model_temperature)

    def general_generation_node(state) -> dict:
        result = chain.invoke({"query": state[query_key]})
        return {"answer": result.content}

    return general_generation_node


def make_fact_generation_node(
    template_text: str, 
    model_name: str, 
    model_temperature: float, 
    query_key: str = "query",
    chunks_key: str = "retrieved_chunks"
):
    chain = _build_chain(template_text, "[CONTEXT]\n{chunks}\n\n[QUERY]\n{query}", model_name, model_temperature)

    def fact_generation_node(state) -> dict:
        result = chain.invoke({
            "query": state[query_key],
            "chunks": "\n\n".join(state[chunks_key]),
        })
        return {"answer": result.content}

    return fact_generation_node


def make_schema_generation_node(
    template_text: str, 
    model_name: str, 
    model_temperature: float, 
    query_key: str = "query",
    schema_key: str = "schema",
    chunks_key: str = "retrieved_chunks"
):
    chain = _build_chain(template_text, "[CONTEXT]\n{chunks}\n\n[QUERY]\n{query}", model_name, model_temperature)

    def schema_generation_node(state) -> dict:
        result = chain.invoke({
            "query": state[query_key],
            'schema': state[schema_key],
            "chunks": "\n\n".join(state[chunks_key]),
        })
        return {"answer": result.content}

    return schema_generation_node
