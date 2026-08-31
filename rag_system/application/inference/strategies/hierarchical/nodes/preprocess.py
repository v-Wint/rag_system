from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

from rag_system.settings import settings

from rag_system.infrastructure.schema_retriver import SchemaRetriever
from rag_system.configs.chunking import HierarchicalConfig
from ..enums import QuestionType


class QueryAnalysis(BaseModel):
    reasoning: str = Field(
        description="Rationale for the chosen classification."
    )
    question_type: QuestionType = Field(
        description="Classified query category based on schema vs record dependency."
    )
    improved_query: str = Field(
        description="Clean, well-phrased natural language question for generation."
    )
    retrieval_query: str = Field(
        description="Keyword-dense search query for vector retrieval (empty if general-question)."
    )


def make_preprocess_node(
    chunking_config: HierarchicalConfig, 
    template_text: str, 
    model_name: str, 
    model_temperature: float,
    query_key='query'
):
    schema_retriever = SchemaRetriever(
        chunking_config.size_metric,
        chunking_config.max_chunk_size,
        chunking_config.max_schema_size,
    )
    llm = ChatDeepSeek(
        model=model_name, 
        temperature=model_temperature, 
        api_key=settings.DEEPSEEK_API_KEY # type: ignore
    )

    llm = llm.with_structured_output(QueryAnalysis)
    prompt = ChatPromptTemplate.from_messages([
        ("system", template_text),
        ("human", "[USER QUERY]\n{query}")
    ])

    chain = prompt | llm

    def preprocess_node(state) -> dict:
        query = state[query_key]
        schema = schema_retriever.invoke()

        analysis = chain.invoke({
            "query": query,
            "schema": schema,
        })

        assert isinstance(analysis, QueryAnalysis)

        return {
            "improved_query": analysis.improved_query,
            "retrieval_query": analysis.retrieval_query,
            "question_type": analysis.question_type.value,
            "schema": schema,
            "debug": {"classification_reasoning": analysis.reasoning}
        }

    return preprocess_node
