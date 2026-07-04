from pathlib import Path
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

from rag_system.settings import settings

from ..schema_retriver import SchemaRetriever
from ..state import RAGState, QuestionType


class QueryAnalysis(BaseModel):
    reasoning: str = Field(
        description=(
            "First, assess how much this query depends on the schema's structural information "
            "(field definitions, categories, what kinds of data exist) versus actual record/document content. "
            "Then give a 1-sentence explanation of the resulting classification."
        )
    )
    question_type: QuestionType = Field(
        description=(
            "'fact-question': the user wants specific data/values from records. Retrieval should pull "
            "document chunks only — the schema itself adds no value here. "
            "'schema-question': the user is asking about structure, what kinds of data are tracked, "
            "overarching capabilities, or a question best contextualized by the schema layout — "
            "but may still benefit from a small number of supporting document chunks alongside heavier "
            "schema context. "
            "'general-question': greetings, casual talk, or irrelevant topics needing zero KB data."
        )
    )
    improved_query: str = Field(
        description=(
            "A clear, well-formed restatement of the user's question, written for a downstream LLM that will "
            "answer using retrieved context. Fix typos and awkward phrasing, spell out vague or ambiguous "
            "wording, and make the question read naturally and unambiguously. Do NOT optimize for keyword "
            "density — this should read as a normal, well-phrased question, not a search query. "
            "If question_type is 'general-question', set this to a cleaned-up version of the original query."
        )
    )
    retrieval_query: str = Field(
        description=(
            "A stripped-down, optimized, keyword-rich search query designed for vector search, based on "
            "contextualized_query. Represent the user's information need as precisely and completely as "
            "possible. Remove conversational fluff and fix typos, but do not omit meaningful content. "
            "If question_type is 'general-question', set this to an empty string."
        )
    )

def make_preprocess_node(collection_name: str, template_file: Path | str, model: str, temperature: float):
    schema_retriever = SchemaRetriever(collection_name)
    llm = ChatGroq(model=model, temperature=temperature, api_key=settings.GROQ_API_KEY) # type: ignore
    parser = JsonOutputParser(pydantic_object=QueryAnalysis)
    with open(template_file, 'r') as f:
        template = ChatPromptTemplate.from_template(
            f.read(), 
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

    chain = template | llm | parser
    
    def preprocess_node(state: RAGState) -> dict:
        schema = schema_retriever.invoke()
        result = chain.invoke({"schema": schema, "query": state.query})
        return {
            "improved_query": result["improved_query"],
            "retrieval_query": result["retrieval_query"],
            "question_type": result["question_type"],
            "classification_reasoning": result["reasoning"],
            "retrieved_schema": schema
        }

    return preprocess_node

if __name__ == '__main__':
    node = make_preprocess_node(
        "chunks__remnote_v1__hierarchical_v1__intfloat_multilingual_e5_base__510__2048",
        'templates/preprocess.txt',
        'llama-3.3-70b-versatile',
        0.05
    )

    print(node(RAGState(query="what did i learn in semester 2.2")))
    print(node(RAGState(query="Hello!!!!!! How are you ?????")))
    print(node(RAGState(query="?")))
    print(node(RAGState(query="What is docker")))
    print(node(RAGState(query="Hello, computer! What is docker")))
