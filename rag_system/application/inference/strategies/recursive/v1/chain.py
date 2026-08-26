from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

from rag_system.infrastructure import Embedder, VectorStore
from rag_system.configs.inference import RecursiveV1InferenceConfig
from rag_system.settings import settings


def build_chain(
    config: RecursiveV1InferenceConfig
):

    embedder = Embedder.from_pretrained(config.chunking.embedding_model)
    store = VectorStore.for_retrieval(config.chunking.slug, embedder)
    retriever = store.as_retriever(search_kwargs={"k": config.retrieval.k})

    llm = ChatDeepSeek(
        model=config.generation.model_name, 
        temperature=config.generation.model_temperature, 
        api_key=settings.DEEPSEEK_API_KEY # type: ignore
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.generation.template_text),
            ("human", "[CONTEXT]\n{chunks}\n\n[QUERY]\n{query}"),
        ]
    )

    generate = prompt | llm | StrOutputParser()

    retrieve_chunks = RunnableLambda(
        lambda x: [d.page_content for d in retriever.invoke(x["query"])] # type: ignore
    )
    build_generate_input = RunnableLambda(
        lambda x: {"chunks": "\n\n".join(x["retrieved_chunks"]), "query": x["query"]} # type: ignore
    )

    chain = (
        {"query": RunnablePassthrough()}
        | RunnablePassthrough.assign(retrieved_chunks=retrieve_chunks)
        | RunnablePassthrough.assign(answer=build_generate_input | generate)
    )

    return chain
