from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

from rag_system.settings import settings


def make_agent_node(model_name: str, model_temperature: float, system_prompt: str, tool):
    llm = ChatDeepSeek(
        model=model_name,
        temperature=model_temperature,
        api_key=settings.DEEPSEEK_API_KEY,  # type: ignore
    ).bind_tools([tool])

    def agent_node(state) -> dict:
        response = llm.invoke(state["messages"])
        return {
            "messages": [response],
            "iteration": state["iteration"] + 1,
        }

    return agent_node


def make_finalize_node(model_name: str, model_temperature: float, system_prompt: str):
    """Final answer synthesis from the gathered KB content.

    The agent's own last message is only a stop signal (and may be narration);
    the answer is generated here from the actual expand() results, so the stored
    answer is always a clean, grounded response.
    """
    llm = ChatDeepSeek(
        model=model_name,
        temperature=model_temperature,
        api_key=settings.DEEPSEEK_API_KEY,  # type: ignore
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", (
            "[GATHERED KNOWLEDGE BASE CONTENT]\n{context}\n\n"
            "Write the final answer to the user's question now, based on the gathered content. "
            "Start directly with the answer — no narration, no planning, and no references to "
            "tools or the exploration process. Never mention node ids (e.g. [010]); name the "
            "location (### path) of the content you used instead. If the question asks what "
            "exists, where something is, or for an overview, enumerate all the relevant "
            "locations you gathered across the whole knowledge base, not just one branch. "
            "If the content does not cover the question, say so explicitly and answer from "
            "general knowledge if appropriate.\n\n"
            "[USER QUESTION]\n{query}"
        )),
    ])
    chain = prompt | llm

    def finalize_node(state) -> dict:
        context = "\n\n".join(
            m.content
            for m in state["messages"]
            if isinstance(m, ToolMessage) and m.content
        )
        result = chain.invoke({
            "context": context or "(no content was gathered)",
            "query": state["query"],
        })
        return {"answer": result.content}

    return finalize_node
