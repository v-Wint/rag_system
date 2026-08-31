from langchain_core.messages import ToolMessage


def make_tools_node(tool):
    def tools_node(state) -> dict:
        last = state["messages"][-1]
        outputs = []
        expanded_ids = []
        for call in last.tool_calls:
            args = call.get("args") or {}
            result = tool.invoke(args)
            outputs.append(ToolMessage(content=result, tool_call_id=call["id"]))
            ids = args.get("node_ids") or []
            expanded_ids.extend(ids if isinstance(ids, list) else [ids])

        return {
            "messages": outputs,
            "debug": [{"expanded_ids": expanded_ids}],
        }

    return tools_node
