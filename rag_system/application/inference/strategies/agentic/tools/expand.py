from langchain_core.tools import tool

from ..render import location, render_expanded


def make_expand_tool(index, max_expand_size: int, preview_length: int):
    @tool
    def expand(node_ids: list[str] = []) -> str:
        """Navigate the knowledge-base tree.

        Expand one or more nodes by their id. Call with an empty list to see
        the top-level overview of the whole knowledge base. Internal (branch)
        ids return a size-bounded listing of their children; leaf ids return
        the node's full content. Every section starts with a location header
        (`### Path > To > Node`) giving the node's path. Ids are stable across
        calls.
        """
        ids = node_ids or [""]
        sections = []
        for node_id in ids:
            node = index.get(node_id)
            if node is None:
                sections.append(f"[ERROR] Unknown node id: {node_id}")
                continue
            header = "### " + location(node)
            if node.children:
                body = render_expanded(node, max_expand_size, preview_length)
            else:
                body = node.text
            sections.append(header + "\n" + body)
        return "\n\n".join(sections)

    return expand
