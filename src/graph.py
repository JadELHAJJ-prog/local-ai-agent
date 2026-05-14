from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from state import AgentState
from tools import tools
from nodes import (
    input_router_node, should_route,
    agent_node, should_use_tool,
    code_generation_node,
    human_approval_node, should_execute_tool,
    output_parser_node, should_retry,
)


# Wire all nodes and conditional edges; SQLite checkpointer enables cross-session state persistence
def build_graph(memory: SqliteSaver):
    graph = StateGraph(AgentState)

    graph.add_node("input_router_node", input_router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("output_parser_node", output_parser_node)
    graph.add_node("tool_node", ToolNode(tools))
    graph.add_node("human_approval_node", human_approval_node)
    graph.add_node("code_generation_node", code_generation_node)

    graph.set_entry_point("input_router_node")

    graph.add_edge("tool_node", "agent_node")
    graph.add_edge("code_generation_node", "human_approval_node")

    # Route code requests directly to code_generation_node, everything else to agent_node
    graph.add_conditional_edges(
        "input_router_node",
        should_route,
        {
            "code_generation_node": "code_generation_node",
            "agent_node": "agent_node",
        },
    )
    # Run the tool if the human approved, send back to agent if rejected so it can rewrite
    graph.add_conditional_edges(
        "human_approval_node",
        should_execute_tool,
        {"tool_node": "tool_node", "agent_node": "agent_node"},
    )
    # Retry via agent_node on empty output, terminate once valid or retries exhausted
    graph.add_conditional_edges(
        "output_parser_node",
        should_retry,
        {"retry": "agent_node", "end": END},
    )
    # Fan out from agent_node to four possible next steps depending on what the LLM requested
    graph.add_conditional_edges(
        "agent_node",
        should_use_tool,
        {
            "code_generation_node": "code_generation_node",  # execute_code called, code not yet improved
            "human_approval_node": "human_approval_node",    # execute_code called, code ready for review
            "tool_node": "tool_node",                        # any other tool called, run it directly
            "output_parser_node": "output_parser_node",      # no tool call, validate the text response
        },
    )

    return graph.compile(checkpointer=memory)
