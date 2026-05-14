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

    # tool_node always loops back to the agent so it can reason about the tool result
    graph.add_edge("tool_node", "agent_node")
    # code_generation_node always proceeds to human approval before any code runs
    graph.add_edge("code_generation_node", "human_approval_node")

    # Entry point dispatch: routes code requests to the coder model, all other types to the agent
    # should_route reads input_type from state
    # returns "code_generation_node" when input_type is "code"
    # returns "agent_node" for "general", "media", and all "document_*" types
    graph.add_conditional_edges(
        "input_router_node",
        should_route,
        {
            "code_generation_node": "code_generation_node",
            "agent_node": "agent_node",
        },
    )

    # Human review dispatch: runs the tool on approval, sends the agent back to rewrite on rejection
    # should_execute_tool reads approved from state
    # returns "tool_node" when approved is True
    # returns "agent_node" when approved is False so the injected feedback message drives a rewrite
    graph.add_conditional_edges(
        "human_approval_node",
        should_execute_tool,
        {"tool_node": "tool_node", "agent_node": "agent_node"},
    )

    # Output validation dispatch: retries the agent on empty responses, terminates on valid output or exhaustion
    # should_retry reads is_valid and retry_count from state
    # returns "retry" when is_valid is False and retry_count is below 3
    # returns "end" when is_valid is True or retry_count has reached 3
    graph.add_conditional_edges(
        "output_parser_node",
        should_retry,
        {"retry": "agent_node", "end": END},
    )

    # Agent dispatch: four paths based on what the LLM decided to do next
    # should_use_tool inspects the last message in state
    # returns "code_generation_node" when the agent calls execute_code (routes through coder model first)
    # returns "tool_node" when the agent calls any other tool such as search_web or analyze_image
    # returns "output_parser_node" when the agent produced a text response with no tool call
    graph.add_conditional_edges(
        "agent_node",
        should_use_tool,
        {
            "code_generation_node": "code_generation_node",
            "tool_node": "tool_node",
            "output_parser_node": "output_parser_node",
        },
    )

    return graph.compile(checkpointer=memory)
