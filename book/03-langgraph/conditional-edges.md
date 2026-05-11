# Conditional Edges

---
<- [Nodes & Edges](nodes-edges.md) | [Home](../README.md) | [Memory](memory.md) ->

---

## What it is

A **conditional edge** is a connection where the destination node depends on the current state. Instead of always going from A to B, the graph calls a routing function that returns a key, and a mapping translates that key to a destination node.

This is the mechanism that gives the agent its branching behavior - deciding whether to use a tool, retry, approve code, or end.

## How it works

The signature:

```python
graph.add_conditional_edges(
    source_node,     # which node's output triggers this routing
    routing_fn,      # function that takes state and returns a string key
    {
        "key1": "destination_node_1",
        "key2": "destination_node_2",
    }
)
```

## The four routing functions

**1. should_route** - runs after input_router_node

```python
# src/nodes.py
def should_route(state: AgentState) -> str:
    input_type = state.get("input_type", "general")
    if input_type == "code":
        return "code_generation_node"
    return "agent_node"
```

Code requests go to code_generation_node. Everything else (general, media, document) goes through agent_node. The agent will call the appropriate tool on its own.

**2. should_use_tool** - runs after agent_node

```python
def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        if tool_name == "execute_code":
            if not state.get("code_generated", False):
                return "code_generation_node"
            return "human_approval_node"
        return "tool_node"
    return "output_parser_node"
```

This is the ReAct loop core. If the agent produced tool_calls, figure out which path. If the agent produced text with no tool call, go validate the output.

**3. should_execute_tool** - runs after human_approval_node

```python
def should_execute_tool(state: AgentState) -> str:
    feedback = state.get("human_feedback", "").lower()
    if any(phrase in feedback for phrase in APPROVAL_PHRASES):
        return "tool_node"
    return "agent_node"
```

APPROVAL_PHRASES from config.py: `["yes", "i like", "looks good", "approved", "ok", "good", "run it", "execute"]`. Approved -> execute the code. Rejected -> back to agent with feedback to rewrite.

**4. should_retry** - runs after output_parser_node

```python
def should_retry(state: AgentState) -> str:
    if state["is_valid"]:
        return "end"
    if state.get("retry_count", 0) >= 3:
        return "end"
    return "retry"
```

Mapped in graph.py as `{"retry": "agent_node", "end": END}`. Up to 3 retries if the response is empty.

## The full conditional edge wiring

```python
# src/graph.py
graph.add_conditional_edges(
    "input_router_node",
    should_route,
    {
        "code_generation_node": "code_generation_node",
        "agent_node": "agent_node",
    },
)
graph.add_conditional_edges(
    "human_approval_node",
    should_execute_tool,
    {"tool_node": "tool_node", "agent_node": "agent_node"},
)
graph.add_conditional_edges(
    "output_parser_node",
    should_retry,
    {"retry": "agent_node", "end": END},
)
graph.add_conditional_edges(
    "agent_node",
    should_use_tool,
    {
        "code_generation_node": "code_generation_node",
        "human_approval_node": "human_approval_node",
        "tool_node": "tool_node",
        "output_parser_node": "output_parser_node",
    },
)
```

## Gotchas and lessons learned

- **The routing function key must be in the mapping.** If `should_route` returns `"tool_node"` but the mapping doesn't have a `"tool_node"` key, LangGraph raises a KeyError at runtime. After I removed `"tool_node"` as a route from `input_router_node`, I also removed it from the mapping in `graph.py`.
- **Routing functions are pure state reads.** They don't call the LLM, they don't have side effects. They just inspect the state and return a string. Keep them simple.
- **END is a special constant.** Import it from `langgraph.graph`: `from langgraph.graph import StateGraph, END`. Using `"end"` as a string key that maps to `END` is the pattern.
- **should_use_tool is the ReAct loop controller.** The fact that `agent_node` can route back to `tool_node` (via should_use_tool), and `tool_node` has a fixed edge back to `agent_node`, creates the ReAct loop. The agent reasons, acts (calls tool), sees the result, reasons again, acts again - until it has enough information to produce a final answer.

---

<- [Nodes & Edges](nodes-edges.md) | [Home](../README.md) | [Memory](memory.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](README.md) · [State](state.md) · [Nodes & Edges](nodes-edges.md) · [Conditional Edges](conditional-edges.md) · [Memory](memory.md) · [Human-in-the-Loop](human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
