# Nodes and Edges

---
<- [State](state.md) | [Home](../README.md) | [Conditional Edges](conditional-edges.md) ->

---

## What it is

A LangGraph graph is made of **nodes** (functions that process state) and **edges** (connections that define the flow between nodes). Together they form a directed graph - or in this project, a directed graph with cycles (the ReAct loop).

## Nodes

A node is any Python function that takes the state as input and returns a partial state update:

```python
# src/nodes.py
def agent_node(state: AgentState) -> dict:
    chain = prompt | llm_with_tools
    response = chain.invoke({
        "date": date.today().isoformat(),
        "messages": trim_messages_window(state["messages"]),
    })
    return {"messages": [response]}
```

That's it. No special base class, no decorator. Just a function. LangGraph calls it with the current state and merges its return value into the state.

Nodes are registered in `graph.py`:

```python
# src/graph.py
graph = StateGraph(AgentState)

graph.add_node("input_router_node", input_router_node)
graph.add_node("agent_node", agent_node)
graph.add_node("output_parser_node", output_parser_node)
graph.add_node("tool_node", ToolNode(tools))
graph.add_node("human_approval_node", human_approval_node)
graph.add_node("code_generation_node", code_generation_node)
```

`ToolNode(tools)` is LangGraph's prebuilt node that automatically handles tool execution. You pass it your list of tools and it takes care of finding the right tool_call in the AIMessage and executing it.

## Fixed edges

A fixed edge is an unconditional connection - node A always goes to node B:

```python
# src/graph.py
graph.add_edge("tool_node", "agent_node")
graph.add_edge("code_generation_node", "human_approval_node")
```

After `tool_node` executes a tool, it always goes back to `agent_node`. The agent needs to see the tool result and generate a response. This is the ReAct loop.

After `code_generation_node` finishes, it always goes to `human_approval_node`. Code generation always requires approval before execution.

## Entry point

The entry point is the first node that runs when a new message arrives:

```python
graph.set_entry_point("input_router_node")
```

Every message starts at `input_router_node`, which classifies the input and routes it to the right node.

## The full graph wiring

```python
# src/graph.py
graph.set_entry_point("input_router_node")

# Fixed edges
graph.add_edge("tool_node", "agent_node")
graph.add_edge("code_generation_node", "human_approval_node")

# Conditional edges (see conditional-edges.md for details)
graph.add_conditional_edges("input_router_node", should_route, {...})
graph.add_conditional_edges("human_approval_node", should_execute_tool, {...})
graph.add_conditional_edges("output_parser_node", should_retry, {...})
graph.add_conditional_edges("agent_node", should_use_tool, {...})

return graph.compile(checkpointer=memory)
```

## Gotchas and lessons learned

- **ToolNode is special.** Unlike other nodes, `ToolNode` is a prebuilt LangGraph component that inspects the last AIMessage in state for tool_calls and executes them. This means it **requires** the message immediately before it to be an AIMessage with tool_calls. If a HumanMessage is the last message in state when ToolNode runs, it crashes with `ValueError: No AIMessage found in input`. This is the exact bug I fixed in the input router - routing directly to ToolNode without going through agent_node first.
- **Cycles are allowed.** Unlike a DAG, LangGraph graphs can have cycles. The `tool_node -> agent_node -> tool_node` pattern is a cycle. LangGraph handles this fine - the cycle terminates when `should_use_tool` returns `"output_parser_node"` instead of `"tool_node"`.
- **graph.compile() is where checkpointing is attached.** The `checkpointer=memory` argument in `graph.compile()` is what enables persistent memory. Without it, the graph runs but state is not saved between calls.

---

<- [State](state.md) | [Home](../README.md) | [Conditional Edges](conditional-edges.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](README.md) · [State](state.md) · [Nodes & Edges](nodes-edges.md) · [Conditional Edges](conditional-edges.md) · [Memory](memory.md) · [Human-in-the-Loop](human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
