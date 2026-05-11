# The Pipe Operator

---
<- [Tools](tools.md) | [Home](../README.md) | [Runnables](runnables.md) ->

---

## What it is

The `|` operator in LangChain creates a **RunnableSequence** - a chain where the output of the left side becomes the input of the right side.

```python
chain = prompt | llm_with_tools
```

This is not Python's bitwise OR. LangChain overloads `__or__` on its Runnable objects to create this composition syntax.

## Why it matters

Without the pipe operator, you'd manually call each step and pass the output:

```python
# Without pipe:
formatted = prompt.invoke({"date": "2025-01-01", "messages": [...]})
response = llm_with_tools.invoke(formatted)

# With pipe:
chain = prompt | llm_with_tools
response = chain.invoke({"date": "2025-01-01", "messages": [...]})
```

The pipe version is cleaner and composes arbitrarily - you can chain as many steps as you want:

```python
chain = prompt | llm | output_parser | formatter
```

Each component in the chain handles one transformation.

## How it works

In `nodes.py`, the agent uses a two-component chain:

```python
# src/nodes.py
def agent_node(state: AgentState) -> dict:
    chain = prompt | llm_with_tools
    response = chain.invoke(
        {
            "date": date.today().isoformat(),
            "messages": trim_messages_window(state["messages"]),
        }
    )
    return {"messages": [response]}
```

When `chain.invoke(...)` is called:

1. **prompt** receives the dict `{"date": "...", "messages": [...]}` and produces a list of formatted messages (with the system prompt and conversation history combined)
2. **llm_with_tools** receives that list of messages and produces an AIMessage (either a text response or a tool call)

The agent_node then wraps the AIMessage in a dict and returns it to LangGraph.

## Gotchas and lessons learned

- **The chain is created fresh inside agent_node.** I could define `chain = prompt | llm_with_tools` at module level, but defining it inside the function is fine and avoids any state issues.
- **chain.invoke() is blocking.** It waits for the full response before returning. For streaming, you use `chain.stream()` instead - but LangGraph's `app.stream()` handles the streaming at the graph level, so you don't need to stream the chain directly.
- **Type compatibility.** The pipe operator works because both `prompt` and `llm_with_tools` implement the Runnable interface. If you try to pipe something that isn't a Runnable, you get a runtime error.

---

<- [Tools](tools.md) | [Home](../README.md) | [Runnables](runnables.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](README.md) · [Messages](messages.md) · [Prompt Templates](prompt-templates.md) · [Tools](tools.md) · [Pipe Operator](pipe-operator.md) · [Runnables](runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
