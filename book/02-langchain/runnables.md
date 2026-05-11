# Runnables

---
<- [Pipe Operator](pipe-operator.md) | [Home](../README.md) | [03 LangGraph](../03-langgraph/README.md) ->

---

## What it is

A **Runnable** is the base interface that everything in LangChain implements. ChatOllama is a Runnable. ChatPromptTemplate is a Runnable. A chain created with `|` is a Runnable. An `@tool` function wrapped for use in a chain is a Runnable.

The interface defines three methods:

- `invoke(input)` - call once, wait for full result
- `batch(inputs)` - call many times in parallel
- `stream(input)` - call once, receive output token by token as a generator

## Why it matters

The Runnable interface is what makes the `|` operator work. Every component exposes the same input/output contract, so they can be composed freely. You don't need to know the internals of `ChatOllama` to chain it after a `ChatPromptTemplate`.

## How it works

In this project, `invoke` is the only method used directly at the chain level:

```python
# In agent_node - full blocking call
response = chain.invoke({
    "date": date.today().isoformat(),
    "messages": trim_messages_window(state["messages"]),
})
```

And in code_generation_node, the coder LLM is called as a Runnable directly:

```python
# In code_generation_node
response = coder_llm.invoke([HumanMessage(content=code_prompt)])
code = _strip_markdown(response.content)
```

Streaming happens at the LangGraph level, not the chain level. In `main.py`:

```python
for chunk, metadata in app.stream(
    {"messages": [message]},
    config=config,
    stream_mode="messages",
):
    if metadata.get("langgraph_node") == "agent_node":
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
```

`app.stream()` is LangGraph's streaming API. It yields message chunks as they arrive from the model. The `stream_mode="messages"` tells LangGraph to yield individual message tokens rather than full state snapshots.

## Gotchas and lessons learned

- **Streaming filters matter.** `app.stream()` yields chunks from every node - tool results, state updates, everything. The filter `metadata.get("langgraph_node") == "agent_node"` ensures only the final text response is printed to the user. Without this filter, you'd see raw ToolMessage content printed as output.
- **invoke vs stream is not always your choice.** LangGraph internally calls `invoke` on nodes. The streaming you see in the terminal is LangGraph streaming chunks up to the caller - the underlying chain call inside agent_node is still blocking.
- **batch is useful for offline processing.** If you wanted to run the agent against hundreds of test inputs, `batch` would parallelize the calls. Not used in this project but worth knowing.

---

<- [Pipe Operator](pipe-operator.md) | [Home](../README.md) | [03 LangGraph](../03-langgraph/README.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](README.md) · [Messages](messages.md) · [Prompt Templates](prompt-templates.md) · [Tools](tools.md) · [Pipe Operator](pipe-operator.md) · [Runnables](runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
