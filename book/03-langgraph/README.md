# LangGraph - Stateful agent graphs

---
<- [Runnables](../02-langchain/runnables.md) | [Home](../README.md) | [State](state.md) ->

---

LangGraph is the framework that turns a collection of functions into a stateful, resumable, interruptible agent. It's the most important piece of infrastructure in this project.

I didn't fully understand why I needed LangGraph until I tried to add memory and human-in-the-loop approval without it. The answer: you can technically do it without LangGraph, but you'd spend weeks rebuilding what LangGraph gives you for free.

LangGraph provides:
- A **state machine** where each node is a function and edges define the flow
- Automatic **checkpointing** - state is saved to SQLite after every node
- **Conditional routing** - the graph decides its own path based on state
- **Human-in-the-loop** - the graph can pause at any point, persist its state, and resume later

## In this chapter

- [state.md](state.md) - TypedDict, add_messages reducer, why reducers matter
- [nodes-edges.md](nodes-edges.md) - nodes as functions, fixed edges, entry point
- [conditional-edges.md](conditional-edges.md) - routing functions, add_conditional_edges
- [memory.md](memory.md) - SqliteSaver, thread_id, checkpointing
- [human-in-the-loop.md](human-in-the-loop.md) - interrupt(), Command(resume), state replay

---

<- [Runnables](../02-langchain/runnables.md) | [Home](../README.md) | [State](state.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](README.md) · [State](state.md) · [Nodes & Edges](nodes-edges.md) · [Conditional Edges](conditional-edges.md) · [Memory](memory.md) · [Human-in-the-Loop](human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
