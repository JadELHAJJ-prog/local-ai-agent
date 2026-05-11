# LangChain - The building blocks

---
<- [LLM Basics](../01-foundations/llm-basics.md) | [Home](../README.md) | [Messages](messages.md) ->

---

LangChain is the layer between your Python code and the LLM. It provides a unified interface for messages, prompts, and tools - so you can swap models without rewriting your logic.

In this project I use LangChain for:
- Structuring messages (HumanMessage, AIMessage, ToolMessage)
- Building the system prompt (ChatPromptTemplate, MessagesPlaceholder)
- Defining tools (@tool decorator, bind_tools)
- Composing the agent chain (the pipe | operator)

I don't use LangChain for memory or graph orchestration - that's LangGraph's job.

## In this chapter

- [messages.md](messages.md) - HumanMessage, AIMessage, ToolMessage, add_messages
- [prompt-templates.md](prompt-templates.md) - ChatPromptTemplate, MessagesPlaceholder
- [tools.md](tools.md) - @tool decorator, docstrings, bind_tools
- [pipe-operator.md](pipe-operator.md) - | operator, RunnableSequence, chain.invoke
- [runnables.md](runnables.md) - The Runnable interface that everything shares

---

<- [LLM Basics](../01-foundations/llm-basics.md) | [Home](../README.md) | [Messages](messages.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](README.md) · [Messages](messages.md) · [Prompt Templates](prompt-templates.md) · [Tools](tools.md) · [Pipe Operator](pipe-operator.md) · [Runnables](runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
