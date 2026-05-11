# Building the Agent

---
<- [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) | [Home](../README.md) | [V1 Basic Agent](v1-basic-agent.md) ->

---

This chapter documents the actual build - not the polished final version, but the sequence of decisions, additions, and fixes that got there.

I built this incrementally. Each section describes one layer added on top of the previous one, and the problems that came with it.

## Build order

1. **v1-basic-agent** - a working agent with one node, one LLM call, no tools
2. **Input router** - classify input before hitting the LLM, add document support
3. **Tools + ReAct loop** - add the 5 tools, wire the tool execution cycle
4. **Code generation node** - add the specialized coder LLM and the two-path code flow
5. **Docker sandbox** - safe code execution environment

## In this chapter

- [v1-basic-agent.md](v1-basic-agent.md) - first working agent, streaming
- [input-router.md](input-router.md) - why we added it, CODE_PATTERNS, routing logic, the crash
- [tools-react-loop.md](tools-react-loop.md) - ReAct pattern, ToolNode, tool flow
- [code-generation-node.md](code-generation-node.md) - coder model, two paths, _strip_markdown
- [docker-sandbox.md](docker-sandbox.md) - why Docker, security flags, Dockerfile.sandbox

---

<- [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) | [Home](../README.md) | [V1 Basic Agent](v1-basic-agent.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](README.md) · [V1 Basic Agent](v1-basic-agent.md) · [Input Router](input-router.md) · [Tools & ReAct Loop](tools-react-loop.md) · [Code Generation Node](code-generation-node.md) · [Docker Sandbox](docker-sandbox.md)
