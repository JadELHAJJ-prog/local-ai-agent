# Introduction - My story, why I built this, what I learned

---
<- [Home](../README.md) | [Home](../README.md) | [01 Foundations](../01-foundations/README.md) ->

---

## Why I built this

I wanted to understand AI from the inside - not by reading papers or watching YouTube videos, but by building something that actually runs on real hardware and does real things.

The project became: build a fully local AI agent. Local LLM, local VLM, local vector DB eventually, everything on my RTX 4060 with 8GB VRAM.

## The process

I ran three structured phases before writing a single line of the actual agent:

**Phase 1 - Research**: I spent time comparing inference runtimes (Ollama vs vLLM), agent frameworks (LangGraph vs smolagents vs AutoGen), and models. Every decision went into `research/research.md`.

**Phase 2 - Design**: I designed the full agent flow on paper before coding. Component interfaces, memory strategy, graph structure, known limitations - all documented in `design/design.md` before implementation. This saved me from building the wrong thing.

**Phase 3 - Build**: I built it, hit bugs, fixed them, and documented everything here.

## What I got wrong

**Llama 3.2 3B was a mistake.** I started with Llama 3.2 3B because it was smaller and faster. It couldn't reliably call tools in the structured format LangGraph expects. I switched to Qwen2.5 7B and the tool calling worked immediately. The lesson: model size and architecture matter a lot for tool use. Don't optimize for speed before you have something working.

**Routing directly to ToolNode crashed.** My input router sent documents and media directly to LangGraph's ToolNode. ToolNode needs a preceding AIMessage with tool_calls - it doesn't know what to do with a raw HumanMessage. The crash was a `ValueError: No AIMessage found in input`. The fix was one line: route everything through agent_node first.

**The double approval box.** When a user rejects code and the agent rewrites it, LangGraph replays the interrupted node. This caused the approval prompt to appear twice. It's a known LangGraph interrupt() replay behavior. I left it as a known limitation for v2 rather than fighting the framework.

**The session bug.** When a user started a new session and immediately asked about a document, it crashed. When they continued an existing session, it worked because the document path was already in checkpoint history. The root cause was the routing bug above, but it took a while to figure out why behavior differed between new and existing sessions.

## What I actually learned

- **LLMs are just statistical text predictors with a structured output format.** The "intelligence" in an AI agent is mostly the framework around the model - the prompt, the tool definitions, the graph structure.
- **The docstring of a tool is its entire interface.** The LLM reads the docstring to decide when and how to use the tool. A bad docstring = a tool that never gets called correctly.
- **State management is the hard part.** Getting the model to respond is easy. Getting state to flow correctly through a graph, with memory, with human interrupts, with retries - that's where the real complexity lives.
- **Start simpler than you think you need to.** My first working agent was 50 lines. Each feature I added broke something I thought was solid.

## The tech stack I landed on

After the research phase:
- **Ollama** for local inference (simpler than vLLM, GGUF, fully offline)
- **LangChain + LangGraph** for the agent framework (production standard, stateful graphs)
- **Qwen2.5 7B** for reasoning and tool calling (better than Llama 3.2 3B)
- **Qwen2.5-Coder 7B** for code generation (specialized model, separation of concerns)
- **Qwen2.5-VL 7B** for vision (best quality in 8GB VRAM budget)
- **SQLite** for persistent memory (zero infrastructure, crash-safe)
- **Docker** for safe code execution (network disabled, memory capped)

---

<- [Home](../README.md) | [Home](../README.md) | [01 Foundations](../01-foundations/README.md) ->

**All pages:** [Home](../README.md) · [Introduction](README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
