# Building a Local AI Agent from Scratch

---
<- *(start)* | [Home](README.md) | [Introduction](00-introduction/README.md) ->

---

This book documents the actual process of building a fully local AI agent - one that can browse the web, execute Python code, analyze images and documents, and remember every conversation - without sending a single byte to the cloud.

## Who this is for

You don't need an AI background. I didn't have one either. I'm a robotics engineer who knew C++ and Python but had never touched LLMs, agents, or anything in this stack before this project.

If you can write Python and you're curious how AI agents actually work under the hood - not the marketing version, the real one - this book is for you.

## What you'll build

A terminal-based AI agent running entirely on your machine:

- Powered by Qwen2.5 7B via Ollama (no API key, no cloud)
- Persistent memory via SQLite (conversations survive restarts)
- 5 tools: web search, code execution, image analysis, video analysis, document analysis
- Safe code execution inside a Docker sandbox
- Human-in-the-loop approval before any code runs
- Stateful conversation graph via LangGraph

## How to use this book

Read in order. Each chapter builds on the previous one.

| Chapter | What you'll learn |
|---------|-------------------|
| [00 - Introduction](00-introduction/README.md) | Why I built this, what I got wrong, what I learned |
| [01 - Foundations](01-foundations/README.md) | Quantization, VRAM, tokens, context windows |
| [02 - LangChain](02-langchain/README.md) | Messages, prompts, tools, the pipe operator |
| [03 - LangGraph](03-langgraph/README.md) | State graphs, memory, human-in-the-loop |
| [04 - Building the Agent](04-building-the-agent/README.md) | The full build, the bugs, the fixes |

## What I wish I had

When I started this project, I couldn't find a single resource that explained all of this together in a beginner-friendly way. Most tutorials either glossed over the infrastructure (just use OpenAI) or drowned in theory without touching real code.

This book is what I wish existed when I started.

---

<- *(start)* | [Home](README.md) | [Introduction](00-introduction/README.md) ->

**All pages:** [Home](README.md) · [Introduction](00-introduction/README.md) · [01 Foundations](01-foundations/README.md) · [Quantization](01-foundations/quantization.md) · [LLM Basics](01-foundations/llm-basics.md) · [02 LangChain](02-langchain/README.md) · [Messages](02-langchain/messages.md) · [Prompt Templates](02-langchain/prompt-templates.md) · [Tools](02-langchain/tools.md) · [Pipe Operator](02-langchain/pipe-operator.md) · [Runnables](02-langchain/runnables.md) · [03 LangGraph](03-langgraph/README.md) · [State](03-langgraph/state.md) · [Nodes & Edges](03-langgraph/nodes-edges.md) · [Conditional Edges](03-langgraph/conditional-edges.md) · [Memory](03-langgraph/memory.md) · [Human-in-the-Loop](03-langgraph/human-in-the-loop.md) · [04 Building the Agent](04-building-the-agent/README.md) · [V1 Basic Agent](04-building-the-agent/v1-basic-agent.md) · [Input Router](04-building-the-agent/input-router.md) · [Tools & ReAct Loop](04-building-the-agent/tools-react-loop.md) · [Code Generation Node](04-building-the-agent/code-generation-node.md) · [Docker Sandbox](04-building-the-agent/docker-sandbox.md)
