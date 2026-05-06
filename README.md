# Personal AI Agent with Tools

> A hands-on learning project building a fully local AI agent capable of browsing the web, executing code, and understanding images — my starting point for learning how LLMs, VLMs, and agentic AI actually work.

## What this is

A local AI agent that runs entirely on my machine — no cloud, no API costs, no data leaving my laptop. The agent accepts text and image inputs, reasons over them using a local LLM, and can autonomously decide to use one of three tools: web search, Python code execution, or visual image analysis. Every conversation is persisted locally so context is never lost between sessions.

## Architecture

![Architecture diagram](design/architecture.png)

The agent follows a structured flow: user input is routed by type, conversation history is retrieved from a local SQLite database, the LangGraph agent reasons over the input and decides whether to invoke a tool, and the response is parsed, saved back to memory, and returned to the user.

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Inference runtime | Ollama | Simple setup, GGUF support, Python API, runs fully offline |
| LLM | Llama 3.2 3B | Designed for agentic tool use, fits in 8GB VRAM alongside VLM |
| VLM | Qwen2.5-VL 7B | Best vision quality in VRAM budget, excellent OCR and document parsing |
| Agent framework | LangChain + LangGraph | Production standard for stateful multi-step agents |
| Memory | SQLite via SqliteSaver | Persistent across restarts, crash-safe, zero infrastructure |

## Project structure

```
local-ai-agent/
├── research/
│   └── research.md        # runtime comparison, framework evaluation, model selection
├── design/
│   ├── design.md          # agent flow, component definitions, architecture
│   └── architecture.png   # system architecture diagram
└── src/                   # coming soon
```

## Research and design

Before writing any code I went through two structured phases:

- [`research/research.md`](research/research.md) — comparison of inference runtimes, agent frameworks, and models. Every decision is documented with rationale and tradeoffs.
- [`design/design.md`](design/design.md) — full agent flow, component interface definitions, memory strategy, known limitations, and architecture diagram.

## Status

- [x] Phase 1 — Research
- [x] Phase 2 — Design
- [x] Phase 3 — Code (in progress)
  - [x] Basic LangGraph ReAct agent
  - [x] Persistent memory via SqliteSaver
  - [x] Web search tool (DuckDuckGo)
  - [x] Code execution tool (Docker sandbox)
  - [ ] Vision tool (VLM)

## Hardware

- GPU: NVIDIA RTX 4060 (8GB VRAM)
- OS: Ubuntu 24
- Models run fully on-device via Ollama