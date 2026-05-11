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
│   └── research.md    # runtime comparison, framework evaluation, model selection
├── design/
│   ├── design.md      # agent flow, component definitions, architecture
│   └── architecture.png
└── src/
    ├── main.py        # entry point — session management and CLI loop
    ├── graph.py       # LangGraph graph wiring and compilation
    ├── nodes.py       # all node functions and edge decision functions
    ├── tools.py       # web search, code execution, image and video analysis tools
    ├── models.py      # llm, vlm, and coder_llm instantiation
    ├── config.py      # env vars, model names, hyperparameters, pattern lists
    └── state.py       # AgentState TypedDict
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
  - [x] Vision tool (VLM)

## Hardware

- GPU: NVIDIA RTX 4060 (8GB VRAM)
- OS: Ubuntu 24
- Models run fully on-device via Ollama

## Installation

### Prerequisites

- Ubuntu 24 (or any Linux distro)
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060)
- NVIDIA drivers installed
- Python 3.11+
- Git
- Docker

### 1. Clone the repository

```bash
git clone https://github.com/your-username/local-ai-agent.git
cd local-ai-agent
```

### 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the required models:

```bash
ollama pull qwen2.5:7b       # LLM for reasoning and tool calling
ollama pull qwen2.5vl:7b     # VLM for image and video analysis
ollama pull qwen2.5-coder:7b # LLm for code generation
```

### 3. Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Build the Docker sandbox

Used for safe code execution:

```bash
docker build -f docker/Dockerfile.sandbox -t agent-sandbox .
```

### 5. Configure environment variables

Edit `.env` with your settings:

```env
LLM_MODEL=qwen2.5:7b
VLM_MODEL=qwen2.5vl:7b
NUM_CTX=8192
TEMPERATURE=0.1
NUM_PREDICT=1024
TOP_P=0.9
SANDBOX_IMAGE=agent-sandbox
MAX_FRAMES=3
```

### 6. Run the agent

```bash
cd src
python main.py
```

### VRAM requirements

| Component | VRAM |
|-----------|------|
| LLM (Qwen2.5 7B) | ~4.7 GB |
| VLM (Qwen2.5-VL 7B) | ~4.5 GB |
| LLM (Qwen2.5-coder 7B) | ~4.7 GB |
| All simultaneously | Not possible on 8GB — Ollama swaps automatically |