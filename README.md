# Personal AI Agent with Tools

> A hands-on learning project building a fully local AI agent capable of browsing the web, executing code, and understanding images and documents - my starting point for learning how LLMs, VLMs, and agentic AI actually work.

## What this is

A local AI agent that runs entirely on my machine - no cloud, no API costs, no data leaving my laptop. The agent accepts text, image, video, and document inputs, reasons over them using a local LLM, and can autonomously decide to use one of five tools. Every conversation is persisted locally so context is never lost between sessions.

## Architecture

![Architecture diagram](design/architecture.png)

The agent follows a structured flow: user input is classified by the input router, conversation history is retrieved from SQLite, the LangGraph agent reasons over the input and decides whether to invoke a tool, and the response is parsed, saved back to memory, and streamed to the user.

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Inference runtime | Ollama | Simple setup, GGUF support, Python API, runs fully offline |
| LLM (reasoning) | Qwen2.5 7B | Reliable tool calling, fits in 8GB VRAM |
| LLM (code) | Qwen2.5-Coder 7B | Specialized for code generation, separation of concerns |
| VLM (vision) | Qwen2.5-VL 7B | Best vision quality in VRAM budget, excellent OCR |
| Agent framework | LangChain + LangGraph | Production standard for stateful multi-step agents |
| Memory | SQLite via SqliteSaver | Persistent across restarts, crash-safe, zero infrastructure |
| Web search | DuckDuckGo via ddgs | Free, no API key required |
| Code sandbox | Docker | Network disabled, memory and CPU capped, isolated filesystem |

## Tools

| Tool | Trigger | Implementation |
|------|---------|---------------|
| `search_web` | news / current events queries | DuckDuckGo, top 3 results |
| `execute_code` | "run", "execute", code requests | Docker sandbox, --network none, 30s timeout |
| `analyze_image` | .jpg .png .gif .webp paths | Base64 encode -> VLM |
| `analyze_video` | .mp4 .avi .mov .mkv paths | OpenCV frame sampling -> batched VLM call |
| `analyze_document` | .pdf .docx .xlsx .xls .csv paths | pdfplumber / python-docx / openpyxl / pandas -> LLM |

## Project structure

```
local-ai-agent/
├── research/
│   └── research.md        # runtime comparison, framework evaluation, model selection
├── design/
│   ├── design.md          # agent flow, component definitions, architecture
│   └── architecture.png
├── book/                  # chapter-by-chapter documentation of the build
│   ├── 00-introduction/
│   ├── 01-foundations/
│   ├── 02-langchain/
│   ├── 03-langgraph/
│   └── 04-building-the-agent/
├── docker/
│   └── Dockerfile.sandbox # isolated Python execution environment
└── src/
    ├── main.py            # entry point - session management and CLI loop
    ├── graph.py           # LangGraph graph wiring and compilation
    ├── nodes.py           # all node functions and edge routing functions
    ├── tools.py           # 5 tools: web search, code, image, video, document
    ├── models.py          # llm, vlm, and coder_llm instantiation
    ├── config.py          # env vars, model names, hyperparameters, pattern lists
    └── state.py           # AgentState TypedDict
```

## Status

- [x] Phase 1 - Research
- [x] Phase 2 - Design
- [x] Phase 3 - Code
  - [x] Basic LangGraph ReAct agent with streaming
  - [x] Persistent memory via SqliteSaver
  - [x] Input router (keyword pattern matching)
  - [x] Web search tool (DuckDuckGo)
  - [x] Code execution tool (Docker sandbox)
  - [x] Image analysis tool (VLM)
  - [x] Video analysis tool (OpenCV + VLM)
  - [x] Document analysis tool (PDF, Word, Excel, CSV)
  - [x] Code generation node (specialized coder LLM)
  - [x] Human-in-the-loop approval before code execution
  - [x] Output parser with retry loop
- [x] Phase 4 - Book documentation

## Hardware

- GPU: NVIDIA RTX 4060 (8GB VRAM)
- OS: Ubuntu 24
- Models run fully on-device via Ollama

> Note: LLM and VLM cannot coexist in 8GB VRAM. Ollama swaps models automatically with ~3-5s latency per swap.

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
ollama pull qwen2.5:7b        # LLM for reasoning and tool calling
ollama pull qwen2.5vl:7b      # VLM for image, video, and document analysis
ollama pull qwen2.5-coder:7b  # LLM for code generation
```

### 3. Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Build the Docker sandbox

```bash
docker build -f docker/Dockerfile.sandbox -t agent-sandbox .
```

### 5. Configure environment variables

Copy `.env` and edit as needed:

```env
LLM_MODEL=qwen2.5:7b
VLM_MODEL=qwen2.5vl:7b
CODER_MODEL=qwen2.5-coder:7b
NUM_CTX=8192
TEMPERATURE=0.1
NUM_PREDICT=1024
TOP_P=0.9
SANDBOX_IMAGE=agent-sandbox
MAX_FRAMES=8
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
| Coder (Qwen2.5-Coder 7B) | ~4.7 GB |
| All simultaneously | Not possible on 8GB - Ollama swaps automatically |
