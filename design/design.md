# Design phase

## Agent flow — how a message is processed end to end

1. **Input routing**: a design pattern used to dynamically direct user input to the most appropriate processing path, model, or specialized agent based on the content of the request
2. **Memory retrieval**: agent fetches conversation history from local SQLite DB that persists data even after system crashes. Retrieved automatically by LangGraph's SqliteSaver using the thread_id.
3. **Reasoning step (LangGraph node)**: LangGraph orchestrates 
   the agent loop as a graph. Each node is a component, each 
   edge is a conditional connection. LangGraph's SqliteSaver 
   checkpoints state at every step to SQLite.
4. **Tool selection**: if tool needed, agent selects between:
   - Web search tool: triggered when user asks for current information or recent events
   - Code execution tool: triggered when user asks to run, execute, or test code
   - Vision tool: triggered when input is image/video *(deferred to v2)*
5. **Tool execution**: the process of actively running a defined tool — a Python function decorated with `@tool` — after the LLM has decided it needs that tool to answer the user's request. Handled automatically by LangGraph's `ToolNode`.
6. **Output parsing + validation**: validates the agent's response before returning it to the user. If invalid, the agent retries up to 3 times before forcing an end.
7. **Memory update**: result stored back to SQLite via SqliteSaver automatically on every graph step.
8. **Response to user**: plain human readable text or image/video

## On memory and storage
- **What database / storage mechanism will you use and why?**
  SQLite local DB via LangGraph's `SqliteSaver`. Chosen over `ConversationBufferMemory` because it persists to disk — conversation history survives restarts and crashes.
- **What gets stored?**
  Full message history (every user message and agent response). `add_messages` reducer ensures append-only behavior — no overwrites. Summarization considered as future mitigation if context window is exceeded.
- **When is history retrieved?**
  Automatically on every `app.invoke()` call using the `thread_id` as the conversation key. LangGraph loads the checkpoint from SQLite before the first node runs.

## Known limitations
- **ConversationBufferMemory rejected**: stores history in RAM only, lost on restart. SQLite chosen instead.
- **SQLite context window**: if conversation history grows very large, it may exceed the model's context window (8192 tokens). Future mitigation: summarization or sliding window.
- **Output validation is minimal**: only empty-check validation implemented in v1. Business-logic validation intentionally kept minimal to avoid false positives (e.g. artifact detection broke valid code responses). Per-tool validation will be added as tools are built.
- **Code execution has no network access**: Docker sandbox runs with `--network none` by design. Code cannot install packages or make external requests. Only Python standard library available inside the container. Pre-built image with common packages deferred to v2.
- **Model hallucination**: Llama 3.2 3B sometimes ignores tool results and answers from training data. Larger model or fine-tuning considered for v2.
- **SSL certificate permission error**: non-blocking warning on web search related to system certificate permissions on Ubuntu 24. Does not affect functionality.

## Components

| Component | Responsibility | Input | Output | Status |
|-----------|---------------|-------|--------|--------|
| Input Router | route the user request to its dedicated model/agent | text/image/video | chosen model/agent | ❌ deferred v2 |
| Memory Manager | persist and retrieve conversation history from SQLite | user message + thread ID (read) / agent response (write) | conversation history (read) / confirmation (write) | ✅ built |
| Agent (LangGraph graph) | orchestrate the full agent loop as a stateful graph with conditional edges | user message + conversation history | final response or tool call | ✅ built |
| Tool Manager — web search | search the web for current information via DuckDuckGo | search query string | text results (title, url, summary) | ✅ built |
| Tool Manager — code execution | execute Python code safely in a Docker sandbox | Python code string | stdout or error message | ✅ built |
| Tool Manager — vision | analyze images and video using VLM | image/video file | text description | ❌ deferred v2 |
| Output Parser | validate agent response, retry up to 3 times if invalid | last AI message | is_valid bool + retry count | ✅ built |

## Architecture

![Architecture diagram](architecture.png)

### Key design decisions reflected in the diagram
- Input Router sits first — separates text from vision before 
  anything else *(not yet implemented)*
- Memory Manager is called twice — retrieve before agent, 
  save after output parser
- Tool Manager is optional path — agent decides whether to 
  invoke it via conditional edge
- Ollama serves both models — swapped depending on task type
- Output Parser always runs — even if no tool was used
- Retry loop — output parser can route back to agent node up to 3 times

## LLM hyperparameters

| Parameter | What it controls | Value | Why |
|-----------|-----------------|-------|-----|
| temperature | randomness of model output | 0.1 | low value for precise reasoning, consistent tool-calling format, and structured outputs |
| num_ctx | max number of tokens in context window | 8192 | covers conversation history + tool results without excessive VRAM overhead |
| top_p | cumulative probability threshold for nucleus sampling | 0.9 | standard default, filters low-probability tokens |
| num_predict | maximum number of tokens to generate | 1024 | enough room for step-by-step reasoning without runaway generation |

## Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Inference runtime | Ollama | local, simple setup, GGUF support, Python API, fully offline |
| LLM | Llama 3.2 3B | designed for agentic tool use, fits in 8GB VRAM alongside VLM |
| VLM | Qwen2.5-VL 7B *(deferred)* | best vision quality in VRAM budget, excellent OCR and document parsing |
| Agent framework | LangChain + LangGraph | production standard, stateful graph, conditional edges, SqliteSaver |
| Memory | SQLite via SqliteSaver | persistent across restarts, crash-safe, zero infrastructure |
| Web search | DuckDuckGo via ddgs | free, no API key, good results |
| Code sandbox | Docker | industry standard isolation, network disabled, memory and CPU capped |

## V1 implementation — scope

### Built in v1
- Agent node (LangGraph StateGraph)
- Memory Manager (SqliteSaver, persistent SQLite)
- System prompt (injected at runtime, not stored in state)
- Output Parser (retry loop, up to 3 retries, empty-check validation)
- Web search tool (`@tool` decorator, DuckDuckGo, runs on host)
- Code execution tool (`@tool` decorator, Docker sandbox, `--network none`, 30s timeout)
- Full ReAct loop (conditional edges: agent → tool → agent or agent → parser → end/retry)

### Deferred to v2
- Input Router (text vs vision detection)
- Vision tool (Qwen2.5-VL 7B via Ollama)
- Gradio UI
- Streaming responses
- Human-in-the-loop (interrupt_before)
- Docker image with pre-installed packages for code execution
- Context window summarization for long conversations