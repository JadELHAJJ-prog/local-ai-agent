# Design phase

## Agent flow - how a message is processed end to end

1. **Input routing**: a design pattern used to dynamically direct user input to the most appropriate processing path, model, or specialized agent based on the content of the request. Implemented via keyword pattern matching - no LLM call at this stage.

2. **Memory retrieval**: agent fetches conversation history from local SQLite DB that persists data even after system crashes. Retrieved automatically by LangGraph's SqliteSaver using the thread_id.

3. **Routing decision**:
   - `"code"` -> code_generation_node (coder LLM generates code, then human approval)
   - all other types (`"media"`, `"document_*"`, `"general"`) -> agent_node (main reasoning loop with tool calling)

4. **Reasoning step (LangGraph node)**: LangGraph orchestrates the agent loop as a graph. Each node is a component, each edge is a conditional connection. LangGraph's SqliteSaver checkpoints state at every step to SQLite.

5. **Tool selection**: if a tool is needed, the agent selects between:
   - Web search tool: triggered when user asks for current information or recent events
   - Code execution tool: triggered when user asks to run, execute, or test code
   - Vision tool (image): triggered when user provides an image path (.jpg .jpeg .png .gif .webp)
   - Vision tool (video): triggered when user provides a video path (.mp4 .avi .mov .mkv)

6. **Code path**: when code execution is requested (either via input router or agent decision), a dedicated coder LLM (`qwen2.5-coder:7b`) generates or improves the code before it reaches the execution tool. This separates reasoning from code quality.

7. **Human-in-the-loop**: before any `execute_code` call, the graph interrupts via LangGraph's `interrupt()`. The user sees the generated code and approves or rejects it with optional feedback. Rejection re-injects feedback as a HumanMessage so the agent rewrites and retries.

8. **Tool execution**: the process of actively running a defined tool - a Python function decorated with `@tool` - after the LLM has decided it needs that tool. Handled automatically by LangGraph's `ToolNode`.

9. **Output parsing + validation**: validates the agent's response before returning it to the user. If the response is empty, the agent retries up to 3 times before forcing an end.

10. **Memory update**: result stored back to SQLite via SqliteSaver automatically on every graph step.

11. **Response to user**: streamed token by token from the agent_node.

## Graph structure

```text
input_router_node (entry point)
    ├── "code"           -> code_generation_node -> human_approval_node
    │                                                 ├── approved -> tool_node -> agent_node
    │                                                 └── rejected -> agent_node (rewrite + retry)
    └── all other types -> agent_node
                              ├── execute_code call, code not yet generated -> code_generation_node
                              ├── execute_code call, code already generated -> human_approval_node
                              ├── other tool call -> tool_node -> agent_node (ReAct loop)
                              └── no tool call -> output_parser_node
                                                     ├── valid -> END
                                                     └── invalid (up to 3 retries) -> agent_node
```

## On memory and storage

- **What database / storage mechanism will you use and why?**
  SQLite local DB via LangGraph's `SqliteSaver`. Chosen over `ConversationBufferMemory` because it persists to disk - conversation history survives restarts and crashes.
- **What gets stored?**
  Full message history (every user message and agent response). `add_messages` reducer ensures append-only behavior - no overwrites.
- **When is history retrieved?**
  Automatically on every `app.invoke()` call using the `thread_id` as the conversation key. LangGraph loads the checkpoint from SQLite before the first node runs.
- **Session management**: at startup the user chooses to start a new session (new UUID) or continue the last one (loaded from `.last_session` file). The thread_id is the key that maps to a full conversation in SQLite.
- **Context window mitigation**: a sliding window trims state to the last 20 messages before each LLM call to avoid exceeding the 8192-token context limit. Full history still persists in SQLite.

## Known limitations

- **SQLite context window**: conversation history is trimmed to 20 messages per LLM call as a sliding window. Full history is preserved in SQLite but old context is not summarized.
- **Output validation is minimal**: only empty-check validation. Business-logic and format validation intentionally kept minimal to avoid false positives.
- **Code execution has no network access**: Docker sandbox runs with `--network none` by design. Pre-built image includes numpy, pandas, matplotlib, scipy, requests, pillow, sympy, statsmodels.
- **Input router false-positive risk**: pattern matching on 50+ keywords (e.g. "run", "implement", "build a") can misclassify non-code requests.
- **Model hallucination**: Qwen2.5 7B occasionally ignores tool results on complex chains. Larger model considered for future.
- **SSL certificate warning**: non-blocking warning on web search related to system certificate permissions on Ubuntu 24. Does not affect functionality.
- **Sequential tool calling**: Qwen2.5 7B calls one tool at a time. Multi-tool tasks require multiple ReAct cycles.
- **Video analysis is one batched VLM call**: all sampled frames are sent in a single message. Memory-efficient but limits per-frame temporal reasoning.
- **Document truncation**: extracted document text is truncated at 6000 characters. Long documents may have content cut off.
- **Double approval box on rejection**: LangGraph interrupt() replay behavior causes the approval prompt to appear twice when code is rejected. Deferred to v2.
- **Model swap latency**: LLM and VLM cannot coexist in 8GB VRAM. Ollama swaps automatically (~3-5s latency per swap).

## Components

| Component | Responsibility | Input | Output | Status |
|-----------|---------------|-------|--------|--------|
| Input Router | classify user input and route to the right node | last user message | input_type ("code" / "media" / "general") | ✅ built |
| Memory Manager | persist and retrieve conversation history from SQLite | user message + thread ID (read) / agent response (write) | conversation history (read) / confirmation (write) | ✅ built |
| Agent (LangGraph graph) | orchestrate the full agent loop as a stateful graph with conditional edges | user message + conversation history | final response or tool call | ✅ built |
| Code Generation Node | generate or improve Python code using a specialized coder LLM | user request or rough code from agent | AIMessage with execute_code tool_call | ✅ built |
| Human Approval Node | interrupt graph and get user approval before executing code | agent's tool call with code | human_feedback + routing decision | ✅ built |
| Tool Manager - web search | search the web for current information via DuckDuckGo | search query string | text results (title, url, summary) | ✅ built |
| Tool Manager - code execution | execute Python code safely in a Docker sandbox | Python code string | stdout or error message | ✅ built |
| Tool Manager - image analysis | analyze a static image using the VLM | image file path + question | text description | ✅ built |
| Tool Manager - video analysis | analyze a video by sampling frames and sending them to the VLM | video file path + question | text description | ✅ built |
| Tool Manager - document analysis | extract and analyze text from PDF, Word, Excel, CSV files | document file path + question | text summary | ✅ built |
| Output Parser | validate agent response, retry up to 3 times if invalid | last AI message | is_valid bool + retry count | ✅ built |
| Session Manager | manage new vs. resumed sessions via thread_id and .last_session file | user choice (1/2) | thread_id | ✅ built |

## Architecture

![Architecture diagram](architecture.png)

### Key design decisions reflected in the diagram

- Input Router sits first - pattern-matches user input before any LLM call
- Three models in the pipeline - qwen2.5:7b for reasoning, qwen2.5-coder:7b for code generation, qwen2.5vl:7b for vision
- Code generation node intercepts code requests - improves code quality before human review
- Human approval gate - LangGraph interrupt() pauses the graph before every code execution
- Memory Manager is called at every graph step - LangGraph SqliteSaver checkpoints continuously
- Tool Manager is optional path - agent decides whether to invoke it via conditional edge
- Output Parser always runs - even if no tool was used
- Retry loop - output parser can route back to agent node up to 3 times
- Streaming - agent_node responses are streamed token by token to the terminal

## LLM hyperparameters

| Parameter | What it controls | LLM value | VLM value | Why |
|-----------|-----------------|-----------|-----------|-----|
| temperature | randomness of model output | 0.1 | 0.1 | low value for precise reasoning and consistent tool-calling format |
| num_ctx | max number of tokens in context window | 8192 | 32768 | VLM needs larger window to handle multi-frame image content |
| top_p | cumulative probability threshold for nucleus sampling | 0.9 | 0.9 | standard default, filters low-probability tokens |
| num_predict | maximum number of tokens to generate | 1024 | 1024 | enough for step-by-step reasoning; coder LLM uses 2048 for longer code outputs |

## Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Inference runtime | Ollama | local, simple setup, GGUF support, Python API, fully offline |
| LLM (reasoning) | Qwen2.5 7B | reliable tool calling, fits 8GB VRAM |
| LLM (code) | Qwen2.5-Coder 7B | specialized for code generation and optimization |
| VLM (vision) | Qwen2.5-VL 7B | best vision quality in VRAM budget, excellent OCR and document parsing |
| Agent framework | LangChain + LangGraph | production standard, stateful graph, conditional edges, SqliteSaver, interrupt() |
| Memory | SQLite via SqliteSaver | persistent across restarts, crash-safe, zero infrastructure |
| Web search | DuckDuckGo via ddgs | free, no API key, good results |
| Code sandbox | Docker | industry standard isolation, network disabled, memory and CPU capped |
| Video decoding | OpenCV (cv2) | frame sampling from video files before VLM analysis |

## Implementation scope

### Built
- Input Router (keyword pattern matching - 50+ patterns across code/media/document/general)
- Agent node (LangGraph StateGraph, ReAct loop)
- Code Generation node (specialized coder LLM, strips markdown fences from output)
- Human-in-the-loop (LangGraph interrupt(), approval + rejection with feedback loop)
- Memory Manager (SqliteSaver, persistent SQLite, session resume via .last_session)
- System prompt (injected at runtime with today's date)
- Output Parser (retry loop, up to 3 retries, empty-check validation)
- Web search tool (@tool decorator, DuckDuckGo, top 3 results)
- Code execution tool (@tool decorator, Docker sandbox, --network none, 30s timeout)
- Image analysis tool (@tool decorator, base64 encoding, VLM call)
- Video analysis tool (@tool decorator, OpenCV frame sampling, batched VLM call)
- Document analysis tool (@tool decorator, pdfplumber/python-docx/openpyxl/pandas, 6000-char truncation, LLM analysis)
- Streaming responses (token-by-token from agent_node)
- Message window trimming (last 20 messages per LLM call)

### Future improvements
- Gradio or web UI (currently terminal only)
- Docker image with pre-installed packages (numpy, pandas, etc.) for richer code execution
- Context window summarization for long conversations (currently sliding window)
- Per-tool output validation (currently only empty-check)
- LLM-based input routing to replace keyword pattern matching
