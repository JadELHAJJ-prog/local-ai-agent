# Input Router

---
<- [V1 Basic Agent](v1-basic-agent.md) | [Home](../README.md) | [Tools & ReAct Loop](tools-react-loop.md) ->

---

## What it is

The input router is the first node the graph runs on every message. It classifies the user's input and sets `input_type` in the state. The routing function then uses that to decide which node to invoke next - without ever calling the LLM.

## Why we added it

Two reasons:

1. **Efficiency.** If the user asks "write a Python function to sort a list," the general LLM doesn't need to reason about it - we know immediately it's a code request. Skip the LLM, go straight to code generation.

2. **Reliability.** The main LLM is a generalist. For code, we have a specialized coder LLM that produces better output. For documents, we want to ensure `analyze_document` is always called, not rely on the LLM to decide.

## How it works

```python
# src/nodes.py
def input_router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content.lower()

    if "[file provided at path:" in content or "[image provided at path:" in content:
        for ext, doc_type in DOCUMENT_EXTENSIONS.items():
            if ext in content:
                return {"input_type": f"document_{doc_type}"}
        return {"input_type": "media"}

    if any(pattern in content for pattern in CODE_PATTERNS):
        return {"input_type": "code"}

    return {"input_type": "general"}
```

The router checks in order:
1. Does the message contain a file path? -> document or media
2. Does the message contain a code keyword? -> code
3. Otherwise -> general

## CODE_PATTERNS - 50+ keywords

```python
# src/config.py
CODE_PATTERNS = [
    "write code", "write a code", "write me a code", "write for me",
    "write a script", "write a program", "write a function", "write a class",
    "create code", "create a script", "create a function",
    "make a code", "make a script", "make a program",
    "give me code", "give me a script",
    "generate code", "generate a function",
    "build a", "build me a", "implement", "implement a",
    "run", "run this", "execute", "execute this",
    "test this code", "python code", "python script",
    ...
]
```

This is brute-force keyword matching. It's not elegant but it's fast, predictable, and debuggable. The alternative (using the LLM to classify) adds latency and a failure point. Pattern matching runs in microseconds and never hallucinates.

## DOCUMENT_EXTENSIONS

```python
# src/config.py
DOCUMENT_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".csv": "csv",
}
```

## The label problem and how it was fixed

In `main.py`, when the user provides a file path, the path is extracted and labeled before being added to the message:

```python
# src/main.py
if image_path:
    ext = os.path.splitext(image_path)[1].lower()
    label = "file" if ext in DOCUMENT_EXTENSIONS else "image"
    message = HumanMessage(
        content=f"{text} [{label} provided at path: {image_path}]"
    )
```

Originally, all file types used `"[image provided at path: ...]"` - even PDFs. This confused the LLM: it saw "image" but the extension was `.pdf`. After the fix, documents use `"[file provided at path: ...]"` and the LLM correctly calls `analyze_document`.

## The crash - routing directly to ToolNode

The original `should_route` routing function was:

```python
# BUG - original version
def should_route(state: AgentState) -> str:
    input_type = state.get("input_type", "general")
    if input_type == "code":
        return "code_generation_node"
    if input_type == "media":
        return "tool_node"           # <- BUG
    if input_type.startswith("document_"):
        return "tool_node"           # <- BUG
    return "agent_node"
```

LangGraph's `ToolNode` searches the message history for the most recent AIMessage with `tool_calls` and executes those calls. If no such AIMessage exists, it raises:

```
ValueError: No AIMessage found in input
```

When a user starts a new session and the very first message is a document request, the message history has only a HumanMessage. There's no AIMessage with tool_calls. ToolNode crashes.

The fix: all non-code paths go through `agent_node` first. The agent sees the document path, calls `analyze_document`, produces an AIMessage with tool_calls. Then ToolNode can run.

```python
# FIXED
def should_route(state: AgentState) -> str:
    input_type = state.get("input_type", "general")
    if input_type == "code":
        return "code_generation_node"
    return "agent_node"
```

## Why the bug was invisible in some cases

When a user continued an existing session and then typed a new message (even unrelated), the document path from the previous session was already in the SQLite checkpoint. When the agent_node ran, it saw the old document path in the history and called `analyze_document` - which produced an AIMessage with tool_calls. So ToolNode had something to work with.

New sessions had no history -> no AIMessage -> crash. Existing sessions had history -> worked. This asymmetry made the bug look intermittent when it was actually deterministic.

## Gotchas and lessons learned

- **Pattern matching has false positives.** The word "run" triggers the code router. "I ran into an error" also contains "ran". I handle this by checking at the router level - if the LLM gets a misclassified message, it falls back gracefully since the agent prompt handles general questions too.
- **The router only reads the last message.** It doesn't look at conversation history. This means if the user previously shared a document path and then asks a follow-up question, the router won't detect the document extension in the new message. The agent handles this correctly via the message history though.

---

<- [V1 Basic Agent](v1-basic-agent.md) | [Home](../README.md) | [Tools & ReAct Loop](tools-react-loop.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](README.md) · [V1 Basic Agent](v1-basic-agent.md) · [Input Router](input-router.md) · [Tools & ReAct Loop](tools-react-loop.md) · [Code Generation Node](code-generation-node.md) · [Docker Sandbox](docker-sandbox.md)
