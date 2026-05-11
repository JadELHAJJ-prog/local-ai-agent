# Tools

---
<- [Prompt Templates](prompt-templates.md) | [Home](../README.md) | [Pipe Operator](pipe-operator.md) ->

---

## What it is

In LangChain, a **tool** is a Python function that the LLM can choose to call. The model doesn't execute code itself - it produces a structured description of which tool to call and with which arguments. Your framework (LangGraph's ToolNode) executes the actual function.

## Why it matters

Tools are how an LLM agent interacts with the real world. Without tools, the model can only produce text. With tools, it can search the web, run code, analyze files, and more.

The key insight: **the docstring of a tool is its entire interface with the LLM.** The model never sees the function body. It sees the function name and the docstring, and uses that to decide when to call the tool and what arguments to pass.

## How it works

The `@tool` decorator wraps a regular Python function and registers it as a callable tool:

```python
# src/tools.py
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information, recent events,
    or anything that requires up-to-date knowledge beyond
    your training data. Input should be a search query string."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        if not results:
            return "No results found."
        return "\n".join(
            [f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}" for r in results]
        )
```

The docstring matters. "Search the web for current information, recent events, or anything that requires up-to-date knowledge" tells the LLM exactly when to use this tool.

## bind_tools

Once you have your tools, you attach them to the LLM:

```python
# src/nodes.py
from tools import tools

llm_with_tools = llm.bind_tools(tools)
```

`bind_tools` tells the LLM about the available tools by injecting their schemas into the prompt. The LLM then knows it can produce structured tool calls in its output.

## All 5 tools in this project

```python
# src/tools.py
tools = [search_web, execute_code, analyze_image, analyze_video, analyze_document]
```

| Tool | Trigger | What it does |
|------|---------|--------------|
| `search_web` | user asks for news / current events | DuckDuckGo search, returns top 3 results |
| `execute_code` | user asks to run / execute code | Runs Python in a Docker sandbox |
| `analyze_image` | user provides .jpg .png .gif .webp path | Base64 encodes and sends to VLM |
| `analyze_video` | user provides .mp4 .avi .mov .mkv path | Samples frames with OpenCV, sends to VLM |
| `analyze_document` | user provides .pdf .docx .xlsx .csv path | Extracts text, sends to LLM for analysis |

## What happens when a tool is called

1. The model produces an AIMessage with `tool_calls`:
```python
AIMessage(
    content="",
    tool_calls=[{
        "name": "analyze_document",
        "args": {"file_path": "/path/to/resume.pdf", "question": "Summarize this document"},
        "id": "call-abc123",
        "type": "tool_call"
    }]
)
```

2. LangGraph's ToolNode sees this AIMessage, executes `analyze_document`, and produces a ToolMessage with the result.

3. The agent_node is called again with the ToolMessage in its history, generates a final response.

## Gotchas and lessons learned

- **Docstrings determine tool selection reliability.** I originally had vague docstrings like "execute Python code." The model would sometimes call execute_code for code it was just supposed to explain. I added "Use this when the user asks to **run** code" and reliability improved immediately.
- **Tool argument types matter.** If your function takes `file_path: str`, the LLM passes a string. If you have complex nested types, the LLM may fail to format them correctly. Keep tool arguments simple: strings, ints, floats.
- **Return values are text.** Whatever your tool returns becomes the content of a ToolMessage, which the model reads as text. Return clear, parseable strings. Returning a Python dict gets str()-ified into something messy.
- **The LLM can call non-existent tools.** If the model hallucinates a tool name that doesn't exist in your bind_tools list, ToolNode will raise a KeyError. This is rare with good models but happens with smaller or less capable ones.

---

<- [Prompt Templates](prompt-templates.md) | [Home](../README.md) | [Pipe Operator](pipe-operator.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](README.md) · [Messages](messages.md) · [Prompt Templates](prompt-templates.md) · [Tools](tools.md) · [Pipe Operator](pipe-operator.md) · [Runnables](runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
