# Prompt Templates

---
<- [Messages](messages.md) | [Home](../README.md) | [Tools](tools.md) ->

---

## What it is

A `ChatPromptTemplate` is a reusable structure for building the list of messages you send to the LLM. It separates the static parts (your system instructions) from the dynamic parts (the conversation history and any variables you inject at runtime).

## Why it matters

Without a template, you'd manually construct the message list every time you call the model. With a template, you define the structure once and just fill in the variables.

More importantly, the template lets you inject runtime values like today's date into the system prompt - something that's impossible if you hardcode the prompt string.

## How it works

In `nodes.py`, the prompt is defined once at module level:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Bubbles, a helpful AI assistant. Today is {date}.

TOOLS - only use when explicitly needed:
- search_web: ONLY if user asks for news, current events, or real-time data.
- execute_code: ONLY if user says "run", "execute", or "test this code".
- analyze_image: use when user provides an image path ending in .jpg .jpeg .png .gif .webp.
- analyze_video: use when user provides a video path ending in .mp4 .avi .mov .mkv.
- analyze_document: use when user provides a file path ending in .pdf .docx .xlsx .xls .csv.
...""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
```

And invoked inside `agent_node` with actual values:

```python
def agent_node(state: AgentState) -> dict:
    chain = prompt | llm_with_tools
    response = chain.invoke(
        {
            "date": date.today().isoformat(),       # fills {date} in the system prompt
            "messages": trim_messages_window(state["messages"]),  # fills MessagesPlaceholder
        }
    )
    return {"messages": [response]}
```

## ChatPromptTemplate.from_messages

`from_messages` takes a list of tuples. Each tuple is `(role, content)`:

- `("system", "...")` -> SystemMessage
- `("human", "...")` -> HumanMessage
- `("ai", "...")` -> AIMessage

Or you can pass a `MessagesPlaceholder` directly as an object (not a tuple).

## MessagesPlaceholder

`MessagesPlaceholder(variable_name="messages")` is a slot that says "insert the full list of messages here." When you invoke the chain with `{"messages": [...]}`, the placeholder expands into the actual message list.

This is how the full conversation history gets passed to the model on every call. The template combines:
1. The static system instructions (with dynamic `{date}` variable)
2. The full conversation history (via `MessagesPlaceholder`)

And hands that combined list to the LLM.

## Gotchas and lessons learned

- **The system prompt is where tool instructions live.** The LLM decides which tool to call based on the system prompt instructions. If the instructions are vague or inconsistent, the model picks the wrong tool. I spent time refining the tool descriptions until tool selection was reliable.
- **{date} injection prevents stale context.** Without injecting today's date, the model might say "as of my knowledge cutoff in 2024..." even for current-year questions. A simple `{date}` injection makes the model aware of when it's running.
- **MessagesPlaceholder is optional.** If your agent doesn't need conversation history (a one-shot tool), you can skip it. For a conversational agent like this one, it's essential.
- **System prompt length affects VRAM.** The system prompt counts toward the context window on every call. Keep it concise. Every token in the system prompt is a token not available for conversation history.

---

<- [Messages](messages.md) | [Home](../README.md) | [Tools](tools.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](README.md) · [Messages](messages.md) · [Prompt Templates](prompt-templates.md) · [Tools](tools.md) · [Pipe Operator](pipe-operator.md) · [Runnables](runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
