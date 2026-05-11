# Messages

---
<- [02 LangChain](README.md) | [Home](../README.md) | [Prompt Templates](prompt-templates.md) ->

---

## What it is

In LangChain, every piece of communication between the user, the model, and tools is a typed message object. There are four you'll use regularly:

- **HumanMessage**: a message from the user
- **AIMessage**: a response from the model (may include tool_calls)
- **ToolMessage**: the result of executing a tool
- **SystemMessage**: instructions baked into the prompt (usually not in the message list directly, but in the ChatPromptTemplate)

## Why it matters

The LLM sees a flat list of messages - it doesn't have a database or memory of its own. Everything the model knows about the current conversation is in this message list. When you add a new user message, you append to the list. When the model responds, you append the AIMessage. When a tool runs, you append the ToolMessage. This is the entire "memory" mechanism at the message level.

## How it works

```python
from langchain_core.messages import HumanMessage, AIMessage

# In main.py - wrapping user input
message = HumanMessage(content="hi, analyse this document [file provided at path: /home/jad-elhajj/Desktop/resume.pdf]")

# The model returns an AIMessage, sometimes with tool_calls
# AIMessage with no tool - just a text response
AIMessage(content="Hello! How can I help?")

# AIMessage with a tool call - model wants to run analyze_document
AIMessage(
    content="",
    tool_calls=[{
        "name": "analyze_document",
        "args": {"file_path": "/home/jad-elhajj/Desktop/resume.pdf", "question": "Summarize this document"},
        "id": "call-abc123",
        "type": "tool_call"
    }]
)
```

After the tool runs, LangGraph adds a ToolMessage automatically:

```python
ToolMessage(
    content="The document contains... [extracted text]",
    tool_call_id="call-abc123"
)
```

## The add_messages reducer

In `state.py`, the messages field uses a special annotation:

```python
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

The `add_messages` annotation tells LangGraph: **append new messages, don't replace the list**. Without it, every node that returns `{"messages": [...]}` would overwrite the entire message history. With it, new messages are appended to the existing list.

This is subtle but critical. Without the reducer, each node's response would destroy the conversation history. With it, the history grows correctly with each step.

## Message flow in the agent

Here's how messages flow through one full request:

```
[HumanMessage: "analyse this PDF"]
    ↓ agent_node
[HumanMessage: "analyse this PDF", AIMessage: tool_call(analyze_document)]
    ↓ tool_node
[HumanMessage: ..., AIMessage: tool_call(...), ToolMessage: "Document content: ..."]
    ↓ agent_node (again)
[..., AIMessage: "Based on the document, here is a summary..."]
```

Each arrow represents a node adding its output to the message list via add_messages.

## Gotchas and lessons learned

- **AIMessage.content can be empty.** When the model decides to call a tool, it often returns an AIMessage with empty content and a populated tool_calls list. My output parser initially flagged these as invalid (empty response) before I understood this pattern.
- **ToolMessage must match tool_call_id.** The ToolMessage's `tool_call_id` must match the `id` in the AIMessage's `tool_calls`. LangGraph's ToolNode handles this automatically. If you manually create ToolMessages (e.g., for testing), get this right or the model will be confused.
- **The model sees all message types equally.** The model doesn't distinguish between "this is a tool result" and "this is user input" in some magical way - it just sees formatted text. The message type determines the formatting of the token prefix (e.g., `<|im_start|>tool\n...`), which the model has been trained to understand.

---

<- [02 LangChain](README.md) | [Home](../README.md) | [Prompt Templates](prompt-templates.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](README.md) · [Messages](messages.md) · [Prompt Templates](prompt-templates.md) · [Tools](tools.md) · [Pipe Operator](pipe-operator.md) · [Runnables](runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
