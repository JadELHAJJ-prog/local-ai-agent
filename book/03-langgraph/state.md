# State

---
<- [03 LangGraph](README.md) | [Home](../README.md) | [Nodes & Edges](nodes-edges.md) ->

---

## What it is

LangGraph's **state** is a typed dictionary that flows through every node in the graph. Every node receives the current state, optionally modifies it, and returns the updated fields. The state is the single source of truth for everything happening in the agent.

## Why it matters

Without shared state, nodes can't communicate. A node that decides the agent needs to call a tool has no way to tell the next node which tool to call - unless that information is in the state. Without state, you'd have to pass data through global variables or complex argument threading.

The state also enables checkpointing. LangGraph saves the entire state to SQLite after every node. If the process crashes, the state is recoverable.

## How it works

The state schema is defined in `state.py`:

```python
# src/state.py
from langgraph.graph.message import add_messages
from typing import Annotated, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation history
    thread_id: str                            # unique session identifier
    retry_count: int                          # how many times we've retried
    is_valid: bool                            # whether last response was valid
    human_feedback: Optional[str]             # yes/no + comments from user
    code_generated: bool                      # whether coder LLM has already run
    input_type: str                           # "code" / "media" / "document_*" / "general"
```

Each field serves a specific purpose:

| Field | Set by | Read by |
|-------|--------|---------|
| `messages` | every node | agent_node, tool_node |
| `thread_id` | main.py config | SqliteSaver |
| `retry_count` | output_parser_node | should_retry |
| `is_valid` | output_parser_node | should_retry |
| `human_feedback` | human_approval_node | should_execute_tool |
| `code_generated` | code_generation_node | should_use_tool |
| `input_type` | input_router_node | should_route |

## The add_messages reducer

```python
messages: Annotated[list, add_messages]
```

The `Annotated[list, add_messages]` syntax means: "this is a list, and when a node returns new values for this field, use the `add_messages` reducer to merge them."

Without the reducer, LangGraph would replace the messages list with whatever the node returns. So if `agent_node` returns `{"messages": [AIMessage(...)]}`, the entire conversation history would be wiped and replaced with just that one message.

With `add_messages`, the reducer appends new messages to the existing list instead of replacing it. The conversation history grows correctly.

## How nodes update state

Each node returns a dict with only the fields it wants to update:

```python
# input_router_node - only updates input_type
def input_router_node(state: AgentState) -> dict:
    ...
    return {"input_type": "document_pdf"}

# agent_node - only updates messages
def agent_node(state: AgentState) -> dict:
    ...
    return {"messages": [response]}

# human_approval_node - updates multiple fields
def human_approval_node(state: AgentState) -> dict:
    ...
    return {"human_feedback": human_response, "code_generated": False}
```

LangGraph merges these partial updates into the full state. Fields not returned by a node are unchanged.

## Gotchas and lessons learned

- **All state fields must have defaults or be optional.** If `retry_count` starts as `None` and a node tries to add 1 to it, you get a TypeError. Set sensible defaults or use `Optional` with explicit None handling.
- **code_generated must be reset.** After code is approved and executed, `code_generated` is reset to `False`. This matters because the routing logic checks this flag to decide whether to generate code or go straight to approval. Without the reset, the second code request in a session skips code generation.
- **input_type is not cleared between turns.** After the first message, `input_type` stays set to whatever the last router decided. This is fine because the router re-runs on every new message and overwrites it.

---

<- [03 LangGraph](README.md) | [Home](../README.md) | [Nodes & Edges](nodes-edges.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](README.md) · [State](state.md) · [Nodes & Edges](nodes-edges.md) · [Conditional Edges](conditional-edges.md) · [Memory](memory.md) · [Human-in-the-Loop](human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
