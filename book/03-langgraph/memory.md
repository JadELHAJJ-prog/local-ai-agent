# Memory

---
<- [Conditional Edges](conditional-edges.md) | [Home](../README.md) | [Human-in-the-Loop](human-in-the-loop.md) ->

---

## What it is

In this project, **memory** means persistent conversation history. Every message - user inputs, agent responses, tool calls, tool results - is stored in a local SQLite database and retrieved automatically on every graph invocation.

The component that handles this is `SqliteSaver`.

## Why it matters

Without persistent memory, the agent starts from scratch on every invocation. The user would have to repeat context every time they restart the program. With SQLite memory, conversations survive process restarts, crashes, and OS reboots.

## How it works

Memory is attached at compile time:

```python
# src/main.py
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("memory.db") as memory:
    app = build_graph(memory)
```

And passed to graph.compile:

```python
# src/graph.py
def build_graph(memory: SqliteSaver):
    ...
    return graph.compile(checkpointer=memory)
```

The `checkpointer=memory` argument tells LangGraph to save the full graph state after every node execution. The SQLite file `memory.db` is created in whatever directory you run `python main.py` from.

## thread_id - the conversation key

Each conversation is identified by a `thread_id`. LangGraph uses the thread_id to store and retrieve checkpoints. When you run `app.stream(input, config={"configurable": {"thread_id": "abc"}})`, LangGraph:

1. Looks up `"abc"` in SQLite and loads the last saved state
2. Runs the graph from that state (prepending the new input)
3. Saves the updated state back to SQLite with key `"abc"`

This means two sessions with different thread_ids are completely independent conversations.

## Session management

In `main.py`, the user chooses to start a new session or continue an existing one:

```python
# src/main.py
if choice == "2":
    thread_id = load_last_thread()
    print(f"Continuing session: {thread_id}\n")
else:
    thread_id = str(uuid.uuid4())
    print(f"New session started: {thread_id}")
    print(f"Save this ID to continue later: {thread_id}\n")

save_last_thread(thread_id)
config = {"configurable": {"thread_id": thread_id}}
```

The `.last_session` file stores the most recent thread_id. On startup, option 2 reads it and continues that conversation. This is a simple trick that avoids building a session selection UI - just remember the last one.

## What gets stored

LangGraph stores the entire `AgentState` dict at every checkpoint: messages, retry_count, is_valid, human_feedback, code_generated, input_type - everything. The most important field is `messages`, which is the full conversation history.

The sliding window trim in `agent_node` only affects what the LLM sees per call:

```python
def trim_messages_window(messages: list, max_messages: int = 20) -> list:
    if len(messages) > max_messages:
        return messages[-max_messages:]
    return messages
```

The full history is always preserved in SQLite. If you have a 100-message conversation, SQLite has all 100, but the LLM only sees the last 20 on each call.

## Gotchas and lessons learned

- **memory.db path is relative to where you run the script.** If you `cd src && python main.py`, memory.db is created inside `src/`. If you run it from the repo root, it's there. Be consistent about where you run from.
- **The session bug I hit.** When a new session started and the user asked about a document in the first message, the agent crashed (routing bug). When they continued an existing session and said "hi", it worked - because the document path was already in the SQLite checkpoint and the agent saw it in the message history. This is what made the bug hard to diagnose: behavior differed based on whether the session was new or resumed.
- **Checkpoints stack up.** SQLite doesn't auto-prune old checkpoints. After months of use, the database could get large. A cron job to prune old threads is a future improvement.
- **thread_id from UUID4 is effectively unique.** UUID4 generates 122 bits of randomness. The probability of collision across all sessions ever run is negligible. Don't worry about it.

---

<- [Conditional Edges](conditional-edges.md) | [Home](../README.md) | [Human-in-the-Loop](human-in-the-loop.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](README.md) · [State](state.md) · [Nodes & Edges](nodes-edges.md) · [Conditional Edges](conditional-edges.md) · [Memory](memory.md) · [Human-in-the-Loop](human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
