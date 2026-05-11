# V1 - The Basic Agent

---
<- [04 Building the Agent](README.md) | [Home](../README.md) | [Input Router](input-router.md) ->

---

## What it is

The first working version of the agent was simple: one LangGraph node, one LLM with tools bound, one streaming loop in main.py. No router, no code generation node, no human approval.

## Why start here

It's tempting to design the entire system upfront and build it all at once. I've done this in robotics projects and it never works - you end up debugging 500 lines of untested code with no idea where the problem is.

Starting with the minimal working thing and adding complexity incrementally means you always know what broke and when.

## The core agent node

```python
# src/nodes.py (simplified v1)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from models import llm
from tools import tools

llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Bubbles, a helpful AI assistant. Today is {date}."),
    MessagesPlaceholder(variable_name="messages"),
])

def agent_node(state: AgentState) -> dict:
    chain = prompt | llm_with_tools
    response = chain.invoke({
        "date": date.today().isoformat(),
        "messages": trim_messages_window(state["messages"]),
    })
    return {"messages": [response]}
```

One function. Takes state, calls the LLM, returns the response to be appended to messages.

## The streaming loop

```python
# src/main.py
for chunk, metadata in app.stream(
    {"messages": [message]},
    config=config,
    stream_mode="messages",
):
    if metadata.get("langgraph_node") == "agent_node":
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
```

`app.stream()` yields chunks from every node in the graph. The filter `metadata.get("langgraph_node") == "agent_node"` ensures only the agent's text responses are printed. Without the filter, you'd see tool results, state updates, and internal messages all dumped to the terminal.

`stream_mode="messages"` tells LangGraph to yield individual message chunks (tokens) rather than full state snapshots. This is what makes the response appear to stream word by word instead of all at once.

`print(chunk.content, end="", flush=True)` - `end=""` prevents newlines between tokens, `flush=True` ensures each token appears immediately.

## The first working session

The first time I ran the agent and it responded coherently to a question, then called `search_web` when I asked about current events, then returned the results in a readable format - that was the moment the whole project clicked.

Up to that point I was mostly configuring infrastructure. This was the first time it felt like a real agent.

## What this version lacked

- No input routing (every message hit the LLM)
- No human approval for code execution
- No specialized code generation
- No document analysis

The v1 was the proof of concept. Everything else was added on top.

---

<- [04 Building the Agent](README.md) | [Home](../README.md) | [Input Router](input-router.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](README.md) · [V1 Basic Agent](v1-basic-agent.md) · [Input Router](input-router.md) · [Tools & ReAct Loop](tools-react-loop.md) · [Code Generation Node](code-generation-node.md) · [Docker Sandbox](docker-sandbox.md)
