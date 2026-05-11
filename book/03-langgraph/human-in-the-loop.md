# Human-in-the-Loop

---
<- [Memory](memory.md) | [Home](../README.md) | [04 Building the Agent](../04-building-the-agent/README.md) ->

---

## What it is

**Human-in-the-loop** means the graph pauses execution and waits for a human to provide input before continuing. In this agent, it's used as a safety gate before executing any code - the user sees the generated code and must approve it before it runs.

LangGraph implements this via `interrupt()` and `Command(resume=...)`.

## Why it matters

Without human-in-the-loop, the agent would automatically execute whatever code it or the user writes - inside a Docker sandbox, but still. The approval step gives the user visibility and control. They can read the code, reject it with feedback, and have the agent rewrite it.

This is also a pattern for production agents that need human oversight on high-stakes actions (sending emails, making purchases, deleting data).

## How it works

Inside `human_approval_node`, the graph interrupts itself:

```python
# src/nodes.py
def human_approval_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print(f"\n{'='*50}")
    print(f"Agent wants to call: '{tool_name}'")
    print(f"{'='*50}")

    if tool_name == "execute_code":
        print("Code to execute:")
        print("-" * 30)
        print(tool_args.get("code", ""))
        print("-" * 30)

    human_response = interrupt("\nApprove? (yes/no + optional feedback): ")

    approved = human_response.lower().startswith("yes") or any(
        phrase in human_response.lower() for phrase in APPROVAL_PHRASES
    )

    if approved:
        return {"human_feedback": human_response, "code_generated": False}
    else:
        return {
            "human_feedback": human_response,
            "code_generated": False,
            "messages": [
                HumanMessage(
                    content=f"The code was rejected. User feedback: '{human_response}'. "
                    f"Please rewrite the code addressing this feedback, then call execute_code again."
                )
            ],
        }
```

When `interrupt()` is called, LangGraph:
1. Saves the current state to SQLite
2. Pauses execution
3. Returns control to the caller (main.py) with `state.next` populated

## Resuming with Command

In `main.py`, the caller checks if the graph is paused and collects the human response:

```python
# src/main.py
state = app.get_state(config)
while state.next:
    human_input = input("\nYour decision: ")
    for chunk, metadata in app.stream(
        Command(resume=human_input),
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "agent_node":
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
    state = app.get_state(config)
```

`Command(resume=human_input)` resumes the graph from the exact checkpoint where it paused. The `interrupt()` call returns the value passed to `Command(resume=...)` - in this case, the human's text input.

## The rejection flow

When the user rejects code, the node injects a HumanMessage with the rejection feedback into the state. The graph then routes back to `agent_node` (via `should_execute_tool` returning `"agent_node"`), which sees this HumanMessage, re-reads the feedback, and calls `code_generation_node` again to rewrite the code.

## The double approval bug

There's a known issue: when code is rejected and rewritten, the approval prompt sometimes appears twice. This happens because LangGraph's `interrupt()` replays the node when resuming from a checkpoint. The first "appearance" is the original interrupt; the second is the replay.

This is a known LangGraph behavior with interrupt() and checkpoint replay. The fix requires careful state management to detect whether the interrupt has already been handled, which adds complexity. I left it as a known limitation for v2 rather than over-engineering the solution.

## Gotchas and lessons learned

- **state.next is the signal.** `app.get_state(config).next` is non-empty when the graph is paused at an interrupt. The `while state.next` loop in main.py handles multiple consecutive interrupts if they ever occur.
- **interrupt() value is the resume value.** The `human_response = interrupt("prompt")` pattern feels like `input("prompt")` but it's not. The string argument is never shown to the user - it's metadata about the interrupt. The actual user prompt is printed before calling interrupt(). The return value of interrupt() is whatever was passed to `Command(resume=...)`.
- **The rejection HumanMessage is injected.** When code is rejected, a new HumanMessage is added to the state explaining the rejection and asking for a rewrite. This is how the agent knows what to fix - it reads the rejection as a new user instruction.
- **Approval detection is fuzzy.** `APPROVAL_PHRASES = ["yes", "i like", "looks good", "approved", "ok", "good", "run it", "execute"]`. The user just needs to start with "yes" or include one of these phrases. This is intentional - strict "yes/no" parsing would frustrate users who type "yes, this looks correct."

---

<- [Memory](memory.md) | [Home](../README.md) | [04 Building the Agent](../04-building-the-agent/README.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](README.md) · [State](state.md) · [Nodes & Edges](nodes-edges.md) · [Conditional Edges](conditional-edges.md) · [Memory](memory.md) · [Human-in-the-Loop](human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
