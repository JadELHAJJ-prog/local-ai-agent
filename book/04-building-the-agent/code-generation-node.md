# Code Generation Node

---
<- [Tools & ReAct Loop](tools-react-loop.md) | [Home](../README.md) | [Docker Sandbox](docker-sandbox.md) ->

---

## What it is

The code generation node is a specialized step that routes code requests through `qwen2.5-coder:7b` before passing the code to the human approval gate. It separates the "what code should I write?" reasoning (done by the general LLM or the user) from "write me clean, correct Python" (done by the specialized coder LLM).

## Why a separate node

Three reasons:

1. **Specialization.** Qwen2.5-Coder 7B is specifically fine-tuned for code. It produces cleaner, more correct Python than the general Qwen2.5 7B on code tasks.

2. **Separation of concerns.** The general LLM reasons about what the user wants. The coder LLM writes the actual code. Each model does what it's best at.

3. **Code quality before review.** When the user sees code for approval, it should be the best version possible - not rough code from a generalist model.

## Two paths into the node

The code generation node can be reached from two different places:

**Path 1: from input_router_node** - user explicitly asked for code ("write a Python script to...")

```python
# src/nodes.py
def code_generation_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]

    # Path 1: no tool_calls yet - generate from user request directly
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        user_request = last_message.content
        code_prompt = f"""You are an expert Python developer.
Write Python code for this task. Return ONLY raw Python code, no markdown, no explanation:

{user_request}"""
        response = coder_llm.invoke([HumanMessage(content=code_prompt)])
        code = _strip_markdown(response.content)

        new_message = AIMessage(
            content="",
            tool_calls=[{
                "name": "execute_code",
                "args": {"code": code},
                "id": f"call-{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }]
        )
        return {"messages": [new_message], "code_generated": True}
```

The coder LLM generates the code. A fake AIMessage is created with an `execute_code` tool_call. This is needed because ToolNode needs to see an AIMessage with tool_calls to know what to execute.

**Path 2: from agent_node** - user asked to run code, and the general LLM wrote rough code in its tool_call

```python
    # Path 2: coming from agent_node - improve existing code in the tool_call
    tool_call = last_message.tool_calls[0]
    rough_code = tool_call["args"]["code"]
    code_prompt = f"""You are an expert Python developer.
Improve and optimize this code. Return ONLY raw Python code, no markdown, no explanation:

{rough_code}"""
    response = coder_llm.invoke([HumanMessage(content=code_prompt)])
    improved_code = _strip_markdown(response.content)

    updated_message = AIMessage(
        id=last_message.id,
        content=last_message.content,
        tool_calls=[{
            "name": "execute_code",
            "args": {"code": improved_code},
            "id": tool_call["id"],
            "type": "tool_call",
        }]
    )
    return {"messages": [updated_message], "code_generated": True}
```

The general LLM already produced rough code. The coder LLM improves it. The AIMessage is updated with the improved code but keeps the original message ID - this is important for LangGraph's message deduplication.

## _strip_markdown - the necessary hack

LLMs love wrapping code in markdown fences even when told not to:

```
```python
def my_function():
    return 42
```
```

If you pass this raw to the Docker sandbox, Python will try to execute the backticks and fail. `_strip_markdown` fixes this:

```python
def _strip_markdown(text: str) -> str:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text
```

I tried telling the coder LLM "Return ONLY raw Python code, no markdown, no explanation" in the prompt. It works most of the time. But "most of the time" is not good enough when the failure mode is a Python syntax error. The strip function is the defensive fallback.

## The code_generated flag

```python
return {"messages": [new_message], "code_generated": True}
```

After the coder LLM runs, `code_generated` is set to `True`. The routing function `should_use_tool` checks this:

```python
if tool_name == "execute_code":
    if not state.get("code_generated", False):
        return "code_generation_node"  # go improve the code first
    return "human_approval_node"       # code already good, go approve it
```

Without this flag, the agent would loop between the general LLM and code generation node indefinitely. The flag says "we've already done the code improvement step, skip it."

After code is approved and executed (or rejected), `code_generated` is reset to `False` in `human_approval_node` so the next code request goes through the full improvement cycle.

## Gotchas and lessons learned

- **The fake AIMessage trick.** When code generation comes from the input router (Path 1), there's no existing AIMessage with tool_calls - just a HumanMessage. We need to create a valid AIMessage with an `execute_code` tool_call to pass to ToolNode later. The fake AIMessage is added to the message history via `add_messages`. This looks like the agent called `execute_code`, which makes the conversation history coherent.
- **Message ID must match in Path 2.** When updating an existing AIMessage (Path 2), the `id=last_message.id` is critical. LangGraph's `add_messages` reducer uses the message ID to decide whether to append or update. Reusing the same ID updates the existing message in place rather than duplicating it.
- **The coder LLM has a longer num_predict.** I set `num_predict=2048` for the coder LLM vs 1024 for the general LLM because code outputs are longer. A function with documentation and type hints can easily hit 300-500 tokens.

---

<- [Tools & ReAct Loop](tools-react-loop.md) | [Home](../README.md) | [Docker Sandbox](docker-sandbox.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](README.md) · [V1 Basic Agent](v1-basic-agent.md) · [Input Router](input-router.md) · [Tools & ReAct Loop](tools-react-loop.md) · [Code Generation Node](code-generation-node.md) · [Docker Sandbox](docker-sandbox.md)
