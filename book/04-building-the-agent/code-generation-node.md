# Code Generation Node

---
<- [Tools & ReAct Loop](tools-react-loop.md) | [Home](../README.md) | [Docker Sandbox](docker-sandbox.md) ->

---

## What it is

The code generation node is a specialized step that routes code requests through `qwen2.5-coder:7b`. Before the generated code reaches the human approval gate, it is tested inside the Docker sandbox.

If the code fails to run, the actual execution error is sent back to the coder LLM together with the original request and the failed code. The coder then attempts to fix the code and test it again, up to a maximum of three attempts.

Only after the code runs successfully, or the retry limit is reached, is it passed to the human approval node.

## Why a separate node

Three reasons:

1. **Specialization.** Qwen2.5-Coder 7B is specifically fine-tuned for code. It produces cleaner, more correct Python than the general Qwen2.5 7B on code tasks.

2. **Separation of concerns.** The general LLM reasons about what the user wants. The coder LLM writes the actual code. Each model does what it's best at.

3. **Code quality before review.** Before the user sees the code for approval, the coder LLM can test it in the Docker sandbox and automatically repair runtime errors. This reduces the chance that the human is asked to approve code that does not even run.

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

        code = _debug_code_in_sandbox(
            original_request=user_request,
            code=code,
        )

        new_message = AIMessage(
            content="",
            tool_calls=[{
                "name": "execute_code",
                "args": {"code": code},
                "id": f"call-{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }]
        )
        return {"messages": [new_message]}
```
The coder LLM first generates the code and `_strip_markdown` removes any accidental Markdown fences. The code is then passed to `_debug_code_in_sandbox`, which tests it and attempts to repair runtime errors before continuing.

Once the self-debug loop succeeds or reaches its retry limit, an `AIMessage` containing an `execute_code` tool call is created. This allows the graph to continue to the human approval stage.

**Path 2: from agent_node** - user asked to run code, and the general LLM wrote rough code in its tool_call

```python
    # Path 2: coming from agent_node - improve existing code in the tool_call
    tool_call = last_message.tool_calls[0]
    rough_code = tool_call["args"]["code"]
    user_request = next(
        (
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        ),
        rough_code,
    )
    code_prompt = f"""You are an expert Python developer.
Improve and optimize this code. Return ONLY raw Python code, no markdown, no explanation:

{rough_code}"""
    response = coder_llm.invoke([HumanMessage(content=code_prompt)])
    improved_code = _strip_markdown(response.content)

    improved_code = _debug_code_in_sandbox(
        original_request=user_request,
        code=improved_code,
    )

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
    return {"messages": [updated_message]}
```

The general LLM has already proposed rough code, so the coder LLM first improves it. The improved version is then tested by the same self-debug loop before being passed to human approval.

The existing tool-call ID and message ID are preserved so LangGraph updates the existing message instead of creating a duplicate.

## Self-debug loop

Before generated code is shown to the human for approval, it is tested inside the Docker sandbox.

The loop is limited to three execution attempts:

```python
MAX_CODE_ATTEMPTS = 3
```

The `_debug_code_in_sandbox` helper runs the generated code using `run_code_in_sandbox`. If the code executes successfully, the loop stops immediately.

If execution fails, the actual sandbox error is sent back to `coder_llm` together with the original user request and the failed code. The coder generates a corrected version, which is then tested again.

```python
def _debug_code_in_sandbox(original_request: str, code: str) -> str:
    for attempt in range(MAX_CODE_ATTEMPTS):
        result = run_code_in_sandbox(code)

        if not result.startswith("Error:"):
            return code

        if attempt == MAX_CODE_ATTEMPTS - 1:
            return code

        repair_prompt = _build_code_repair_prompt(
            original_request=original_request,
            code=code,
            error=result,
        )

        response = coder_llm.invoke(
            [HumanMessage(content=repair_prompt)]
        )

        code = _strip_markdown(response.content)

    return code
```

The flow is:

```text
Generate code
    ↓
Test in Docker sandbox
    ↓
Success? → continue to human approval
    ↓ No
Send code + error back to coder_llm
    ↓
Generate corrected code
    ↓
Test again
```

The sandbox execution in this loop is only a preflight check to verify that the code runs. It does not replace human approval. After the loop succeeds or uses all three attempts, the final code is still sent to `human_approval_node` before the normal `execute_code` tool is allowed to run.

The code-generation retry loop is separate from the existing `retry_count` state field, which is used by the output-parser retry logic.

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

## Gotchas and lessons learned

- **The fake AIMessage trick.** When code generation comes from the input router (Path 1), there's no existing AIMessage with tool_calls - just a HumanMessage. We need to create a valid AIMessage with an `execute_code` tool_call to pass to ToolNode later. The fake AIMessage is added to the message history via `add_messages`. This looks like the agent called `execute_code`, which makes the conversation history coherent.
- **Message ID must match in Path 2.** When updating an existing AIMessage (Path 2), the `id=last_message.id` is critical. LangGraph's `add_messages` reducer uses the message ID to decide whether to append or update. Reusing the same ID updates the existing message in place rather than duplicating it.
- **The coder LLM has a longer num_predict.** I set `num_predict=2048` for the coder LLM vs 1024 for the general LLM because code outputs are longer. A function with documentation and type hints can easily hit 300-500 tokens.
- **Sandbox testing does not replace human approval.** The self-debug loop executes generated code automatically inside the isolated Docker sandbox only as a preflight check to determine whether the code runs successfully. After the loop finishes, the final code still goes through `human_approval_node` before the normal `execute_code` tool execution.
- **Code-generation retries are separate from output-parser retries.** The self-debug loop uses a local `attempt` counter and `MAX_CODE_ATTEMPTS`. It does not use the `retry_count` field in `AgentState`, which belongs to the unrelated output-parser retry mechanism.
---

<- [Tools & ReAct Loop](tools-react-loop.md) | [Home](../README.md) | [Docker Sandbox](docker-sandbox.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](README.md) · [V1 Basic Agent](v1-basic-agent.md) · [Input Router](input-router.md) · [Tools & ReAct Loop](tools-react-loop.md) · [Code Generation Node](code-generation-node.md) · [Docker Sandbox](docker-sandbox.md)
