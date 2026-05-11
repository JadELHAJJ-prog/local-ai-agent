# LLM Basics

---
<- [Quantization](quantization.md) | [Home](../README.md) | [02 LangChain](../02-langchain/README.md) ->

---

## What it is

An LLM (Large Language Model) takes a sequence of tokens as input and predicts the next token. It does this repeatedly to generate a full response. That's the entire mechanism - everything else is built on top of this prediction loop.

Understanding the vocabulary here will make your debugging life much easier.

## Tokens

The model doesn't see characters or words - it sees **tokens**. A token is roughly:

- ~4 characters of English text
- ~0.75 words

A sentence like "What is the weather today?" is about 7 tokens. A paragraph is 50-150 tokens. A full page is 500-750 tokens.

Why does this matter? Because everything in the LLM world is measured in tokens: context windows, pricing, generation limits. When I say my context window is 8192 tokens, that's about 6,000 words - roughly 10 pages of text.

## Context window

The context window is the maximum number of tokens the model can process at once - both input and output combined. Everything the model "knows" about your conversation must fit inside this window.

This is why I added a sliding window in the agent:

```python
# src/nodes.py
def trim_messages_window(messages: list, max_messages: int = 20) -> list:
    if len(messages) > max_messages:
        return messages[-max_messages:]
    return messages
```

The full conversation history lives in SQLite forever. But each time the LLM is called, I only send the last 20 messages. This prevents the context window from filling up on long conversations. The trade-off: the model can't reference things from early in a very long conversation.

The context window is set in `config.py`:

```python
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
```

And passed to Ollama in `models.py`:

```python
llm = ChatOllama(
    model=LLM_MODEL,
    num_ctx=NUM_CTX,
    ...
)

vlm = ChatOllama(
    model=VLM_MODEL,
    num_ctx=32768,  # VLM needs more - multiple image frames
    ...
)
```

The VLM gets a much larger context window because images take thousands of tokens to represent when encoded as base64.

## Temperature

Temperature controls how random or deterministic the model's outputs are.

- **0.0**: deterministic - same input always produces the same output
- **1.0**: highly random - creative and unpredictable
- **0.1**: mostly deterministic with tiny variation - what I use

For an AI agent doing tool calling, you want low temperature. The model needs to produce structured output (JSON-formatted tool calls) consistently. High temperature means it might format the tool call wrong, or hallucinate extra fields, or decide not to use the tool when it should.

```python
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
```

## top_p (nucleus sampling)

After the model computes probabilities for the next token, `top_p` filters the candidates. With `top_p=0.9`, the model only considers the tokens that together make up 90% of the probability mass, and ignores the rest.

The effect: it prevents the model from picking extremely unlikely tokens while still allowing some diversity. It works together with temperature.

```python
TOP_P = float(os.getenv("TOP_P", "0.9"))
```

In practice, `top_p=0.9` with `temperature=0.1` is a conservative but reliable setting for agentic work. I haven't needed to tune this.

## num_predict

The maximum number of tokens the model will generate in a single response.

```python
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "1024"))
```

For the main LLM, 1024 tokens (~750 words) is enough for reasoning and tool calling. For the coder LLM, I set it to 2048 because code outputs are longer:

```python
coder_llm = ChatOllama(
    model=CODER_MODEL,
    num_ctx=NUM_CTX,
    temperature=0.1,
    num_predict=2048,  # longer for code
    top_p=TOP_P,
)
```

If you set `num_predict` too low and the model is generating a long response, it will cut off mid-sentence.

## Gotchas and lessons learned

- **Context window ≠ memory.** The context window is stateless - the model doesn't "remember" anything between API calls. Every call starts fresh. Persistence is your job, which is why we use SQLite.
- **Temperature 0 isn't always better.** At exactly 0, some models get stuck in repetition loops. 0.1 is the safe minimum.
- **Token counts are model-specific.** The same text tokenizes differently across models. A 7B model and a 70B model may use different tokenizers. Don't assume token counts transfer between models.
- **The KV cache eats VRAM.** Every token in the context window takes up KV cache memory. With `num_ctx=8192` on a 7B model, the KV cache can take 1-2 GB of VRAM on top of the model weights. This is why I can't fit all three models simultaneously.

---

<- [Quantization](quantization.md) | [Home](../README.md) | [02 LangChain](../02-langchain/README.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](README.md) · [Quantization](quantization.md) · [LLM Basics](llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
