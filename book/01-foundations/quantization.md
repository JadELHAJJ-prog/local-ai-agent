# Quantization

---
<- [01 Foundations](README.md) | [Home](../README.md) | [LLM Basics](llm-basics.md) ->

---

## What it is

A neural network is a massive collection of floating-point numbers called weights. A 7-billion-parameter model has 7 billion of these numbers. The question is: how many bits do you use to store each one?

By default, weights are stored as 32-bit floats (fp32). That's 4 bytes per weight. 7B × 4 bytes = 28 GB just to load the model into memory - more than most consumer GPUs have.

**Quantization** reduces the number of bits per weight. Instead of 32-bit precision, you use 8-bit, 4-bit, or even 2-bit. You lose some precision, but the model still works surprisingly well at 4-bit.

The rough VRAM formula:

```
VRAM (GB) ≈ (params_billions × bits_per_weight) / 8
```

Examples:
- 7B params × 16-bit = 14 GB (fp16, still too large for 8GB VRAM)
- 7B params × 8-bit = 7 GB (fits, but tight)
- 7B params × 4-bit ≈ 3.5 GB (fits with room, real usage ~4.7 GB with overhead)

The 4-bit sweet spot is why almost all local AI work on consumer hardware uses 4-bit quantized models.

## GGUF vs GPTQ

These are two different quantization file formats with different trade-offs.

**GGUF** (what Ollama uses):
- Designed for CPU+GPU hybrid inference
- The model can split across CPU RAM and GPU VRAM
- If it doesn't all fit in VRAM, it offloads layers to CPU
- Slower when offloading, but flexible
- File format: `.gguf` (previously `.ggml`)

**GPTQ**:
- GPU-only, requires the full model in VRAM
- Faster inference when it fits
- Less flexible than GGUF for consumer hardware

For this project I used Ollama, which uses GGUF. The advantage: if one model doesn't fit fully in VRAM, Ollama can still run it (just slower). This matters because my setup has two 7B models and an 8GB GPU.

## Q4_K_M - what the letters mean

When you run `ollama pull qwen2.5:7b`, Ollama downloads a quantized version. The naming convention like `Q4_K_M` means:

- `Q4`: 4-bit quantization
- `K`: K-quantization (a smarter algorithm that quantizes different layers at different precision)
- `M`: medium size variant (K-quants come in S/M/L)

Q4_K_M is the most common choice - good balance between quality and size.

## My VRAM situation

I have an RTX 4060 with 8GB VRAM. The three models I use:

| Model | Purpose | VRAM |
|-------|---------|------|
| qwen2.5:7b | Reasoning and tool calling | ~4.7 GB |
| qwen2.5vl:7b | Vision (images, video) | ~4.5 GB |
| qwen2.5-coder:7b | Code generation | ~4.7 GB |

Total if all loaded simultaneously: ~14 GB. My GPU has 8 GB. They can't all live in VRAM at the same time.

Ollama handles this automatically. When the agent calls the VLM after using the main LLM, Ollama unloads qwen2.5:7b from VRAM, loads qwen2.5vl:7b, and runs the request. This swap takes about 3-5 seconds. It's not fast, but it works, and it happens transparently in the background.

This is a real limitation of running on consumer hardware. I documented it as a known limitation rather than pretending it doesn't exist.

## Why Qwen2.5 instead of Llama

I started with Llama 3.2 3B. It was faster and smaller, and I thought smaller = better for a local setup.

It didn't work. Llama 3.2 3B's tool calling was unreliable - it would sometimes format the tool call correctly, sometimes not, and sometimes ignore tools entirely. LangGraph's ToolNode is strict about the format. When the model produces a malformed tool call, the whole graph fails.

I switched to Qwen2.5 7B. Tool calling worked immediately and consistently. The lesson: for agentic workflows, tool-calling reliability matters more than raw speed.

## Gotchas and lessons learned

- **The VRAM formula is approximate.** Real usage is higher than the formula predicts because of KV cache (the memory used to store the context window) and runtime overhead. Add 10-20% to your estimate.
- **Model swaps have latency.** When Ollama swaps models, there's a 3-5 second pause. Design your agent to minimize model switches (e.g., don't alternate between LLM and VLM on every turn).
- **Bigger isn't always better.** Qwen2.5 7B is substantially more capable than Llama 3.2 3B for tool calling, even though both are "small" models. Architecture and training data matter as much as size.
- **num_ctx costs VRAM.** Every token in your context window costs VRAM for the KV cache. Setting `num_ctx=8192` uses more VRAM than `num_ctx=2048`. If you're close to the VRAM limit, reducing context window is one of the first levers.

---

<- [01 Foundations](README.md) | [Home](../README.md) | [LLM Basics](llm-basics.md) ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](README.md) · [Quantization](quantization.md) · [LLM Basics](llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](../04-building-the-agent/README.md) · [V1 Basic Agent](../04-building-the-agent/v1-basic-agent.md) · [Input Router](../04-building-the-agent/input-router.md) · [Tools & ReAct Loop](../04-building-the-agent/tools-react-loop.md) · [Code Generation Node](../04-building-the-agent/code-generation-node.md) · [Docker Sandbox](../04-building-the-agent/docker-sandbox.md)
