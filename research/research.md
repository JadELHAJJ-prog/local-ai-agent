# Research phase

## Questions I need to answer
- What inference runtimes exist for running LLMs locally?
- What agent frameworks exist and what do they abstract?
- Which models fit on 8GB VRAM?
- What is quantization and why does it matter?

## What I learned
- What is quantization LLM explained: based on Medium, quantization is a compression technique that involes mapping high precision values to a lower precision one. That means modifyng the precision of their weights and activations making less memory intensive. This targets, memory, gpu and cost positevly and accuracy negatively.

- GGUF vs GPTQ: are quantization methods. GGUF by llama.cpp team it allows users to run LLMs on a cpu while ofloading some layers to GPU and offers speed improvements. it scales down model weights. GPTQ is one shot weight quantization based on approximate second order info. it supports quantization to 8 4 3 or even2 bits without significatn drop in perfm. 

## Inference runtimes comparison

| Tool | How you run it | Supports GPU offload? | Has Python API? | Best for |
|------|---------------|----------------------|-----------------|---------|
| Ollama | ollama run llama3| yes, cuda| yes| running LLMs locally on your own machine to ensure total privacy, zero API costs, and offline functionality |
| llama.cpp | clone repo and follow instructions| yes | yes, llama-cpp-python | running LLMs locally on consumer-grade hardware with maximum efficiency and minimal dependencies |
| LM Studio | lmstudio.ai| yes | yes python sdk| running LLMs locally on your computer|
| vLLM | pip install vllm| yes | yes | high-throughput, cost-efficient, and fast production serving of LLMs |

## VRAM rule of thumb
At 4-bit quantization: ~0.5–0.7 GB VRAM per billion parameters.
- 7B model ≈ 4–5 GB  fits
- 13B model ≈ 7–9 GB  tight  
- 70B model ≈ 35–40 GB  not possible
Ollama uses GGUF format by default.

## VRAM formula for 4bit quantization
VRAM needed = (params × 0.5) + 1 to 2 GB overhead


### vLLM evaluation
- **VRAM**: RTX 4060 8GB is sufficient for serving a 7B model at 4-bit quantization
- **Vision support**: vLLM supports vision-language models including LLaVA
- **Python API**: Native Python API via `from vllm import LLM, SamplingParams`, 
  also exposes an OpenAI-compatible REST API

### Decision rationale
This project is a Personal AI Agent with Tools — an agent that 
can browse the web, read files, run code and answer questions, 
all running locally. Ollama reduces setup complexity compared to 
vLLM and lets me focus on agent architecture rather than 
inference infrastructure. Therefore I chose Ollama as the 
inference runtime, accepting the tradeoff of less 
production-grade serving capability in exchange for faster setup 
and more focus on agent architecture.

## Agent frameworks comparison

| Framework | What it abstracts | Learning curve | Best for |
|-----------|------------------|----------------|---------|
| From scratch (pure Python) | nothing | very steep| zero to hero without deadline|
 LangChain | it provides a prebuilt agent architecture and model integrations| steep, lots of abstraction layers| end users not learners |
| LlamaIndex | data ingestion pipeline and query interfaces over documents and vector stores | good | everyone |
| smolagents | the agent loop(think->act->observe cycle) | not that much | learners who want minimal abstraction without building everything from scratch. |
| AutoGen | multi agent communications to solve an issue | high but overkill for our usecase | huge projects|

### My conclusion
I chose LangChain over smolagents for two reasons: it aligns 
with an internal company project, and LangChain/LangGraph is 
the production standard for complex, reliable multi-agent 
systems with durable state management and human-in-the-loop 
capabilities. The tradeoff is accepting more abstraction layers, 
which means some underlying mechanisms will be hidden. This is 
acceptable given the project timeline and business context.

## Model selection

### LLM candidates (reasoning + tool use)
| Model | Params | VRAM at 4-bit | Tool use support? | Notes |
|-------|--------|--------------|-------------------|-------|
| Llama 3.2 3B | 3B | 2-3 GB | yes | https://ollama.com/library/llama3.2|
| Mistral 7B | 7B | 4-5 GB | yes | https://ollama.com/library/mistral |
| Qwen2.5 7B | 7B| 4-5 GB | yes | https://ollama.com/library/qwen2.5|

### VLM candidates (vision)
| Model | VRAM at 4-bit | Image understanding quality | Notes |
|-------|--------------|----------------------------|-------|
| LLaVA 1.6 7B | 4-5 GB | good | (https://ollama.com/library/llava) |
| moondream2 | 2-3 GB | basic | https://ollama.com/library/moondream|
| Qwen2.5-VL 7b | 4-5 GB | excellent | https://ollama.com/library/qwen2.5vl|

### My choices

*LLM: and why*: Llama 3.2 is lighter and fits easily alongside a VLM in 8GB VRAM and it is designed to act as an agentic AI.

*VLM: and why*: Qwen2.5-VL  understand things visually, being agentic, capable of visual localization in different formats, generated structured outputs, better in high resolution image/video understanding and document parsing. . While LLaVA 1.6 7B is a strong, widely used, and fast model, Qwen2.5-VL 7B provides better OCR, fine-grained details, and deeper integration of visual and textual information.

### VRAM budget check
- Llama 3.2 3B: ~2.5 GB
- Qwen2.5-VL 7B: ~4.5 GB
- Total: ~7 GB
- Available: 8 GB
- Headroom: ~1 GB (tight)

**Decision:** models will be loaded one at a time when possible, 
or swapped depending on the task — text tasks use LLM only, 
vision tasks load VLM.


## Final tech stack
| Layer | Tool | Reason |
|-------|------|--------|
| Inference runtime | Ollama | local, simple, GGUF |
| LLM | Llama 3.2 3B | fits VRAM, tool calling |
| VLM | Qwen2.5-VL 7B | best vision quality in budget |
| Agent framework | LangChain + LangGraph | production standard, company alignment |
| Memory/persistence | SQLite via SqliteSaver | persistent, crash-safe |