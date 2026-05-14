from langchain_ollama import ChatOllama
from config import LLM_MODEL, VLM_MODEL, CODER_MODEL, NUM_CTX, TEMPERATURE, NUM_PREDICT, TOP_P

llm = ChatOllama(
    model=LLM_MODEL,
    num_ctx=NUM_CTX,
    temperature=TEMPERATURE,
    num_predict=NUM_PREDICT,
    top_p=TOP_P,
)

# Larger context window to handle multiple base64-encoded frames in vision tasks
vlm = ChatOllama(
    model=VLM_MODEL,
    num_ctx=32768,
    temperature=TEMPERATURE,
    num_predict=NUM_PREDICT,
    top_p=TOP_P,
)

# Higher num_predict allows complete code output without mid-function truncation
coder_llm = ChatOllama(
    model=CODER_MODEL,
    num_ctx=NUM_CTX,
    temperature=0.1,
    num_predict=2048,
    top_p=TOP_P,
)
