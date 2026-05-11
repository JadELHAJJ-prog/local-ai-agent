from dotenv import load_dotenv
import os

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:7b")
CODER_MODEL = os.getenv("CODER_MODEL", "qwen2.5-coder:7b")
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "agent-sandbox")

NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "1024"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "8"))

CODE_PATTERNS = [
    "write code", "write a code", "write me a code", "write for me", "write me a",
    "write a script", "write a program", "write a function", "write a class",
    "write a module", "write a snippet",
    "create code", "create a code", "create a script", "create a program",
    "create a function", "create a class", "create a module",
    "make a code", "make a script", "make a program", "make a function",
    "make me a", "make a class",
    "give me code", "give me a code", "give me a script", "give me a program",
    "give me a function",
    "generate code", "generate a code", "generate a script", "generate a function",
    "build a", "build me a", "build a script", "build a program",
    "implement", "implement a", "implement the", "implement this",
    "run", "run this", "run a", "execute", "execute this", "execute a",
    "test this code", "test this script",
    "write and run", "code for", "script for", "program for", "function for",
    "code to", "script to", "program to", "show me the code",
    "can you code", "can you write", "i need code", "i need a script",
    "i need a program", "python code", "python script", "python function",
]

APPROVAL_PHRASES = [
    "yes", "i like", "looks good", "approved", "ok", "good", "run it", "execute",
]

DOCUMENT_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".csv": "csv",
}
