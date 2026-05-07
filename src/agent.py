import cv2
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
from langgraph.types import Command

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from state import AgentState
from ddgs import DDGS

import subprocess
import tempfile
import os
from datetime import date
import re
import base64
from dotenv import load_dotenv
import uuid
import json


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def save_last_thread(thread_id: str):
    with open(".last_session", "w") as f:
        json.dump({"thread_id": thread_id}, f)


def load_last_thread() -> str:
    try:
        with open(".last_session") as f:
            return json.load(f)["thread_id"]
    except:
        print("No previous session found. Starting new session.")
        return str(uuid.uuid4())


load_dotenv()
# models
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:7b")
CODER_MODEL = os.getenv("CODER_MODEL", "qwen2.5-coder:7b")
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "agent-sandbox")

# hyperparameters
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "1024"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "8"))

# Define LLM and VLM
llm = ChatOllama(
    model=LLM_MODEL,
    num_ctx=NUM_CTX,
    temperature=TEMPERATURE,
    num_predict=NUM_PREDICT,
    top_p=TOP_P,
)

vlm = ChatOllama(
    model=VLM_MODEL,
    num_ctx=32768,
    temperature=TEMPERATURE,
    num_predict=NUM_PREDICT,
    top_p=TOP_P,
)

coder_llm = ChatOllama(
    model=CODER_MODEL,
    num_ctx=NUM_CTX,
    temperature=0.1,
    num_predict=2048,
    top_p=TOP_P,
)

CODE_PATTERNS = [
    # write variants
    "write code",
    "write a code",
    "write me a code",
    "write for me",
    "write me a",
    "write a script",
    "write a program",
    "write a function",
    "write a class",
    "write a module",
    "write a snippet",
    # create variants
    "create code",
    "create a code",
    "create a script",
    "create a program",
    "create a function",
    "create a class",
    "create a module",
    # make variants
    "make a code",
    "make a script",
    "make a program",
    "make a function",
    "make me a",
    "make a class",
    # give variants
    "give me code",
    "give me a code",
    "give me a script",
    "give me a program",
    "give me a function",
    # generate variants
    "generate code",
    "generate a code",
    "generate a script",
    "generate a function",
    # build variants
    "build a",
    "build me a",
    "build a script",
    "build a program",
    # implement variants
    "implement",
    "implement a",
    "implement the",
    "implement this",
    # execution variants
    "run",
    "run this",
    "run a",
    "execute",
    "execute this",
    "execute a",
    "test this code",
    "test this script",
    # other common patterns
    "write and run",
    "code for",
    "script for",
    "program for",
    "function for",
    "code to",
    "script to",
    "program to",
    "show me the code",
    "can you code",
    "can you write",
    "i need code",
    "i need a script",
    "i need a program",
    "python code",
    "python script",
    "python function",
]

APPROVAL_PHRASES = [
    "yes",
    "i like",
    "looks good",
    "approved",
    "ok",
    "good",
    "run it",
    "execute",
]


def input_router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content.lower()

    # check for media path
    if "[file provided at path:" in content or "[image provided at path:" in content:
        return {"input_type": "media"}

    # check for code request
    if any(pattern in content for pattern in CODE_PATTERNS):
        return {"input_type": "code"}

    return {"input_type": "general"}


def should_route(state: AgentState) -> str:
    input_type = state.get("input_type", "general")
    if input_type == "code":
        return "code_generation_node"
    if input_type == "media":
        return "tool_node"
    return "agent_node"


# Define tools
@tool
def execute_code(code: str) -> str:
    """Execute Python code safely in an isolated Docker container.
    Use this when the user asks to run code, perform calculations,
    or test a Python script. Input should be valid Python code."""
    tmp_path = None
    try:
        # step 1 - write code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        # step 2 - run in docker sandbox
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--memory",
                "128m",
                "--cpus",
                "0.5",
                "-v",
                f"{tmp_path}:/app/code.py:ro",
                SANDBOX_IMAGE,
                "python",
                "/app/code.py",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # step 3 - return output or error
        if result.returncode == 0:
            return result.stdout or "Code executed successfully with no output."
        else:
            return f"Error:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"
    finally:
        # step 4 - cleanup
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@tool
def search_web(query: str) -> str:
    """Search the web for current information, recent events,
    or anything that requires up-to-date knowledge beyond
    your training data. Input should be a search query string."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        if not results:
            return "No results found."
        return "\n".join(
            [
                f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}"
                for r in results
            ]
        )


@tool
def analyze_image(image_path: str, question: str = "What is in this image?") -> str:
    """Analyze an image using vision AI. Use this when the user
    provides an image path and wants to know what's in it,
    extract text from it, or ask questions about it.
    Input should be the path to the image file."""

    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )

    vlm_response = vlm.invoke([message])
    return (
        vlm_response.content
        if vlm_response.content
        else "No response from vision model."
    )


@tool
def analyze_video(
    video_path: str,
    question: str = "What is happening in this video?",
) -> str:
    """Analyze a video using vision AI by sampling key frames.
    Use this when the user provides a video file path (.mp4, .avi, .mov, .mkv).
    Input should be the path to the video file."""

    max_frames = MAX_FRAMES

    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Error: Could not open video file at {video_path}"

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // max_frames)
        frames_analyzed = 0

        # build one message with all frames + question
        content = [{"type": "text", "text": question}]

        for frame_idx in range(0, total_frames, interval):
            if frames_analyzed >= max_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (512, 512))
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            base64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")

            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

            frames_analyzed += 1
            print(f"Extracted frame {frames_analyzed}/{max_frames}...", flush=True)

        if frames_analyzed == 0:
            return "No frames could be extracted from this video."

        print("Sending frames to vision model...", flush=True)

        # one single VLM call with all frames
        message = HumanMessage(content=content)
        vlm_response = vlm.invoke([message])

        return vlm_response.content or "No response from vision model."

    finally:
        cap.release()


tools = [search_web, execute_code, analyze_image, analyze_video]
llm_with_tools = llm.bind_tools(tools)

# Define prompt template with tool instructions and examples
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Bubbles, a helpful AI assistant. Today is {date}.

TOOLS — only use when explicitly needed:
- search_web: ONLY if user asks for news, current events, or real-time data. NEVER for greetings, math, coding, or general questions.
- execute_code: ONLY if user says "run", "execute", or "test this code".
- analyze_image: use when user provides an image path ending in 
  .jpg .jpeg .png .gif .webp. Extract the path and analyze it.
- analyze_video: use when user provides a video path ending in 
  .mp4 .avi .mov .mkv. Extract the path and analyze it.

RULE: If the user says hi, hello, how are you, or asks a general question — respond directly. DO NOT use any tool.

- If a tool call is rejected with feedback, you MUST call the 
  tool again with the corrected approach. Never answer directly 
  after a rejection.
  
Examples of NO tool needed:
- "hi" -> just greet back
- "what is 2+2" -> just answer
- "who are you" -> just introduce yourself

Examples of tool needed:
- "what is the latest news about AI" -> use search_web
- "run this python script" -> use execute_code""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def trim_messages_window(messages: list, max_messages: int = 20) -> list:
    """Keep only the last N messages to avoid context overflow."""
    if len(messages) > max_messages:
        return messages[-max_messages:]
    return messages


# Define the agent node
def agent_node(state: AgentState) -> dict:
    chain = prompt | llm_with_tools
    response = chain.invoke(
        {
            "date": date.today().isoformat(),
            "messages": trim_messages_window(state["messages"]),
        }
    )
    return {"messages": [response]}


# Helper functions for graph logic
def is_not_empty(content: str) -> bool:
    return bool(content and content.strip())


# Decision functions for graph edges
def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        if tool_name == "execute_code":
            if not state.get("code_generated", False):
                return "code_generation_node"
            return "human_approval_node"
        return "tool_node"
    return "output_parser_node"


# Decide if we need human approval (for code execution) or can directly call tool
def should_execute_tool(state: AgentState) -> str:
    feedback = state.get("human_feedback", "").lower()
    if any(phrase in feedback for phrase in APPROVAL_PHRASES):
        return "tool_node"
    return "agent_node"


# Decide if we should retry the agent's response or end the graph execution
def output_parser_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content

    is_valid = is_not_empty(content)
    retry_count = state.get("retry_count", 0)

    return {
        "is_valid": is_valid,
        "retry_count": retry_count + 1 if not is_valid else retry_count,
    }


# Node to get human approval before executing code tool
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
        # this fixes the \n display issue
        print(tool_args.get("code", ""))
        print("-" * 30)
    else:
        print(f"With args: {tool_args}")

    human_response = interrupt("\nApprove? (yes/no + optional feedback): ")

    if human_response.lower().startswith("yes") or any(
        phrase in human_response.lower() for phrase in APPROVAL_PHRASES
    ):
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


# Decide if we should retry the agent's response or end the graph execution
def should_retry(state: AgentState) -> str:
    if state["is_valid"]:
        return "end"
    if state.get("retry_count", 0) >= 3:
        return "end"
    return "retry"


# Parse user input to extract image path if present, and clean the message for the agent
def parse_user_input(user_input: str) -> tuple[str, str | None]:
    """Extract image path from user message if present."""
    match = re.search(
        r"(/[\w/.\-_]+\.(?:jpg|jpeg|png|gif|webp|mp4|avi|mov|mkv))", user_input
    )
    if match:
        image_path = match.group(1)
        # remove path from message
        clean_message = user_input.replace(image_path, "").strip()
        return clean_message, image_path
    return user_input, None


# Node to generate code using the coder model
def code_generation_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]

    # coming from input_router — no tool_calls
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        user_request = last_message.content
        code_prompt = f"""You are an expert Python developer.
Write Python code for this task. Return ONLY raw Python code, no markdown, no explanation:

{user_request}"""

        response = coder_llm.invoke([HumanMessage(content=code_prompt)])
        improved_code = response.content
        if "```python" in improved_code:
            improved_code = improved_code.split("```python")[1].split("```")[0].strip()
        elif "```" in improved_code:
            improved_code = improved_code.split("```")[1].split("```")[0].strip()

        import uuid as _uuid

        new_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute_code",
                    "args": {"code": improved_code},
                    "id": f"call-{_uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }
            ],
        )
        return {"messages": [new_message], "code_generated": True}

    # coming from agent_node — has tool_calls, improve existing code
    tool_call = last_message.tool_calls[0]
    rough_code = tool_call["args"]["code"]
    code_prompt = f"""You are an expert Python developer.
Improve and optimize this code. Return ONLY raw Python code, no markdown, no explanation:

{rough_code}"""
    response = coder_llm.invoke([HumanMessage(content=code_prompt)])
    improved_code = response.content
    if "```python" in improved_code:
        improved_code = improved_code.split("```python")[1].split("```")[0].strip()
    elif "```" in improved_code:
        improved_code = improved_code.split("```")[1].split("```")[0].strip()
    updated_message = AIMessage(
        id=last_message.id,
        content=last_message.content,
        tool_calls=[
            {
                "name": "execute_code",
                "args": {"code": improved_code},
                "id": tool_call["id"],
                "type": "tool_call",
            }
        ],
    )
    return {"messages": [updated_message], "code_generated": True}


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent_node", agent_node)
graph.add_node("output_parser_node", output_parser_node)
graph.add_node("tool_node", ToolNode(tools))
graph.set_entry_point("input_router_node")
graph.add_node("input_router_node", input_router_node)
graph.add_edge("tool_node", "agent_node")
graph.add_node("human_approval_node", human_approval_node)
graph.add_node("code_generation_node", code_generation_node)
graph.add_edge("code_generation_node", "human_approval_node")
graph.add_conditional_edges(
    "input_router_node",
    should_route,
    {
        "code_generation_node": "code_generation_node",
        "tool_node": "tool_node",
        "agent_node": "agent_node",
    },
)
graph.add_conditional_edges(
    "human_approval_node",
    should_execute_tool,
    {"tool_node": "tool_node", "agent_node": "agent_node"},
)
graph.add_conditional_edges(
    "output_parser_node", should_retry, {"retry": "agent_node", "end": END}
)
graph.add_conditional_edges(
    "agent_node",
    should_use_tool,
    {
        "code_generation_node": "code_generation_node",
        "human_approval_node": "human_approval_node",
        "tool_node": "tool_node",
        "output_parser_node": "output_parser_node",
    },
)

# Run the agent
if __name__ == "__main__":
    with SqliteSaver.from_conn_string("memory.db") as memory:
        app = graph.compile(checkpointer=memory)

        print("=== Bubbles AI Agent ===")
        print("1. Start new conversation")
        print("2. Continue previous conversation")
        while True:
            choice = input("Choose (1/2): ").strip()
            if choice in ("1", "2"):
                break
            print("Invalid choice. Please enter 1 or 2.")

        if choice == "2":
            thread_id = load_last_thread()
            print(f"Continuing session: {thread_id}\n")
        else:
            thread_id = str(uuid.uuid4())
            print(f"New session started: {thread_id}")
            print(f"Save this ID to continue later: {thread_id}\n")
        save_last_thread(thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        print("Agent ready. Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            print("Agent: ", end="", flush=True)
            text, image_path = parse_user_input(user_input)
            if image_path:
                # inject image path into message so agent knows to use analyze_image
                message = HumanMessage(
                    content=f"{text} [image provided at path: {image_path}]"
                )
            else:
                message = HumanMessage(content=text)

            # stream until interrupt or end
            interrupted = False
            for chunk, metadata in app.stream(
                {"messages": [message]},
                config=config,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") == "agent_node":
                    if hasattr(chunk, "content") and chunk.content:
                        print(chunk.content, end="", flush=True)

            # check if graph was interrupted
            state = app.get_state(config)
            while state.next:  # graph is paused waiting for human
                human_input = input("\nYour decision: ")
                # resume with human input
                for chunk, metadata in app.stream(
                    Command(resume=human_input),
                    config=config,
                    stream_mode="messages",
                ):
                    if metadata.get("langgraph_node") == "agent_node":
                        if hasattr(chunk, "content") and chunk.content:
                            print(chunk.content, end="", flush=True)
                state = app.get_state(config)
            print("\n")
