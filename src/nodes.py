import re
import uuid
from datetime import date

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import interrupt

from state import AgentState
from config import CODE_PATTERNS, APPROVAL_PHRASES, DOCUMENT_EXTENSIONS
from models import llm, coder_llm
from tools import tools

llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Bubbles, a helpful AI assistant. Today is {date}.

TOOLS - only use when explicitly needed:
- search_web: ONLY if user asks for news, current events, or real-time data. NEVER for greetings, math, coding, or general questions.
- execute_code: ONLY if user says "run", "execute", or "test this code".
- analyze_image: use when user provides an image path ending in .jpg .jpeg .png .gif .webp. Extract the path and analyze it.
- analyze_video: use when user provides a video path ending in .mp4 .avi .mov .mkv. Extract the path and analyze it.
- analyze_document: use when user provides a file path ending in .pdf .docx .xlsx .xls .csv. Extract the path and analyze it.

RULE: If the user says hi, hello, how are you, or asks a general question - respond directly. DO NOT use any tool.

- If a tool call is rejected with feedback, you MUST call the tool again with the corrected approach. Never answer directly after a rejection.
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


# Bound conversation history to prevent context overflow on long sessions
def trim_messages_window(messages: list, max_messages: int = 20) -> list:
    # Only slice when history has grown beyond the allowed window
    if len(messages) > max_messages:
        return messages[-max_messages:]
    return messages


# Extract an attached file or media path from the raw message so it can be labeled separately
def parse_user_input(user_input: str) -> tuple[str, str | None]:
    match = re.search(
        r"(/[\w/.\-_]+\.(?:jpg|jpeg|png|gif|webp|mp4|avi|mov|mkv|pdf|docx|xlsx|xls|csv))",
        user_input,
    )
    # A file or media path was found; strip it from the text and return it separately
    if match:
        image_path = match.group(1)
        clean_message = user_input.replace(image_path, "").strip()
        return clean_message, image_path
    return user_input, None


# --- Router ---
# Classify the input type so the graph can dispatch to the correct specialized node
def input_router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content.lower()

    # Message contains an attached file or media marker injected by parse_user_input
    if "[file provided at path:" in content or "[image provided at path:" in content:
        # Match the extension against known document types to get a specific parser label
        for ext, doc_type in DOCUMENT_EXTENSIONS.items():
            # Stop at the first extension that appears in the message content
            if ext in content:
                return {"input_type": f"document_{doc_type}"}
        # Path present but extension is a media type, not a document
        return {"input_type": "media"}

    # Message matches one of the code-request trigger phrases
    if any(pattern in content for pattern in CODE_PATTERNS):
        return {"input_type": "code"}

    return {"input_type": "general"}


def should_route(state: AgentState) -> str:
    input_type = state.get("input_type", "general")
    # Code requests skip the general agent and go straight to the coder model
    if input_type == "code":
        return "code_generation_node"
    return "agent_node"


# --- Agent ---
def agent_node(state: AgentState) -> dict:
    chain = prompt | llm_with_tools
    response = chain.invoke(
        {
            "date": date.today().isoformat(),
            "messages": trim_messages_window(state["messages"]),
        }
    )
    return {"messages": [response]}


def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # The agent requested a tool call, decide which path to follow
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        # execute_code needs special treatment: improve the code then get human sign-off
        if tool_name == "execute_code":
            # Code has not been through the coder model yet, improve it first
            if not state.get("code_generated", False):
                return "code_generation_node"
            # Code is ready, route to human approval before running
            return "human_approval_node"
        # All other tools run directly without approval
        return "tool_node"
    # No tool call requested, validate the response text
    return "output_parser_node"


# --- Code generation ---
def code_generation_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]

    # Entry from input_router: no tool_call yet, so generate fresh code from the user request
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        user_request = last_message.content
        code_prompt = f"""You are an expert Python developer.
Write Python code for this task. Return ONLY raw Python code, no markdown, no explanation:

{user_request}"""
        response = coder_llm.invoke([HumanMessage(content=code_prompt)])
        code = _strip_markdown(response.content)

        new_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute_code",
                    "args": {"code": code},
                    "id": f"call-{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }
            ],
        )
        return {"messages": [new_message], "code_generated": True}

    # Entry from agent_node: LLM already produced rough code, pass it to coder model for improvement
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


def _strip_markdown(text: str) -> str:
    # Prefer the python-fenced block when the model used a language tag
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    # Fall back to a generic fenced block when no language tag is present
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    # No fences found, return the text as-is
    return text


# --- Human approval ---
# Pause graph execution and surface the pending tool call for human review before running it
def human_approval_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print(f"\n{'='*50}")
    print(f"Agent wants to call: '{tool_name}'")
    print(f"{'='*50}")

    # Show the code body for execute_code; show raw args for any other tool
    if tool_name == "execute_code":
        print("Code to execute:")
        print("-" * 30)
        print(tool_args.get("code", ""))
        print("-" * 30)
    else:
        print(f"With args: {tool_args}")

    human_response = interrupt("\nApprove? (yes/no + optional feedback): ")

    # Treat the response as approved if it starts with "yes" or matches any approval phrase
    approved = human_response.lower().startswith("yes") or any(
        phrase in human_response.lower() for phrase in APPROVAL_PHRASES
    )

    # On approval, clear the code_generated flag so the next run starts fresh
    if approved:
        return {"human_feedback": human_response, "code_generated": False}
    # On rejection, inject a feedback message so the agent rewrites the code
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


def should_execute_tool(state: AgentState) -> str:
    feedback = state.get("human_feedback", "").lower()
    # Only execute the tool when the human explicitly approved it
    if any(phrase in feedback for phrase in APPROVAL_PHRASES):
        return "tool_node"
    # Rejection routes back to the agent so it can rewrite based on the feedback message
    return "agent_node"


# --- Output parser ---
# Validate that the agent produced a non-empty response and track retries for empty outputs
def output_parser_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    is_valid = bool(last_message.content and last_message.content.strip())
    retry_count = state.get("retry_count", 0)
    return {
        "is_valid": is_valid,
        "retry_count": retry_count + 1 if not is_valid else retry_count,
    }


# Cap retries at 3 to avoid infinite loops on persistent empty responses
def should_retry(state: AgentState) -> str:
    # Output was non-empty, nothing to retry
    if state["is_valid"]:
        return "end"
    # Exhausted all retries, give up rather than looping forever
    if state.get("retry_count", 0) >= 3:
        return "end"
    return "retry"
