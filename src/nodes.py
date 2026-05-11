import re
import uuid
from datetime import date

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import interrupt

from state import AgentState
from config import CODE_PATTERNS, APPROVAL_PHRASES
from models import llm, coder_llm
from tools import tools

llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Bubbles, a helpful AI assistant. Today is {date}.

TOOLS — only use when explicitly needed:
- search_web: ONLY if user asks for news, current events, or real-time data. NEVER for greetings, math, coding, or general questions.
- execute_code: ONLY if user says "run", "execute", or "test this code".
- analyze_image: use when user provides an image path ending in .jpg .jpeg .png .gif .webp. Extract the path and analyze it.
- analyze_video: use when user provides a video path ending in .mp4 .avi .mov .mkv. Extract the path and analyze it.

RULE: If the user says hi, hello, how are you, or asks a general question — respond directly. DO NOT use any tool.

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


def trim_messages_window(messages: list, max_messages: int = 20) -> list:
    if len(messages) > max_messages:
        return messages[-max_messages:]
    return messages


def parse_user_input(user_input: str) -> tuple[str, str | None]:
    match = re.search(
        r"(/[\w/.\-_]+\.(?:jpg|jpeg|png|gif|webp|mp4|avi|mov|mkv))", user_input
    )
    if match:
        image_path = match.group(1)
        clean_message = user_input.replace(image_path, "").strip()
        return clean_message, image_path
    return user_input, None


# --- Router ---

def input_router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content.lower()

    if "[file provided at path:" in content or "[image provided at path:" in content:
        return {"input_type": "media"}

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
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        if tool_name == "execute_code":
            if not state.get("code_generated", False):
                return "code_generation_node"
            return "human_approval_node"
        return "tool_node"
    return "output_parser_node"


# --- Code generation ---

def code_generation_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]

    # coming from input_router — no tool_calls yet, generate from user request
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

    # coming from agent_node — improve existing code in the tool_call
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
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text


# --- Human approval ---

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
    else:
        print(f"With args: {tool_args}")

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


def should_execute_tool(state: AgentState) -> str:
    feedback = state.get("human_feedback", "").lower()
    if any(phrase in feedback for phrase in APPROVAL_PHRASES):
        return "tool_node"
    return "agent_node"


# --- Output parser ---

def output_parser_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    is_valid = bool(last_message.content and last_message.content.strip())
    retry_count = state.get("retry_count", 0)
    return {
        "is_valid": is_valid,
        "retry_count": retry_count + 1 if not is_valid else retry_count,
    }


def should_retry(state: AgentState) -> str:
    if state["is_valid"]:
        return "end"
    if state.get("retry_count", 0) >= 3:
        return "end"
    return "retry"
