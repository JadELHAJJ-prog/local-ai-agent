import uuid
from datetime import date

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import interrupt

from state import AgentState
from config import CODE_PATTERNS, APPROVAL_PHRASES, DOCUMENT_EXTENSIONS
from models import llm, coder_llm
from tools import tools

# Bind tools to the reasoning LLM once at module load so every agent_node call reuses the same binding
llm_with_tools = llm.bind_tools(tools)

# System prompt injected at runtime so the agent always knows today's date
# MessagesPlaceholder passes the full conversation history into the prompt as-is
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


# Slice the message list down to the most recent window to stay within the model context limit
def trim_messages_window(messages: list, max_messages: int = 20) -> list:
    # Only trim when history has grown beyond the allowed window size
    if len(messages) > max_messages:
        return messages[-max_messages:]
    return messages


# --- Router ---
# Classify the input type so the graph can dispatch to the correct specialized node
def input_router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    # Normalize to lowercase for case-insensitive pattern matching
    content = last_message.content.lower()

    # File or media path was attached by parse_user_input: identify document type by extension
    if "[file provided at path:" in content or "[image provided at path:" in content:
        # Walk each known extension until one matches the message content
        for ext, doc_type in DOCUMENT_EXTENSIONS.items():
            if ext in content:
                return {"input_type": f"document_{doc_type}"}
        # Path present but extension is a media type, not a document format
        return {"input_type": "media"}

    # One or more code-request keywords detected: skip the general agent, go to code generation
    if any(pattern in content for pattern in CODE_PATTERNS):
        return {"input_type": "code"}

    # Default path for greetings, questions, and anything not classified above
    return {"input_type": "general"}


def should_route(state: AgentState) -> str:
    input_type = state.get("input_type", "general")
    # Code requests bypass the general agent and go straight to the coder model
    if input_type == "code":
        return "code_generation_node"
    # All other input types enter the main reasoning loop
    return "agent_node"


# --- Agent ---
def agent_node(state: AgentState) -> dict:
    chain = prompt | llm_with_tools
    response = chain.invoke(
        {
            "date": date.today().isoformat(),
            # Trim history before each LLM call to prevent context overflow
            "messages": trim_messages_window(state["messages"]),
        }
    )
    return {"messages": [response]}


def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # Agent produced a tool call: route based on which tool was requested
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        # execute_code always goes through the coder model for improvement before human review
        if tool_name == "execute_code":
            return "code_generation_node"
        # All other tools run directly through ToolNode without an approval step
        return "tool_node"
    # No tool call produced: send the response text to the output validator
    return "output_parser_node"


# --- Code generation ---
def _build_code_prompt(action: str, content: str) -> str:
    # Shared instruction header ensures both generation and improvement paths use identical format rules
    return (
        f"You are an expert Python developer.\n"
        f"{action}. Return ONLY raw Python code, no markdown, no explanation:\n\n{content}"
    )


def code_generation_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]

    # Router entry point: no tool call exists yet, generate fresh code from the user request
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        user_request = last_message.content
        response = coder_llm.invoke(
            [HumanMessage(content=_build_code_prompt("Write Python code for this task", user_request))]
        )
        code = _strip_markdown(response.content)

        # Wrap the generated code in an AIMessage that looks like a tool call so ToolNode can run it
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
        return {"messages": [new_message]}

    # Agent entry point: the LLM already proposed rough code, pass it to the coder model for improvement
    tool_call = last_message.tool_calls[0]
    rough_code = tool_call["args"]["code"]
    response = coder_llm.invoke(
        [HumanMessage(content=_build_code_prompt("Improve and optimize this code", rough_code))]
    )
    improved_code = _strip_markdown(response.content)

    # Preserve the original message id and tool call id so LangGraph message deduplication works correctly
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
    return {"messages": [updated_message]}


def _strip_markdown(text: str) -> str:
    # Model used a python-tagged fence: extract the block between the opening and closing backticks
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    # Model used a generic fence with no language tag: extract between the first pair of backticks
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    # No fences found: the model returned raw code, use the text directly
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

    # Show the code body when reviewing execute_code, show raw args for any other tool
    if tool_name == "execute_code":
        print("Code to execute:")
        print("-" * 30)
        print(tool_args.get("code", ""))
        print("-" * 30)
    else:
        print(f"With args: {tool_args}")

    # Pause the graph here until the human provides a decision via Command(resume=...)
    human_response = interrupt("\nApprove? (yes/no + optional feedback): ")

    # Check whether the human response contains any recognized approval phrase
    approved = any(phrase in human_response.lower() for phrase in APPROVAL_PHRASES)

    if approved:
        return {"human_feedback": human_response, "approved": True}
    # Rejection: inject a feedback message so the agent rewrites on the next cycle
    else:
        return {
            "human_feedback": human_response,
            "approved": False,
            "messages": [
                HumanMessage(
                    content=f"The code was rejected. User feedback: '{human_response}'. "
                    f"Please rewrite the code addressing this feedback, then call execute_code again."
                )
            ],
        }


def should_execute_tool(state: AgentState) -> str:
    # Route to tool_node only when the human explicitly approved the tool call
    if state.get("approved", False):
        return "tool_node"
    # Rejection sends the graph back to the agent so it can rewrite based on the injected feedback
    return "agent_node"


# --- Output parser ---
# Validate that the agent produced a non-empty response and track retries for empty outputs
def output_parser_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    # A response is valid when it contains at least one non-whitespace character
    is_valid = bool(last_message.content and last_message.content.strip())
    retry_count = state.get("retry_count", 0)
    # Increment the retry counter only when validation fails
    return {
        "is_valid": is_valid,
        "retry_count": retry_count + 1 if not is_valid else retry_count,
    }


# Cap retries at 3 to avoid infinite loops on persistent empty responses
def should_retry(state: AgentState) -> str:
    # Valid response: end the graph immediately
    if state["is_valid"]:
        return "end"
    # Retry budget exhausted: end rather than looping forever
    if state.get("retry_count", 0) >= 3:
        return "end"
    # Budget remaining: send back to the agent for another attempt
    return "retry"
