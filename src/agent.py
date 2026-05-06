from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
from langgraph.types import Command

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
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

# Define LLM and VLM
llm = ChatOllama(
    model="qwen2.5:7b",
    num_ctx=8192,
    temperature=0.1,
    num_predict=1024,
    top_p=0.9,
)
vlm = ChatOllama(
    model="qwen2.5vl:7b",
    num_ctx=8192,
    temperature=0.1,
    num_predict=1024,
    top_p=0.9,
)


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
                "agent-sandbox",
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


tools = [search_web, execute_code, analyze_image]
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

RULE: If the user says hi, hello, how are you, or asks a general question — respond directly. DO NOT use any tool.

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


# Define the agent node
def agent_node(state: AgentState) -> dict:
    chain = prompt | llm_with_tools
    response = chain.invoke(
        {
            "date": date.today().isoformat(),
            "messages": state["messages"],
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
            return "human_approval_node"
        return "tool_node"
    return "output_parser_node"


# Decide if we need human approval (for code execution) or can directly call tool
def should_execute_tool(state: AgentState) -> str:
    feedback = state.get("human_feedback", "")
    if feedback.lower().startswith("yes"):
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

    if human_response.lower().startswith("yes"):
        return {"human_feedback": human_response}
    else:
        return {
            "human_feedback": human_response,
            "messages": [
                HumanMessage(
                    content=f"Tool call rejected. Human feedback: {human_response}. Please revise your approach."
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
    match = re.search(r"(/[\w/.\-_]+\.(?:jpg|jpeg|png|gif|webp))", user_input)
    if match:
        image_path = match.group(1)
        # remove path from message
        clean_message = user_input.replace(image_path, "").strip()
        return clean_message, image_path
    return user_input, None


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent_node", agent_node)
graph.add_node("output_parser_node", output_parser_node)
graph.add_node("tool_node", ToolNode(tools))
graph.set_entry_point("agent_node")
graph.add_edge("tool_node", "agent_node")
graph.add_node("human_approval_node", human_approval_node)
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
        "human_approval_node": "human_approval_node",
        "tool_node": "tool_node",
        "output_parser_node": "output_parser_node",
    },
)

# Run the agent
if __name__ == "__main__":
    with SqliteSaver.from_conn_string("memory.db") as memory:
        app = graph.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "test-003"}}
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
