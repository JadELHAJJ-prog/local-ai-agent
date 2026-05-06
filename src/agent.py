# agent.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

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
                "python:3.11-slim",
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


tools = [search_web, execute_code]
# 1. initialize the LLM
llm = ChatOllama(
    model="qwen2.5:7b",
    num_ctx=8192,
    temperature=0.1,
    num_predict=1024,
    top_p=0.9,
)

llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Bubbles, a helpful AI assistant. Today is {date}.

TOOLS — only use when explicitly needed:
- search_web: ONLY if user asks for news, current events, or real-time data. NEVER for greetings, math, coding, or general questions.
- execute_code: ONLY if user says "run", "execute", or "test this code".

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


# 2. define the agent node
def agent_node(state: AgentState) -> dict:
    chain = prompt | llm_with_tools
    response = chain.invoke(
        {
            "date": date.today().isoformat(),
            "messages": state["messages"],
        }
    )
    return {"messages": [response]}


def is_not_empty(content: str) -> bool:
    return bool(content and content.strip())


def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return "output_parser_node"


def output_parser_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content

    is_valid = is_not_empty(content)
    retry_count = state.get("retry_count", 0)

    return {
        "is_valid": is_valid,
        "retry_count": retry_count + 1 if not is_valid else retry_count,
    }


# 3. build the graph
graph = StateGraph(AgentState)
graph.add_node("agent_node", agent_node)
graph.add_node("output_parser_node", output_parser_node)
graph.add_node("tool_node", ToolNode(tools))
graph.set_entry_point("agent_node")
graph.add_edge("tool_node", "agent_node")


def should_retry(state: AgentState) -> str:
    if state["is_valid"]:
        return "end"
    if state.get("retry_count", 0) >= 3:
        return "end"
    return "retry"


graph.add_conditional_edges(
    "output_parser_node", should_retry, {"retry": "agent_node", "end": END}
)
graph.add_conditional_edges(
    "agent_node",
    should_use_tool,
    {"tool_node": "tool_node", "output_parser_node": "output_parser_node"},
)

if __name__ == "__main__":
    with SqliteSaver.from_conn_string("memory.db") as memory:
        app = graph.compile(checkpointer=memory)
        print("Agent ready. Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            print("Agent: ", end="", flush=True)
            for chunk, metadata in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": "test-003"}},
                stream_mode="messages",
            ):
                # only print chunks from agent_node, skip tool output
                if metadata.get("langgraph_node") == "agent_node":
                    if hasattr(chunk, "content") and chunk.content:
                        print(chunk.content, end="", flush=True)
            print("\n")
