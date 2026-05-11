import json
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from graph import build_graph
from nodes import parse_user_input


def save_last_thread(thread_id: str):
    with open(".last_session", "w") as f:
        json.dump({"thread_id": thread_id}, f)


def load_last_thread() -> str:
    try:
        with open(".last_session") as f:
            return json.load(f)["thread_id"]
    except Exception:
        print("No previous session found. Starting new session.")
        return str(uuid.uuid4())


if __name__ == "__main__":
    with SqliteSaver.from_conn_string("memory.db") as memory:
        app = build_graph(memory)

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
                message = HumanMessage(
                    content=f"{text} [image provided at path: {image_path}]"
                )
            else:
                message = HumanMessage(content=text)

            for chunk, metadata in app.stream(
                {"messages": [message]},
                config=config,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") == "agent_node":
                    if hasattr(chunk, "content") and chunk.content:
                        print(chunk.content, end="", flush=True)

            state = app.get_state(config)
            while state.next:
                human_input = input("\nYour decision: ")
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
