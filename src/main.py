import json
import os
import re
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from config import DOCUMENT_EXTENSIONS
from graph import build_graph


# Extract an attached file or media path from the raw message so it can be labeled separately
def parse_user_input(user_input: str) -> tuple[str, str | None]:
    # Match any absolute path whose extension is a supported image, video, or document type
    match = re.search(
        r"(/[\w/.\-_]+\.(?:jpg|jpeg|png|gif|webp|mp4|avi|mov|mkv|pdf|docx|xlsx|xls|csv))",
        user_input,
    )
    # A file or media path was found: strip it from the text and return it separately
    if match:
        image_path = match.group(1)
        clean_message = user_input.replace(image_path, "").strip()
        return clean_message, image_path
    # No path found: return the original input unchanged with no path
    return user_input, None


# Filter the stream to agent_node chunks only and print each content piece as it arrives
def _stream_agent(app, input_, config):
    for chunk, metadata in app.stream(input_, config=config, stream_mode="messages"):
        # Skip chunks from router, tool, approval, and parser nodes — only show agent text
        if metadata.get("langgraph_node") == "agent_node":
            # Guard against chunks that carry no text content such as tool call scaffolding
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)


# Persist the last thread ID so users can resume their session across restarts
def save_last_thread(thread_id: str):
    with open(".last_session", "w") as f:
        json.dump({"thread_id": thread_id}, f)


def load_last_thread() -> str:
    try:
        with open(".last_session") as f:
            return json.load(f)["thread_id"]
    # Fall back to a new session ID if the file is missing or corrupted
    except Exception:
        print("No previous session found. Starting new session.")
        return str(uuid.uuid4())


if __name__ == "__main__":
    with SqliteSaver.from_conn_string("memory.db") as memory:
        app = build_graph(memory)

        print("=== Bubbles AI Agent ===")
        print("1. Start new conversation")
        print("2. Continue previous conversation")
        # Keep asking until the user enters a valid menu option
        while True:
            choice = input("Choose (1/2): ").strip()
            if choice in ("1", "2"):
                break
            print("Invalid choice. Please enter 1 or 2.")

        # Restore the saved thread ID to continue where the last session left off
        if choice == "2":
            thread_id = load_last_thread()
            print(f"Continuing session: {thread_id}\n")
        # Generate a fresh UUID for a brand new conversation
        else:
            thread_id = str(uuid.uuid4())
            print(f"New session started: {thread_id}")
            print(f"Save this ID to continue later: {thread_id}\n")

        save_last_thread(thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        print("Agent ready. Type 'exit' to quit.\n")

        # Main conversation loop: runs until the user types "exit"
        while True:
            user_input = input("You: ")
            # Exit condition checked before any processing
            if user_input.lower() == "exit":
                break

            print("Agent: ", end="", flush=True)
            text, image_path = parse_user_input(user_input)

            # Attach a labeled file marker when input contains a file or media path
            if image_path:
                ext = os.path.splitext(image_path)[1].lower()
                # Label as "file" for documents so the router classifies them separately from images
                label = "file" if ext in DOCUMENT_EXTENSIONS else "image"
                message = HumanMessage(
                    content=f"{text} [{label} provided at path: {image_path}]"
                )
            # Plain text message with no attachment
            else:
                message = HumanMessage(content=text)

            _stream_agent(app, {"messages": [message]}, config)

            state = app.get_state(config)
            # Poll for pending graph interrupts that require human input before continuing
            while state.next:
                human_input = input("\nYour decision: ")
                # Resume the paused graph with the human decision and stream its continued output
                _stream_agent(app, Command(resume=human_input), config)
                state = app.get_state(config)

            print("\n")
