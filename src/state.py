# state.py
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str
    retry_count: int
    is_valid: bool