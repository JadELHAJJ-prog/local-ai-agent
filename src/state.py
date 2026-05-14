# state.py
from langgraph.graph.message import add_messages
from typing import Annotated, Optional
from typing_extensions import TypedDict


# Shared mutable state propagated through every node in the LangGraph graph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str
    retry_count: int
    is_valid: bool
    human_feedback: Optional[str]  # stores human's yes/no + comments
    code_generated: bool
    input_type: str
