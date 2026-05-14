from langgraph.graph.message import add_messages
from typing import Annotated, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str
    retry_count: int
    is_valid: bool
    human_feedback: Optional[str]
    approved: Optional[bool]
    input_type: str
