# agent.py
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from state import AgentState

# 1. initialize the LLM
llm = ChatOllama(
    model="llama3.2",
    num_ctx=8192,
    temperature=0.1,
    num_predict=1024,
    top_p=0.9,
)


# 2. define the agent node
def agent_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 3. build the graph
graph = StateGraph(AgentState)
graph.add_node("agent_node", agent_node)
graph.set_entry_point("agent_node")
graph.add_edge("agent_node", END)

# 4. compile
app = graph.compile()

# 5. test it
if __name__ == "__main__":
    result = app.invoke(
        {"messages": [("user", "hello, who are you?")], "thread_id": "test-001"}
    )
    print(result["messages"][-1].content)
