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

if __name__ == "__main__":
    print("Agent ready. Type 'exit' to quit.\n")
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        history.append(("user", user_input))
        result = app.invoke({"messages": history, "thread_id": "test-001"})
        history = result["messages"]  # update history with full state
        print(f"Agent: {result['messages'][-1].content}\n")
