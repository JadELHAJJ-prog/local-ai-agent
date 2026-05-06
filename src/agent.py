# agent.py
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from state import AgentState
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

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
    system_prompt = SystemMessage(
        content="You are a helpful assistant named Bubbles. Answer the user's question based on the conversation history. You can search the web, execute code, do computer vision tasks, and more to help answer the user's question. Be kind and concise in your responses."
    )
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# 3. build the graph
graph = StateGraph(AgentState)
graph.add_node("agent_node", agent_node)
graph.set_entry_point("agent_node")
graph.add_edge("agent_node", END)

if __name__ == "__main__":
    with SqliteSaver.from_conn_string("memory.db") as memory:
        app = graph.compile(checkpointer=memory)
        print("Agent ready. Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": "test-001"}},
            )
            print(f"Agent: {result['messages'][-1].content}\n")
