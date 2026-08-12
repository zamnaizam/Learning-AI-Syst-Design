import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class StateAgent(TypedDict):
    msg: List[HumanMessage]

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

def process(state: StateAgent) -> StateAgent:
    response = llm.invoke(state["msg"])
    state['msg'].append(AIMessage(content=response.content))
    print(f"Response: {response.content}\n")
    return state

graph = StateGraph(StateAgent)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)


agent = graph.compile()

conversation_history = []


user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"msg": conversation_history})
    conversation_history = result["msg"]    # carry memory forward
    user_input = input("Enter: ")
