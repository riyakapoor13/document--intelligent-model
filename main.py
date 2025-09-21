import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import AgentState
from agents import (
    query_router_node,
    text_analyst_node,
    table_interpreter_node,
    image_visionary_node,
    synthesizer_node
)

# Load API keys from .env file
load_dotenv()
if not os.getenv("OPENAI_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("API keys for OpenAI and Google must be set in the .env file.")

# --- Define the Graph ---
workflow = StateGraph(AgentState)

# --- Add Nodes ---
workflow.add_node("router", query_router_node)
workflow.add_node("text_analyst", text_analyst_node)
workflow.add_node("table_interpreter", table_interpreter_node)
workflow.add_node("image_visionary", image_visionary_node)
workflow.add_node("synthesizer", synthesizer_node)

# --- Define Edges with Conditional Routing ---
workflow.set_entry_point("router")

def decide_next_node(state: AgentState):
    print(f"---DECISION: Routing to '{state['route']}'---")
    return state['route']

workflow.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "image_analysis": "image_visionary",
        "table_query": "table_interpreter",
        "general_analysis": "text_analyst",
    }
)

# All specialist nodes lead to the synthesizer
workflow.add_edge("image_visionary", "synthesizer")
workflow.add_edge("table_interpreter", "synthesizer")
workflow.add_edge("text_analyst", "synthesizer")
workflow.add_edge("synthesizer", END)

# --- Compile and Run ---
app = workflow.compile()
print("Synapse Multi-Agent AI Graph with LLM Router Compiled Successfully!")
print("-" * 50)

# Create a sample document for the PoC
with open("sample_document.txt", "w") as f:
    f.write("Financial Report Q4 Analysis.\n\n")
    f.write("The fourth quarter experienced unforeseen challenges. While Q3 revenue was strong at $3.1M, ")
    f.write("supply chain disruptions led to a significant dip in Q4.\n\n")
    f.write("We are confident that our strategy will lead to recovery in the next fiscal year.")

inputs = {
    "query": "Based on the growth chart and financial table, what reason does the text give for the Q4 revenue dip?",
    "document_path": "sample_document.txt"
}

print(f"Invoking the Synapse graph with the query: '{inputs['query']}'\n")

for output in app.stream(inputs):
    for key, value in output.items():
        print(f"## Output from Node: '{key}'")
        print("---")
        print(value)
        print("---\n")

final_state = app.invoke(inputs)
print("=" * 50)
print("✅ FINAL COMPREHENSIVE ANSWER:")
print("=" * 50)
print(final_state['final_answer'])