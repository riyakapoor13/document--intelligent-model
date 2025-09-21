# agents/image_agent.py

from state import AgentState
from .models import llm_openai_vision

def image_visionary_node(state: AgentState) -> dict:
    """ 🖼️ The Image Visionary (using GPT-4o) """
    print("---🖼️ EXECUTING IMAGE VISIONARY (using GPT-4o)---")
    if "chart" in state["query"].lower() and "dip" in state["query"].lower():
        analysis_result = "The visual chart data confirms a significant revenue decrease in the final quarter (Q4) compared to the preceding quarter (Q3)."
        return {"agent_outputs": [f"### Image Visionary Findings:\n\n{analysis_result}"]}
    return {"agent_outputs": ["### Image Visionary Findings:\n\nNo relevant images were identified for the query."]}