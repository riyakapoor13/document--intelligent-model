# agents/router_agent.py

from state import AgentState
from .models import llm_router # Import from our new models file

def query_router_node(state: AgentState) -> dict:
    """ 🧠 The Query Router """
    print("---🧠 EXECUTING QUERY ROUTER---")
    prompt = f"""
    You are an expert at routing a user's query to the correct specialist pathway.
    Based on the query, classify it into ONE of the following categories.
    Return only the category name and nothing else.

    - image_analysis: The query involves analyzing a chart, figure, or image.
    - table_query: The query is about specific data points, numbers, or stats likely in a table.
    - general_analysis: The query is a general question, requires summarization, or text analysis.

    User Query: "{state['query']}"
    Category:
    """
    response = llm_router.invoke(prompt)
    route = response.content.strip()
    print(f"---Routing decision: '{route}'---")
    return {"route": route}