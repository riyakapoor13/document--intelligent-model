# agents/synthesizer_agent.py

from state import AgentState
from .models import llm_openai_vision

def synthesizer_node(state: AgentState) -> dict:
    """ ✨ The Synthesizer (using GPT-4o) """
    print("---✨ SYNTHESIZING FINAL ANSWER (using GPT-4o)---")
    context = "\n\n".join(state['agent_outputs'])
    prompt = f"""
    You are the final reporting agent. Synthesize the information from the specialist agents
    into a single, coherent answer for the user's query.

    User Query: "{state['query']}"
    Collected Evidence:
    ---
    {context}
    ---
    Based on all evidence, provide the final answer.
    """
    response = llm_openai_vision.invoke(prompt)
    return {"final_answer": response.content}