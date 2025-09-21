# agents/table_agent.py

import pandas as pd
import io
from contextlib import redirect_stdout
from unstructured.partition.auto import partition
from unstructured.documents.elements import Table
from state import AgentState
from .models import llm_gemini_pro

def table_interpreter_node(state: AgentState) -> dict:
    """ 📊 The Table Interpreter (using Gemini) """
    print("---📊 EXECUTING TABLE INTERPRETER (using Gemini Pro)---")
    try:
        elements = partition(state["document_path"])
        table_elements = [el for el in elements if isinstance(el, Table)]
    except Exception as e:
        return {"agent_outputs": [f"### Table Interpreter Error: Failed to partition document: {e}"]}

    if not table_elements:
        return {"agent_outputs": ["### Table Interpreter Findings: No tables were found."]}

    try:
        html_table = table_elements[0].metadata.text_as_html
        df_list = pd.read_html(io.StringIO(html_table))
        if not df_list:
            return {"agent_outputs": ["### Table Interpreter Findings: Could not parse a DataFrame."]}
        df = df_list[0]
    except Exception as e:
        return {"agent_outputs": [f"### Table Interpreter Error: Failed to convert table to DataFrame: {e}"]}

    prompt = f"""
    You are a Python data analyst. Given a pandas DataFrame `df`, write a script to answer the query. Print the final result.
    DataFrame: df = pd.DataFrame({df.to_dict()})
    User Query: "{state['query']}"
    Your Python script (print the result):
    """
    response = llm_gemini_pro.invoke(prompt)
    python_code = response.content.strip('`').strip('python\n')
    
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            exec(python_code, {'pd': pd, 'df': df})
        output = f.getvalue()
        return {"agent_outputs": [f"### Table Interpreter Findings:\n\n{output}"]}
    except Exception as e:
        return {"agent_outputs": [f"### Table Interpreter Error: Executing code failed: {e}"]}