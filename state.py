# state.py

import operator
from typing import TypedDict, Annotated, List

class AgentState(TypedDict):
    """
    Represents the shared memory of the multi-agent system.
    """
    query: str
    document_path: str
    route: str
    agent_outputs: Annotated[List[str], operator.add]
    final_answer: str