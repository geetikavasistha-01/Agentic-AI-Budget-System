import os
import sys
# --- Fix import path so this works with both "python mcp/controller.py" and "python -m mcp.controller" ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langgraph.graph import StateGraph
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama
from agents.validator import validate_financial_csv
from agents.rag_qa import get_financial_rag
from agents.notifier_email import send_notification_email


def build_agent_graph():
    """
    Builds a LangGraph MCP orchestrating:
    1. Validation agent
    2. RAG QA agent
    3. Notifier agent
    in sequence.
    Returns a compiled workflow graph.
    """
    llm = Ollama(model="mistral")

    # Wrap validator agent
    validator_agent = initialize_agent(
        tools=[validate_financial_csv],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Wrap RAG QA agent
    rag_qa_agent = get_financial_rag()
    notifier_agent = send_notification_email

    # Build LangGraph
    graph = StateGraph()
    graph.add_node("validate", validator_agent)
    graph.add_node("qa", rag_qa_agent)
    graph.add_node("notify", notifier_agent)

    graph.set_entry_point("validate")
    graph.set_successor("validate", "qa")
    graph.set_successor("qa", "notify")

    return graph.compile()
