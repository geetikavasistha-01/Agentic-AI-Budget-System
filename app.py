import streamlit as st
import pandas as pd
import os
import traceback

from agents.validator import validate_financial_csv
from agents.rag_qa import get_financial_rag
from agents.notifier_email import send_notification_email
from mcp.controller import build_agent_graph

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Agentic AI Financial Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🤖 Multi-Agent AI Financial Assistant")

# --- Section: CSV Upload ---
st.header("1. Upload Financial Statement CSV")
uploaded_file = st.file_uploader(
    "Upload a CSV file (columns: Tag, Value)",
    type=["csv"],
    key="csv_uploader"
)
dataset_path = "datasets/financial_statements.csv"

if uploaded_file:
    # Save uploaded file to datasets/
    os.makedirs("datasets", exist_ok=True)
    with open(dataset_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preview CSV
    try:
        df = pd.read_csv(dataset_path)
        st.subheader("CSV Preview")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # --- Section: Validation ---
    st.header("2. Validation Results")
    try:
        validation_results = validate_financial_csv(dataset_path)
        # If agent returns a list, join for display
        if isinstance(validation_results, list):
            validation_message = "\n".join(validation_results)
        else:
            validation_message = str(validation_results)
        if "❌" in validation_message or "⚠️" in validation_message:
            st.warning(validation_message)
            # --- Section: Email Alert ---
            st.info("Sending alert email...")
            email_status = send_notification_email(validation_message)
            st.write(email_status)
        else:
            st.success(validation_message)
    except Exception as e:
        st.error(f"Validator agent error: {e}\n{traceback.format_exc()}")

else:
    st.info("Please upload a CSV file to begin.")

# --- Section: Ask a Financial Question (RAG) ---
st.header("3. Ask a Financial Question (RAG)")
question = st.text_input(
    "Ask about a financial term (e.g., 'What is us-gaap:NetIncomeLoss?')",
    key="rag_question"
)

if question:
    try:
        rag_qa_agent = get_financial_rag()
        rag_answer = rag_qa_agent.run(question)
        st.success(rag_answer)
    except Exception as e:
        st.error(f"RAG QA agent error: {e}\n{traceback.format_exc()}")

# --- Section: (Optional) Run Full MCP Pipeline ---
st.header("4. (Optional) Run Full MCP Pipeline")
if st.button("Run Full Agent Pipeline (Validate → QA → Notify)"):
    try:
        graph = build_agent_graph()
        # Example input: file path and a sample question
        agent_input = {
            "csv_path": dataset_path,
            "question": question or "What is us-gaap:NetIncomeLoss?"
        }
        result = graph.run(agent_input)
        st.code(str(result), language="python")
    except Exception as e:
        st.error(f"MCP pipeline error: {e}\n{traceback.format_exc()}")

# --- Footer ---
st.markdown("---")
st.caption("Built with LangChain, LangGraph, Ollama, ChromaDB, and Streamlit. © 2025")
