# Agentic-AI-Budget-System

A full-stack, multi-agent AI financial assistant system that validates financial statements, explains financial terms, and sends alerts using Python, LangChain, LangGraph, Ollama, ChromaDB, Streamlit, and SMTP.

---

## 🚀 Features

- **Upload & Validate Financial Statements (CSV)**: Checks for missing values, negative income, liabilities > assets, and more.
- **RAG QA Agent**: Ask natural language questions about financial terms using a local LLM (Ollama) and a custom glossary.
- **Automated Email Alerts**: Sends notifications if validation fails.
- **LangGraph MCP**: Orchestrates agents in a pipeline (`validate → rag → notify`).
- **Streamlit UI**: Easy-to-use frontend for upload, validation, Q&A, and pipeline execution.

---

## 📁 Project Structure

```
Agentic-AI-Budget-System/
├── agents/
│   ├── validator.py         # Validator agent
│   ├── rag_qa.py            # RAG QA agent
│   └── notifier_email.py    # Email notifier agent
├── mcp/
│   └── controller.py        # LangGraph MCP controller
├── datasets/
│   └── financial_statements.csv  # Sample input CSV
├── documents/
│   └── sec_glossary.txt     # Financial glossary (for RAG)
├── .env                     # Email credentials
├── requirements.txt         # Python dependencies
├── app.py                   # Streamlit frontend
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/geetikavasistha-01/Agentic-AI-Budget-System.git
   cd Agentic-AI-Budget-System
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and run Ollama (for local LLM):**
   - [Install Ollama](https://ollama.com/download)
   - Pull a model (e.g., mistral):
     ```bash
     ollama pull mistral
     ollama serve &
     ```

5. **Set up your `.env` file:**
   Create a `.env` file in the project root with:
   ```env
   ALERT_SENDER_EMAIL=youremail@gmail.com
   ALERT_RECEIVER_EMAIL=cfo@company.com
   ALERT_EMAIL_PASSWORD=your_app_password_here
   ```
   > Use an [App Password](https://support.google.com/accounts/answer/185833?hl=en) for Gmail.

---

## 🏗️ Architecture & Agents

- **Validator Agent (`agents/validator.py`)**
  - Validates uploaded financial CSVs for missing/invalid data.
- **RAG QA Agent (`agents/rag_qa.py`)**
  - Answers questions using Retrieval-Augmented Generation over `documents/sec_glossary.txt` and a local LLM.
- **Notifier Agent (`agents/notifier_email.py`)**
  - Sends email alerts using SMTP and credentials from `.env`.
- **LangGraph MCP (`mcp/controller.py`)**
  - Orchestrates the pipeline: validate → rag → notify.
- **Streamlit Frontend (`app.py`)**
  - Upload CSV, preview data, run validation, ask questions, and run the full pipeline.

---

## 🖥️ Usage

### 1. **Start Ollama server** (if not already running):
```bash
ollama serve &
```

### 2. **Run the Streamlit app:**
```bash
streamlit run app.py
```

### 3. **Web Interface Workflow:**
- **Upload CSV:** Upload a financial statement file (columns: Tag, Value).
- **Preview Data:** See your data in a table.
- **Validate:** Results are shown, and an email is sent if problems are found.
- **Ask a Question:** Enter a financial term/question for the RAG agent.
- **Full Pipeline:** (Optional) Run validation → QA → notification in sequence.

### 4. **Command-line (for MCP):**
You can also run the orchestrator directly:
```bash
python mcp/controller.py
```

---

## 📝 Sample Data

**Sample CSV (`datasets/financial_statements.csv`):**
```
Tag,Value
us-gaap:Assets,1000000
us-gaap:Liabilities,1200000
us-gaap:NetIncomeLoss,-50000
```

**Sample Glossary (`documents/sec_glossary.txt`):**
```
us-gaap:NetIncomeLoss = Net income is the profit or loss after all expenses and taxes.
```

---

## 🛠️ Development & Customization
- Agents are modular Python files in `agents/`.
- Add new validation rules or glossary terms as needed.
- Integrate new LLMs by updating `rag_qa.py` and `controller.py`.
- Logging is included for debugging and monitoring.

---

## 📨 Email Troubleshooting
- Use an App Password for Gmail (not your main password).
- Check your spam folder for alerts.
- Ensure `.env` is filled and not committed to GitHub.

---

## 💾 Commit Message Example

```
Initial multi-agent AI financial assistant: validator, RAG, notifier, MCP, Streamlit UI
```

---

## 📚 References
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://docs.trychroma.com/)
- [Streamlit](https://streamlit.io/)

---

## © 2025 geetikavasistha-01