"""
RAG QA Agent for Financial Glossary using LangChain, Chroma, and Ollama

- Loads sec_glossary.txt from documents/
- Uses Chroma as the vector DB
- Uses OllamaEmbeddings for vector encoding
- Uses Ollama LLM (e.g., mistral or llama3)
- Returns a RetrievalQA chain for querying
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from typing import Optional

def get_financial_rag(
    glossary_path: Optional[str] = None,
    ollama_model: str = "mistral",
    persist_directory: str = ".chroma_sec_glossary"
) -> RetrievalQA:
    """
    Loads the SEC glossary, builds a Chroma vector DB, and returns a RetrievalQA chain using Ollama LLM.
    Args:
        glossary_path (str, optional): Path to sec_glossary.txt. Defaults to documents/sec_glossary.txt.
        ollama_model (str): Name of the Ollama model to use (e.g., 'mistral', 'llama3').
        persist_directory (str): Directory for Chroma persistence.
    Returns:
        RetrievalQA: LangChain RetrievalQA chain ready for .run() queries.
    """
    if glossary_path is None:
        glossary_path = os.path.join(os.path.dirname(__file__), '../documents/sec_glossary.txt')

    # Load glossary lines
    try:
        with open(glossary_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        raise RuntimeError(f"Error loading glossary: {e}")

    # Parse glossary into LangChain Documents
    docs = []
    for line in lines:
        if '=' in line:
            tag, desc = line.split('=', 1)
            docs.append(Document(page_content=desc.strip(), metadata={"tag": tag.strip()}))
    if not docs:
        raise ValueError("Glossary is empty or malformed.")

    # Split text if needed (not strictly necessary for short glossary entries)
    splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = splitter.split_documents(docs)

    # Set up embeddings and vector store
    embeddings = OllamaEmbeddings(model=ollama_model)
    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)

    # Set up retriever and LLM
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    llm = Ollama(model=ollama_model)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa
