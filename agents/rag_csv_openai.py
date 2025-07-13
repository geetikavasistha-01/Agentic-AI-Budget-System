import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.schema import Document
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
import logging

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-west-2")

logging.basicConfig(level=logging.INFO)


def csv_to_documents(csv_path: str) -> List[Document]:
    """
    Reads a CSV and converts each row into a LangChain Document with metadata.
    """
    df = pd.read_csv(csv_path)
    docs = []
    for idx, row in df.iterrows():
        content = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "row": idx,
                    "source": os.path.basename(csv_path)
                }
            )
        )
    return docs


def build_vectorstore(docs: List[Document], namespace: str = "finance-docs"):
    """
    Embeds docs and stores them in Pinecone under the given namespace.
    Returns a LangChain vectorstore retriever.
    """
    # Initialize Pinecone v3+ API
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="page_content",
        namespace=namespace,
    )
    # Add documents
    vectorstore.add_documents(docs)
    return vectorstore.as_retriever()


def get_financial_rag_openai(csv_path: str, namespace: str = "finance-docs") -> RetrievalQA:
    """
    Returns a RetrievalQA chain for the uploaded CSV using Pinecone and OpenAI GPT-4.
    Args:
        csv_path (str): Path to the CSV file.
        namespace (str): Pinecone namespace to use (default: "finance-docs").
    Returns:
        RetrievalQA: A LangChain RetrievalQA chain ready to answer questions.
    """
    docs = csv_to_documents(csv_path)
    retriever = build_vectorstore(docs, namespace=namespace)
    llm = OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa


def test_rag(csv_path: str, question: str):
    """
    Test block to load CSV and answer a sample question.
    """
    print(f"Loading CSV: {csv_path}")
    qa = get_financial_rag_openai(csv_path)
    print(f"Question: {question}")
    result = qa({"query": question})
    print("Answer:", result["result"])
    print("Source Document(s):", result["source_documents"])


if __name__ == "__main__":
    # Example usage
    sample_csv = "datasets/financial_statements.csv"
    test_question = "What is the net income for 2023?"
    test_rag(sample_csv, test_question)
