# agents/rag_qa.py

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

def get_financial_rag():
    # Load glossary file
    loader = TextLoader("documents/sec_glossary.txt")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_docs = splitter.split_documents(docs)

    # Embedding model
    embeddings = OllamaEmbeddings(model="codellama")

    # Pinecone init
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

    index_name = os.getenv("PINECONE_INDEX_NAME")

    # Check if index exists, otherwise create it
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=4096, metric="cosine")

    # Create vector store (loads into Pinecone)
    vectorstore = LangChainPinecone.from_documents(
        split_docs,
        embedding=embeddings,
        index_name=index_name
    )

    # Create retriever + chain
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    llm = Ollama(model="mistral")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain
