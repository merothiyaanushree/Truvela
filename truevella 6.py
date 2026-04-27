import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

@st.cache_resource
def initialize_rag_system():
    # Auto-generate a dummy legal file if it doesn't exist to ensure deployability
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "workplace_ethics.txt")
    
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("According to standard workplace ethics, harassment is strictly prohibited. Employees facing unfair termination should document all exchanges and contact HR immediately.")

    # Load and split documents
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Vector Embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # Use a local pipeline for text generation to keep it self-contained
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_length=256)
    local_llm = HuggingFacePipeline(pipeline=llm_pipeline)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever()
    )
    return qa_chain

def get_ethical_guidance(qa_chain, scenario):
    query = f"Based ONLY on the provided legal/ethical documents, how should one handle this scenario: '{scenario}'?"
    try:
        return qa_chain.run(query)
    except Exception as e:
        return "Guidance system is initializing. Please ensure you document your situation carefully."