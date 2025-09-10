# rag_pipeline.py

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
from pathlib import Path
import glob


def load_documents_from_directory(directory):
    documents = []
    for filepath in glob.glob(f"{directory}/*"):
        if filepath.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filepath.endswith(".txt"):
            loader = TextLoader(filepath)
        elif filepath.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        else:
            print(f"Unsupported file type: {filepath}")
            continue
        docs = loader.load()
        documents.extend(docs)
    return documents


def build_or_load_vectorstore(documents, embeddings, persist_path="vectorstore"):
    if os.path.exists(persist_path):
        print(f"📂 Loading existing vectorstore from {persist_path}")
        return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"📁 Creating and saving new vectorstore to {persist_path}")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(persist_path)
        return vectorstore


def create_rag_chain(api_key, knowledge_base_dir):
    documents = load_documents_from_directory(knowledge_base_dir)
    if not documents:
        raise ValueError("❌ No supported documents found in knowledge base directory.")

    # ✅ Split documents
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(documents)

    # ✅ Instantiate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # ✅ Build vectorstore
    vectorstore = build_or_load_vectorstore(documents, embeddings)

    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)

    prompt_template = PromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the user's question.
    
    Context:
    {context}
    
    Question:
    {question}
    """)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    return qa_chain