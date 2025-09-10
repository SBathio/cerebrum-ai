# rag_pipeline.py

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def build_rag_chain(api_key: str):
    loader = TextLoader("data/knowledge_base/customer_policy.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return rag_chain