# main.py

import os

from cerebrum_ai.chains.rag_pipeline import create_rag_chain

print("🔍 OPENAI_API_KEY found:", os.getenv("OPENAI_API_KEY"))

api_key = os.getenv("OPENAI_API_KEY")
knowledge_base_dir = "data/knowledge_base"

rag_chain = create_rag_chain(api_key=api_key, knowledge_base_dir=knowledge_base_dir)

print("✅ OPENAI API key loaded.")
print("🚀 CerebrumAI: RAG system launching...")

query = input("🧠 Ask a question: ")

# Use invoke() instead of deprecated __call__()
result = rag_chain.invoke(query)

print("\n💬 Answer:")
print(result['result'])

print("\n📚 Source Documents:")
for doc in result['source_documents']:
    print(f"- {doc.metadata['source']}")