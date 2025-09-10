# ui.py

import streamlit as st
from chains.rag_pipeline import create_rag_chain
import os

# Set up page config and header
st.set_page_config(page_title="CerebrumAI", page_icon="🧠")
st.markdown(
    "<h1 style='text-align: center;'>🧠 CerebrumAI: Retrieval-<br>Augmented Generation</h1>",
    unsafe_allow_html=True
)

# Load API key and knowledge base directory
api_key = os.getenv("OPENAI_API_KEY")
knowledge_base_dir = "data/knowledge_base"

# Load RAG chain (cached to avoid reloading on every query)
@st.cache_resource
def load_chain():
    return create_rag_chain(api_key, knowledge_base_dir)

rag_chain = load_chain()

# Input box
query = st.text_input("Ask your question:")

# Submit button logic
if st.button("Submit") and query:
    st.success(f"You asked: {query}")
    with st.spinner("🔎 Retrieving and generating answer..."):
        try:
            result = rag_chain.invoke(query)
            print("🧠 Full result:", result)  # ✅ Debug print

            # Extract and show the answer
            answer = result.get("result")
            if answer:
                st.markdown("### 💬 Answer")
                st.markdown(f"**{answer}**")
            else:
                st.warning("⚠️ No answer was returned.")

            # Show source documents if available
            sources = result.get("source_documents", [])
            if sources:
                st.markdown("### 📚 Source Documents")
                for doc in sources:
                    source = doc.metadata.get("source", "Unknown")
                    st.markdown(f"- `{source}`")
            else:
                st.info("ℹ️ No source documents found.")

        except Exception as e:
            st.error(f"❌ Error: {e}")