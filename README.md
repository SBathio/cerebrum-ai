# 🧠 CerebrumAI

CerebrumAI is an enterprise-ready, modular GenAI platform that combines the power of LLMs, multi-agent systems, and Retrieval-Augmented Generation (RAG) to automate decision-making, document intelligence, and reasoning at scale.

## ✨ Features

- Multi-agent workflows (LangGraph, CrewAI)
- LLM + tool use + SQL hybrid intelligence
- Prompt engineering + reasoning patterns (ReAct, CoT, ToT)
- Vector database integration (FAISS, Pinecone)
- API connectors, file ingestion, custom tools
- Cloud-native, scalable design with CI/CD

## 📁 Project Structure
cerebrum-ai/
├── agents/          # Agent logic (planner, retriever, executor, critic)
├── chains/          # Prompt + RAG chains
├── tools/           # APIs, SQL runner, calculators
├── data/            # Source knowledge (PDFs, CSVs)
├── ui/              # Streamlit UI
├── infra/           # Docker, deployment scripts
├── notebooks/       # Experiments and Jupyter workflows
├── logs/            # Traces and debug logs
├── tests/           # Pytest test suite
├── main.py          # App entry point
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
