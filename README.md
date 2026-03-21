# FinSight 📊
> AI-powered Financial Data Assistant — natural language querying for structured financial data.

## Overview
FinSight is a locally-running AI system that allows users to query financial databases 
using natural language. Built as a foundation for exploring LLM-based data analysis, 
RAG pipelines, and Text-to-SQL architectures.

## Features
- 🔍 **Text-to-SQL** — converts natural language questions into SQL queries
- 🤖 **Local LLM** — runs fully offline via Ollama (no data leaves your machine)
- 📊 **Interactive UI** — Streamlit-based interface for querying and visualizing results
- 🔒 **Privacy-first** — no cloud dependency, suitable for sensitive financial data

## Tech Stack
- **LLM:** Qwen2.5-Coder 7B via Ollama
- **Framework:** Python, LangChain
- **Database:** SQLite (extensible to PostgreSQL, SAP/DWH)
- **UI:** Streamlit
- **Data:** Pandas, SQLAlchemy

## Roadmap
- [x] Text-to-SQL pipeline
- [x] Local LLM integration (Ollama)
- [x] Streamlit UI
- [ ] RAG pipeline with Qdrant vector database
- [ ] Schema-aware context injection
- [ ] Anomaly detection layer
- [ ] LoRA fine-tuning on financial domain data
- [ ] FastAPI REST endpoint
- [ ] Docker containerization
- [ ] MLflow experiment tracking

## Getting Started
```bash
# Clone the repo
git clone https://github.com/qmalikGH/finsight.git
cd finsight

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull the model
ollama pull qwen2.5-coder:7b

# Run the app
streamlit run app.py
```

## Example Queries
- *"Show me total revenue by region"*
- *"Which cost center has the highest expenses?"*
- *"Show all transactions from January"*