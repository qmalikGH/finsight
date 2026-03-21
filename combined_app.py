import ollama
import sqlite3
import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# --- Database ---
def create_sample_db():
    conn = sqlite3.connect("finsight_demo.db")
    pd.DataFrame({
        "date": ["2025-01-15", "2025-02-20", "2025-03-10", "2025-01-28"],
        "amount": [15000, 23000, 18500, 9200],
        "region": ["Nord", "Süd", "West", "Nord"],
        "cost_center": ["CC100", "CC200", "CC100", "CC300"]
    }).to_sql("financial_transactions", conn, if_exists="replace", index=False)
    return conn

SCHEMA = """
Tabelle: financial_transactions
Spalten: date (TEXT), amount (REAL), region (TEXT), cost_center (TEXT)

Beispieldaten:
- date Format: 'YYYY-MM-DD', Daten liegen im Jahr 2025
- Regionen: Nord, Süd, West
- Cost Centers: CC100, CC200, CC300
"""

# --- RAG ---
COLLECTION = "finsight_docs"

@st.cache_resource
def load_rag_resources():
    client = QdrantClient(path="./qdrant_db")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        docs = [
            "CC100 hatte im Januar eine Sonderbestellung von Kunde Müller GmbH über 15.000 EUR.",
            "Region Nord ist unser stärkster Markt, hauptsächlich durch Industriekunden.",
            "CC200 ist die Kostenstelle für Marketing, hohe Ausgaben im Februar durch Messebeteiligung.",
            "CC300 ist neu seit Januar 2025, noch im Aufbau mit geringem Umsatz.",
            "Umsatzeinbruch in Region West im März durch Lieferengpässe beim Hauptlieferanten.",
        ]
        vectors = embedder.encode(docs).tolist()
        points = [
            PointStruct(id=i, vector=vectors[i], payload={"text": docs[i]})
            for i in range(len(docs))
        ]
        client.upsert(collection_name=COLLECTION, points=points)

    return client, embedder

def search_context(question: str, client, embedder, top_k: int = 2):
    vector = embedder.encode(question).tolist()
    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k
    ).points
    return [r.payload["text"] for r in results]

# --- Combined pipeline ---
def analyze(question: str, client, embedder):
    context_docs = search_context(question, client, embedder)
    context_text = "\n".join(f"- {doc}" for doc in context_docs)

    conn = create_sample_db()
    response = ollama.chat(model="qwen2.5-coder:3b", messages=[{
        "role": "user",
        "content": (
            f"Schema:\n{SCHEMA}\n\n"
            f"Zusätzlicher Kontext:\n{context_text}\n\n"
            f"Frage: {question}\n\n"
            "Antworte NUR mit SQL, ohne Erklärung."
        )
    }])

    sql = response["message"]["content"].strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    result = pd.read_sql_query(sql, conn)
    return context_docs, sql, result

# --- UI ---
st.set_page_config(page_title="FinSight", page_icon="📊")
st.title("📊 FinSight")
st.caption("AI-powered Financial Data Assistant — RAG + Text-to-SQL")

client, embedder = load_rag_resources()

question = st.text_input(
    "Stelle eine Frage zu deinen Finanzdaten:",
    placeholder="z.B. Zeig mir den Umsatz pro Region"
)

if st.button("Analysieren"):
    if question:
        with st.spinner("Analysiere..."):
            context_docs, sql, result = analyze(question, client, embedder)

        st.subheader("Gefundener Kontext (RAG)")
        for doc in context_docs:
            st.info(doc)

        st.subheader("Generiertes SQL")
        st.code(sql, language="sql")

        st.subheader("Ergebnis")
        st.dataframe(result)
    else:
        st.warning("Bitte eine Frage eingeben.")
