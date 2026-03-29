from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import ollama
import sqlite3
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import logging
import datetime
import re

# ── Logging / Audit Trail ──────────────────────────────────────────────
logging.basicConfig(
    filename="audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(title="FinSight MCP Server")

# ── API Key Auth ───────────────────────────────────────────────────────
API_KEYS = {"finsight-secret-key-123"}  # später aus .env laden
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Depends(api_key_header)):
    if key not in API_KEYS:
        logging.warning(f"Unauthorized access attempt with key: {key}")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key

# ── Prompt Injection Filter ────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"forget everything",
    r"you are now",
    r"act as",
    r"jailbreak",
    r"system prompt",
    r"--",
    r"drop table",
    r"delete from",
    r"insert into",
]

def check_injection(text: str) -> bool:
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logging.warning(f"Injection attempt detected: {text}")
            return True
    return False

# ── DB & RAG Setup ─────────────────────────────────────────────────────
def get_db():
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
Daten im Jahr 2025. Regionen: Nord, Süd, West. Cost Centers: CC100, CC200, CC300.
"""

qdrant = QdrantClient(path="./qdrant_db")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

DOCS = [
    "CC100 hatte im Januar eine Sonderbestellung von Kunde Müller GmbH über 15.000 EUR.",
    "Region Nord ist unser stärkster Markt, hauptsächlich durch Industriekunden.",
    "CC200 ist die Kostenstelle für Marketing.",
    "CC300 ist neu seit Januar 2025, noch im Aufbau.",
    "Umsatzeinbruch in Region West im März durch Lieferengpässe.",
]

if not qdrant.collection_exists("finsight_docs"):
    qdrant.create_collection("finsight_docs", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    vectors = embedder.encode(DOCS).tolist()
    qdrant.upsert("finsight_docs", points=[PointStruct(id=i, vector=vectors[i], payload={"text": DOCS[i]}) for i in range(len(DOCS))])

# ── Request Model ──────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

# ── Main Endpoint ──────────────────────────────────────────────────────
@app.post("/query")
async def query(request: Request, body: QueryRequest, api_key: str = Depends(verify_api_key)):
    client_ip = request.client.host

    # 1. Injection Check
    if check_injection(body.question):
        logging.warning(f"BLOCKED | IP: {client_ip} | Query: {body.question}")
        raise HTTPException(status_code=400, detail="Query blocked: potential injection detected")

    # 2. Audit Log
    logging.info(f"QUERY | IP: {client_ip} | Question: {body.question}")

    # 3. RAG
    vector = embedder.encode(body.question).tolist()
    results = qdrant.query_points("finsight_docs", query=vector, limit=2).points
    context = "\n".join([r.payload["text"] for r in results])

    # 4. SQL generieren
    response = ollama.chat(model="qwen2.5-coder:3b", messages=[{
        "role": "user",
        "content": f"Schema:\n{SCHEMA}\nKontext:\n{context}\nFrage: {body.question}\nNur SQL, keine Erklärung."
    }])
    sql = response["message"]["content"].strip().replace("```sql", "").replace("```", "").strip()

    # 5. Query ausführen
    conn = get_db()
    try:
        result = pd.read_sql_query(sql, conn).to_dict(orient="records")
    except Exception as e:
        logging.error(f"SQL Error: {sql} | {e}")
        raise HTTPException(status_code=500, detail=f"SQL Error: {str(e)}")

    logging.info(f"SUCCESS | Rows returned: {len(result)}")
    return {"question": body.question, "sql": sql, "context": context, "result": result}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}