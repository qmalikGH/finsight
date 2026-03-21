from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import ollama

# Lokale Qdrant Datenbank & Embedding Modell
client = QdrantClient(path="./qdrant_db")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION = "finsight_docs"

# Einmalig: Wissensbasis aufbauen
def setup_knowledge_base():
    # Sammlung anlegen
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
    print(f"✅ {len(docs)} Dokumente gespeichert")

# Relevante Docs zur Frage finden
def search(question: str, top_k: int = 2):
    vector = embedder.encode(question).tolist()
    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k
    ).points
    return [r.payload["text"] for r in results]

# RAG-Frage stellen
def rag_ask(question: str):
    context = search(question)
    context_text = "\n".join(context)

    print(f"📚 Gefundener Kontext:\n{context_text}\n")

    response = ollama.chat(model="qwen2.5-coder:3b", messages=[{
        "role": "user",
        "content": f"""Du bist ein Finanzanalyst-Assistent.

Kontext aus der Wissensbasis:
{context_text}

Frage: {question}

Antworte kurz und präzise auf Deutsch."""
    }])

    return response["message"]["content"]

# Test
setup_knowledge_base()
print("\n" + rag_ask("Was ist mit Region West passiert?"))
