import ollama
import sqlite3
import pandas as pd
import streamlit as st

# Datenbank
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

def ask(question: str):
    conn = create_sample_db()
    response = ollama.chat(model="qwen2.5-coder:3b", messages=[{
        "role": "user",
        "content": f"Schema:\n{SCHEMA}\n\nFrage: {question}\n\nAntworte NUR mit SQL, ohne Erklärung."
    }])
    sql = response["message"]["content"].strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    result = pd.read_sql_query(sql, conn)
    return sql, result

# UI
st.set_page_config(page_title="FinSight", page_icon="📊")
st.title("📊 FinSight")
st.caption("AI-powered Financial Data Assistant")

question = st.text_input("Stelle eine Frage zu deinen Finanzdaten:", placeholder="z.B. Zeig mir den Umsatz pro Region")

if st.button("Analysieren"):
    if question:
        with st.spinner("Analysiere..."):
            sql, result = ask(question)
        
        st.subheader("Generiertes SQL")
        st.code(sql, language="sql")
        
        st.subheader("Ergebnis")
        st.dataframe(result)
    else:
        st.warning("Bitte eine Frage eingeben.")