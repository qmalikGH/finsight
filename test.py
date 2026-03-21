import ollama
import sqlite3
import pandas as pd

# Testdatenbank erstellen
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
    
    # LLM generiert SQL
    response = ollama.chat(model="qwen2.5-coder:3b", messages=[{
        "role": "user",
        "content": f"Schema:\n{SCHEMA}\n\nFrage: {question}\n\nAntworte NUR mit SQL, ohne Erklärung."
    }])
    
    sql = response["message"]["content"].strip()
    # Code-Blöcke entfernen falls das Modell sie hinzufügt
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    print(f"Generiertes SQL:\n{sql}\n")
    
    result = pd.read_sql_query(sql, conn)
    print("Ergebnis:")
    print(result)
    return result

# Test
ask("Welcher Cost Center hat den höchsten Umsatz?")
ask("Zeig mir alle Transaktionen im Januar")