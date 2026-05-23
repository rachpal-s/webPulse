"""
Run this once to add missing columns to your existing webpulse.db
Usage: python migrate_db.py
"""
import sqlite3
from pathlib import Path

DB = Path(__file__).parent / "data" / "webpulse.db"

if not DB.exists():
    print(f"DB not found at {DB}")
    exit(1)

conn = sqlite3.connect(DB)

migrations = [
    ("url_history",  "ALTER TABLE url_history ADD COLUMN is_daily INTEGER DEFAULT 0"),
    ("url_history",  "ALTER TABLE url_history ADD COLUMN last_briefed TEXT DEFAULT ''"),
    ("portfolio",    """CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL, name TEXT NOT NULL,
        qty REAL DEFAULT 0, avg_price REAL DEFAULT 0,
        sector TEXT DEFAULT '', exchange TEXT DEFAULT 'NSE',
        notes TEXT DEFAULT '', active INTEGER DEFAULT 1,
        created_at REAL, updated_at REAL
    )"""),
    ("morning_briefs", """CREATE TABLE IF NOT EXISTS morning_briefs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        brief_date TEXT NOT NULL UNIQUE,
        status TEXT DEFAULT 'pending',
        started_at REAL, completed_at REAL,
        rag_session_id TEXT DEFAULT '',
        articles_scraped INTEGER DEFAULT 0,
        html_content TEXT DEFAULT '',
        error_msg TEXT DEFAULT ''
    )"""),
    ("brief_insights", """CREATE TABLE IF NOT EXISTS brief_insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        brief_date TEXT NOT NULL,
        prompt_key TEXT NOT NULL,
        prompt_text TEXT NOT NULL,
        answer_html TEXT DEFAULT '',
        generated_at REAL,
        sources_json TEXT DEFAULT '{}',
        UNIQUE(brief_date, prompt_key)
    )"""),
    ("brief_insights.sources_json",
     "ALTER TABLE brief_insights ADD COLUMN sources_json TEXT DEFAULT '{}'"),
]

for label, sql in migrations:
    try:
        conn.execute(sql)
        conn.commit()
        print(f"OK  {label}: {sql[:55]}…")
    except Exception as e:
        print(f"--  {label}: already exists ({e})")

print("\nDone. Restart the app.")
conn.close()