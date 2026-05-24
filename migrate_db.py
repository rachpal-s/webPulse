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
    ("brief_categories.description",
     "ALTER TABLE brief_categories ADD COLUMN description TEXT DEFAULT ''"),
    ("brief_insights.sources_json",
     "ALTER TABLE brief_insights ADD COLUMN sources_json TEXT DEFAULT '{}'"),
    ("brief_insights.brief_id",
     "ALTER TABLE brief_insights ADD COLUMN brief_id INTEGER DEFAULT NULL"),
    ("url_history.category_id",
     "ALTER TABLE url_history ADD COLUMN category_id INTEGER DEFAULT NULL"),
    ("morning_briefs.category_id",
     "ALTER TABLE morning_briefs ADD COLUMN category_id INTEGER DEFAULT 1"),
    ("morning_briefs.category_name",
     "ALTER TABLE morning_briefs ADD COLUMN category_name TEXT DEFAULT 'General'"),
    ("brief_categories", """CREATE TABLE IF NOT EXISTS brief_categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        icon TEXT DEFAULT '📰',
        color TEXT DEFAULT '#22d3ee',
        is_builtin INTEGER DEFAULT 0,
        auto_run INTEGER DEFAULT 0,
        created_at REAL
    )"""),
    ("category_prompts", """CREATE TABLE IF NOT EXISTS category_prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_id INTEGER NOT NULL,
        prompt_key TEXT NOT NULL,
        label TEXT NOT NULL,
        prompt_text TEXT NOT NULL,
        UNIQUE(category_id, prompt_key)
    )"""),
]

for label, sql in migrations:
    try:
        conn.execute(sql)
        conn.commit()
        print(f"OK  {label}: {sql[:55]}…")
    except Exception as e:
        print(f"--  {label}: already exists ({e})")

# ── Recreate morning_briefs with correct UNIQUE constraint ───────────────────
print()
print("Checking morning_briefs UNIQUE constraint...")
schema = conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='morning_briefs'"
).fetchone()
if schema and "UNIQUE(brief_date, category_id)" not in schema[0]:
    print("  Recreating morning_briefs table...")
    conn.executescript("""
        ALTER TABLE morning_briefs RENAME TO morning_briefs_old;
        CREATE TABLE morning_briefs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brief_date TEXT NOT NULL,
            category_id INTEGER DEFAULT 1,
            category_name TEXT DEFAULT 'General',
            status TEXT DEFAULT 'pending',
            started_at REAL, completed_at REAL,
            rag_session_id TEXT DEFAULT '',
            articles_scraped INTEGER DEFAULT 0,
            html_content TEXT DEFAULT '',
            error_msg TEXT DEFAULT '',
            UNIQUE(brief_date, category_id)
        );
        INSERT OR IGNORE INTO morning_briefs
            (id,brief_date,category_id,category_name,status,
             started_at,completed_at,rag_session_id,
             articles_scraped,html_content,error_msg)
        SELECT id,brief_date,category_id,category_name,status,
               started_at,completed_at,rag_session_id,
               articles_scraped,html_content,error_msg
        FROM morning_briefs_old;
        DROP TABLE morning_briefs_old;
    """)
    print("  OK  morning_briefs recreated with UNIQUE(brief_date, category_id)")
else:
    print("  OK  constraint already correct")

# ── Recreate brief_insights with correct UNIQUE constraint ───────────────────
print()
print("Checking brief_insights UNIQUE constraint...")
schema = conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='brief_insights'"
).fetchone()
if schema and 'UNIQUE(brief_date, brief_id, prompt_key)' not in schema[0]:
    print("  Recreating brief_insights table...")
    conn.executescript("""
        ALTER TABLE brief_insights RENAME TO brief_insights_old;
        CREATE TABLE brief_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brief_date TEXT NOT NULL,
            brief_id INTEGER,
            prompt_key TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            answer_html TEXT DEFAULT '',
            generated_at REAL,
            sources_json TEXT DEFAULT '{}',
            UNIQUE(brief_date, brief_id, prompt_key)
        );
        INSERT OR IGNORE INTO brief_insights
            (id, brief_date, brief_id, prompt_key, prompt_text,
             answer_html, generated_at, sources_json)
        SELECT id, brief_date, brief_id, prompt_key, prompt_text,
               answer_html, generated_at, COALESCE(sources_json, '{}')
        FROM brief_insights_old;
        DROP TABLE brief_insights_old;
    """)
    print("  OK  brief_insights recreated with UNIQUE(brief_date, brief_id, prompt_key)")
else:
    print("  OK  constraint already correct")

# Clean up orphaned insights with NULL brief_id
orphans = conn.execute(
    "DELETE FROM brief_insights WHERE brief_id IS NULL"
).rowcount
if orphans:
    conn.commit()
    print(f"Cleaned {orphans} orphaned insights (brief_id=NULL)")

# ── Create url_queue table ───────────────────────────────────────────────────
print()
print("Creating url_queue table...")
conn.executescript("""
    CREATE TABLE IF NOT EXISTS url_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        category_id INTEGER NOT NULL,
        source_url TEXT DEFAULT '',
        title TEXT DEFAULT '',
        summary TEXT DEFAULT '',
        relevance_score REAL DEFAULT 0,
        discovered_at REAL NOT NULL,
        status TEXT DEFAULT 'pending',
        UNIQUE(url, category_id)
    );
    CREATE INDEX IF NOT EXISTS idx_queue_cat
        ON url_queue(category_id, status, discovered_at);
""")
conn.commit()
print("  OK  url_queue created")

print("\nDone. Restart the app.")
conn.close()