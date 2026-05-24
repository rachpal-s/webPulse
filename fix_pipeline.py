"""
fix_pipeline.py — Ensure queue methods are inside VectorStore class.
Run once: python fix_pipeline.py
"""
import re

path = 'rag/pipeline.py'
content = open(path, encoding='utf-8').read()

# Check if methods are already inside VectorStore
from rag.pipeline import VectorStore
if hasattr(VectorStore, 'clear_old_queue'):
    print("Already fixed — clear_old_queue is inside VectorStore")
    exit(0)

print("Queue methods found outside VectorStore — fixing...")

# Find the methods (they exist but are at wrong indentation/position)
# Extract them
method_names = ['upsert_queue_urls', 'get_queued_urls', 'get_queue_preview',
                'mark_queue_used', 'exclude_queue_url', 'clear_old_queue']

# Find where VectorStore ends (first line starting with 'def ' or 'class ' at col 0
# after the class definition)
lines = content.split('\n')

# Find end of VectorStore class
vs_start = next(i for i,l in enumerate(lines) if l.startswith('class VectorStore'))
vs_end = len(lines)
for i in range(vs_start+1, len(lines)):
    if lines[i] and not lines[i][0].isspace() and lines[i][0] not in ('#', '\n', ''):
        vs_end = i
        break

print(f"VectorStore: lines {vs_start+1}-{vs_end}")

# Check if any queue methods are OUTSIDE the class (at module level)
queue_methods_outside = []
for i, l in enumerate(lines):
    for m in method_names:
        if f'def {m}' in l:
            indent = len(l) - len(l.lstrip())
            location = 'INSIDE' if i < vs_end else 'OUTSIDE'
            indented = 'correct' if indent == 4 else f'indent={indent}'
            print(f"  {m}: line {i+1} {location} class, {indented}")
            if i >= vs_end or indent == 0:
                queue_methods_outside.append(i)

if not queue_methods_outside:
    print("Methods appear to be inside class but not recognized.")
    print("Check for syntax errors around line 1000:")
    for i, l in enumerate(lines[995:1010], 996):
        print(f"  {i}: {repr(l)}")
    exit(1)

# The methods are outside — add them inside VectorStore before its end
# First remove the loose method blocks
QUEUE_METHODS = '''
    # ── URL Queue methods ─────────────────────────────────────────────────────

    def upsert_queue_urls(self, category_id: int, source_url: str, pages: list[dict]):
        import time as _time
        conn = self._connect()
        now = _time.time()
        for p in pages:
            conn.execute("""
                INSERT INTO url_queue
                    (url,category_id,source_url,title,summary,relevance_score,discovered_at,status)
                VALUES (?,?,?,?,?,?,?,'pending')
                ON CONFLICT(url,category_id) DO UPDATE SET
                    title=excluded.title,summary=excluded.summary,
                    relevance_score=excluded.relevance_score,
                    discovered_at=excluded.discovered_at,status='pending'
            """, (p["url"],category_id,source_url,
                  p.get("title",""),p.get("summary",""),p.get("score",0),now))
        conn.commit()
        conn.close()

    def get_queued_urls(self, category_id: int,
                         max_age_hours: int = 6, limit: int = 30) -> list[dict]:
        import time as _time
        cutoff = _time.time() - max_age_hours * 3600
        conn = self._connect()
        rows = conn.execute("""
            SELECT * FROM url_queue
            WHERE category_id=? AND status='pending' AND discovered_at>?
            ORDER BY relevance_score DESC, discovered_at DESC LIMIT ?
        """, (category_id, cutoff, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_queue_preview(self, max_age_hours: int = 6) -> list[dict]:
        import time as _time
        cutoff = _time.time() - max_age_hours * 3600
        conn = self._connect()
        rows = conn.execute("""
            SELECT q.*,c.name as category_name,c.icon as category_icon
            FROM url_queue q JOIN brief_categories c ON q.category_id=c.id
            WHERE q.status='pending' AND q.discovered_at>?
            ORDER BY q.category_id, q.relevance_score DESC
        """, (cutoff,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def mark_queue_used(self, urls: list[str], category_id: int):
        conn = self._connect()
        for url in urls:
            conn.execute("UPDATE url_queue SET status='used' WHERE url=? AND category_id=?",
                         (url,category_id))
        conn.commit(); conn.close()

    def exclude_queue_url(self, url: str, category_id: int):
        conn = self._connect()
        conn.execute("UPDATE url_queue SET status='excluded' WHERE url=? AND category_id=?",
                     (url,category_id))
        conn.commit(); conn.close()

    def clear_old_queue(self, max_age_hours: int = 24) -> int:
        import time as _time
        cutoff = _time.time() - max_age_hours * 3600
        conn = self._connect()
        n = conn.execute("DELETE FROM url_queue WHERE discovered_at<?", (cutoff,)).rowcount
        conn.commit(); conn.close()
        return n

'''

# Remove any existing loose queue method blocks (outside class)
# Find the block: starts at first loose def, ends before next top-level def/class
def find_block_end(lines, start):
    for i in range(start+1, len(lines)):
        if lines[i] and not lines[i][0].isspace() and lines[i].strip():
            return i
    return len(lines)

# Remove loose blocks (work backwards to preserve indices)
to_remove = sorted(set(queue_methods_outside), reverse=True)
for start in to_remove:
    end = find_block_end(lines, start)
    print(f"Removing loose block lines {start+1}-{end}")
    del lines[start:end]

# Re-find VectorStore end after removals
content2 = '\n'.join(lines)
lines2 = content2.split('\n')
vs_start2 = next(i for i,l in enumerate(lines2) if l.startswith('class VectorStore'))
vs_end2 = len(lines2)
for i in range(vs_start2+1, len(lines2)):
    if lines2[i] and not lines2[i][0].isspace() and lines2[i][0] not in ('#',):
        vs_end2 = i
        break

print(f"Inserting queue methods at line {vs_end2} (before class end)")
lines2.insert(vs_end2, QUEUE_METHODS)
final = '\n'.join(lines2)

import ast
try:
    ast.parse(final)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax error: {e}")
    exit(1)

open(path, 'w', encoding='utf-8').write(final)
print("Done — restart the app")

# Verify
from importlib import reload, import_module
import rag.pipeline as rp
reload(rp)
print("clear_old_queue in VectorStore:", hasattr(rp.VectorStore, 'clear_old_queue'))