import sqlite3
conn = sqlite3.connect('data/webpulse.db')

print('=== morning_briefs ===')
for r in conn.execute(
    'SELECT id, brief_date, category_id, category_name, status, articles_scraped '
    'FROM morning_briefs ORDER BY id DESC LIMIT 10'
).fetchall():
    print(' ', r)

print()
print('=== insights per brief_id ===')
for r in conn.execute(
    'SELECT brief_id, COUNT(*) as cnt FROM brief_insights GROUP BY brief_id'
).fetchall():
    print(' ', r)

print()
print('=== sample insights ===')
for r in conn.execute(
    'SELECT id, brief_date, brief_id, prompt_key FROM brief_insights ORDER BY id DESC LIMIT 10'
).fetchall():
    print(' ', r)

print()
print('=== UNIQUE constraint on morning_briefs ===')
r = conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='morning_briefs'"
).fetchone()
if r:
    print(r[0][-300:])
else:
    print('NOT FOUND')

conn.close()