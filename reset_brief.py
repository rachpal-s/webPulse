import sqlite3
conn = sqlite3.connect('data/webpulse.db')
conn.execute("UPDATE morning_briefs SET status='failed', error_msg='manually reset' WHERE status='running'")
conn.commit()
rows = conn.execute("SELECT brief_date, status FROM morning_briefs").fetchall()
print("Current briefs:", rows)
conn.close()
print("Done — restart the app and click Run Now")