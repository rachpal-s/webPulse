"""
regenerate_insights.py — Re-run insight prompts for a specific brief.
Use when insights are missing or linked to wrong brief_id.

Usage:
  python regenerate_insights.py              # shows all briefs
  python regenerate_insights.py <brief_id>   # re-runs insights for that brief
"""
import asyncio
import sys
import sqlite3
from datetime import date

DB = 'data/webpulse.db'


def list_briefs():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        'SELECT id, brief_date, category_name, status, articles_scraped, '
        '(SELECT COUNT(*) FROM brief_insights WHERE brief_id=morning_briefs.id) as insight_count '
        'FROM morning_briefs ORDER BY id DESC LIMIT 20'
    ).fetchall()
    conn.close()
    print(f'{"ID":>4}  {"Date":<12}  {"Category":<15}  {"Status":<8}  {"Articles":>8}  {"Insights":>8}')
    print('-' * 65)
    for r in rows:
        print(f'{r[0]:>4}  {r[1]:<12}  {r[2]:<15}  {r[3]:<8}  {r[4]:>8}  {r[5]:>8}')


async def regen(brief_id: int):
    import sys
    sys.path.insert(0, '.')
    from config import get_settings
    from rag.pipeline import get_store, query as rag_query
    from jobs.morning_brief import _get_prompts

    cfg = get_settings()
    store = get_store()

    # Load brief
    conn = sqlite3.connect(DB)
    row = conn.execute(
        'SELECT id, brief_date, category_id, category_name, rag_session_id '
        'FROM morning_briefs WHERE id=?', (brief_id,)
    ).fetchone()
    conn.close()

    if not row:
        print(f'Brief {brief_id} not found')
        return

    brief_id, brief_date, category_id, category_name, session_id = row
    print(f'Brief {brief_id}: {brief_date} / {category_name} / session={session_id[:8]}…')

    if not session_id:
        print('ERROR: No RAG session — cannot regenerate without re-scraping')
        return

    # Get category-specific prompts or defaults
    cat_prompts_raw = store.get_category_prompts(category_id)
    cat_prompts = [
        {"key": p["prompt_key"], "label": p["label"], "prompt": p["prompt_text"]}
        for p in cat_prompts_raw
    ] if cat_prompts_raw else []
    prompts = cat_prompts if cat_prompts else _get_prompts()
    print(f'Running {len(prompts)} prompts…')

    for dp in prompts:
        print(f'  → {dp["key"]}', end='', flush=True)
        try:
            result = await rag_query(session_id, dp['prompt'], top_k=8)
            store.save_insight(
                brief_date, dp['key'], dp['label'],
                result.answer, result.sources_used,
                brief_id=brief_id
            )
            print(' ✓')
        except Exception as e:
            print(f' ✗ {e}')
            store.save_insight(
                brief_date, dp['key'], dp['label'],
                f'<p style="color:#f87171">Error: {e}</p>',
                brief_id=brief_id
            )

    print(f'\nDone. Refresh /morning-brief to see insights for brief {brief_id}.')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        list_briefs()
        print('\nUsage: python regenerate_insights.py <brief_id>')
    else:
        asyncio.run(regen(int(sys.argv[1])))