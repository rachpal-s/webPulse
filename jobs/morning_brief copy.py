"""
jobs/morning_brief.py — Morning market brief batch job.

Sequence:
  1. Fetch daily URLs from url_history
  2. Scrape each (headlines + top N articles)
  3. Ingest all content into RAG session
  4. Generate market overview via LLM
  5. For each portfolio holding → targeted Q&A
  6. Assemble HTML brief
  7. Save to DB + send email if SMTP configured
"""
import asyncio
import logging
import traceback
from datetime import date, datetime
from pathlib import Path

from config import get_settings
from rag.ollama import get_ollama_client
from rag.pipeline import get_store, ingest_documents, query as rag_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("morning_brief")
cfg = get_settings()

# ── Default insight prompts ───────────────────────────────────────────────────
DEFAULT_PROMPTS = [
    {
        "key": "trending_news",
        "label": "Top Trending News",
        "prompt": (
            "What are the top 10 trending news stories today that may impact market dynamics? "
            "List them as an HTML numbered list with a one-line explanation of potential market impact for each."
        ),
    },
    {
        "key": "market_outlook",
        "label": "India Market Outlook",
        "prompt": (
            "Based on today's news context, how is the Indian stock market (Sensex/Nifty) likely to behave today? "
            "Consider global cues, FII/DII activity, sector trends, and macro factors. "
            "Give a clear directional view: bullish, bearish, or range-bound, with key reasons."
        ),
    },
    {
        "key": "stock_calls",
        "label": "Expert Stock Recommendations",
        "prompt": (
            "Based on the news context, which specific stocks have been explicitly recommended by analysts or experts? "
            "Create an HTML table with columns: Stock, Recommendation (BUY/SELL/HOLD), Target Price (if mentioned), "
            "Analyst/Source, and Key Reason. Only include stocks with explicit recommendations in the news."
        ),
    },
    {
        "key": "focus_areas",
        "label": "Focus Areas Today",
        "prompt": (
            "Based on today's news, what are the key focus areas, themes, or sectors that investors should watch today? "
            "Include: sectors in spotlight, key events or data releases, geopolitical factors, and any earnings announcements. "
            "Format as an HTML table with columns: Area, Why It Matters, Likely Impact."
        ),
    },
    {
        "key": "risk_factors",
        "label": "Risk Factors & Caution Zones",
        "prompt": (
            "Based on today's news context, what are the key risk factors or caution zones for the market today? "
            "Include global risks, domestic concerns, overvalued sectors, or stocks facing headwinds. "
            "Be specific and factual. Format as a concise HTML bulleted list."
        ),
    },
]

# ── Prompt templates ──────────────────────────────────────────────────────────

OVERVIEW_PROMPT = """You are a senior financial analyst preparing a morning market brief.

Based on the news articles provided as context, write a structured morning market overview.

Format as clean HTML (no markdown). Include:
<h3>Market Sentiment</h3> — overall tone (bullish/bearish/mixed) with key reasons
<h3>Top Stories</h3> — 4-6 most important market-moving news items as <ul><li> list
<h3>Sectors in Focus</h3> — which sectors are in news and why
<h3>Watch Today</h3> — key events, data releases, or stocks to watch

Be concise, factual, and specific. Use numbers where available."""

HOLDING_PROMPT = """You are a portfolio analyst. Based only on today's news context, 
provide a brief insight about {symbol} ({name}).

Answer these questions in 2-3 sentences as clean HTML <p> tags:
1. Is there any direct news about {symbol} today?
2. Are there any macro/sector factors that could affect this stock?
3. What is the sentiment — positive, negative, or neutral?

If there is no relevant news, say so clearly in one sentence.
Do not fabricate news. Only use what's in the context."""


# ── Email sender ──────────────────────────────────────────────────────────────

async def send_email(subject: str, html_body: str):
    if not cfg.smtp_enabled:
        log.info("SMTP disabled — skipping email")
        return
    try:
        import aiosmtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = cfg.smtp_from
        msg["To"] = cfg.smtp_to
        msg.attach(MIMEText(html_body, "html"))

        # Port 465 = SSL/TLS directly; port 587 = STARTTLS
        use_tls = cfg.smtp_port == 465
        await aiosmtplib.send(
            msg,
            hostname=cfg.smtp_host,
            port=cfg.smtp_port,
            username=cfg.smtp_user,
            password=cfg.smtp_password,
            use_tls=use_tls,
            start_tls=not use_tls,
        )
        log.info("Email sent to %s", cfg.smtp_to)
    except Exception as e:
        log.error("Email failed: %s", e)


# ── Main brief runner ─────────────────────────────────────────────────────────

async def run_morning_brief(force: bool = False) -> dict:
    """
    Run the morning brief. Returns status dict.
    force=True skips the already-run-today check.
    """
    today = date.today().isoformat()
    store = get_store()
    ollama = get_ollama_client()

    # ── Check if already ran today ────────────────────────────────────────────
    existing = store.get_brief(today)
    if existing and existing["status"] == "done" and not force:
        log.info("Brief for %s already done — skipping", today)
        return {"status": "already_done", "date": today}

    print(f"[BRIEF] Starting morning brief for {today}", flush=True)
    log.info("Starting morning brief for %s", today)
    store.start_brief(today)

    try:
        # ── Step 1: Get daily URLs ────────────────────────────────────────────
        daily_urls = store.get_daily_urls()
        print(f"[BRIEF] Daily URLs found: {len(daily_urls)}", flush=True)
        if not daily_urls:
            log.warning("No daily URLs configured. Mark URLs as daily in the portfolio page.")
            store.finish_brief(today, _no_urls_html(today), "", 0,
                               "No daily URLs configured")
            return {"status": "no_urls"}

        log.info("Scraping %d daily URLs", len(daily_urls))
        print(f"[BRIEF] Scraping {len(daily_urls)} URLs: {[u['url'] for u in daily_urls]}", flush=True)

        # ── Step 2: Scrape each URL ───────────────────────────────────────────
        # Use sys.modules to get _full_scrape from main (already loaded by uvicorn)
        import sys as _sys
        _main_mod = _sys.modules.get("main") or __import__("main")
        _full_scrape = _main_mod._full_scrape

        all_documents = []
        site_summaries = []

        async def _scrape_url(url: str, strategies=None):
            if strategies is None:
                strategies = ["trafilatura","newspaper3k","readability","goose3","beautifulsoup"]
            try:
                return await _full_scrape(url, strategies=strategies)
            except Exception as e:
                # HTTPException or network error — log and return None
                log.warning("_scrape_url %s: %s", url, e)
                return None

        for url_rec in daily_urls:
            url = url_rec["url"]
            log.info("Scraping %s", url)
            try:
                result = await asyncio.wait_for(_scrape_url(url), timeout=60)
                if not result:
                    log.warning("No result for %s", url)
                    continue

                headlines = (result.headlines or [])[:cfg.brief_articles_per_site]
                title = result.title or url
                content = result.content or ""
                tables_md = result.tables_md or ""
                page_type = result.page_type or "unknown"

                site_summaries.append({
                    "url": url, "title": title,
                    "page_type": page_type,
                    "headline_count": len(headlines),
                })

                # Add main page content
                main_content = (tables_md + "\n\n" + content).strip()
                if len(main_content.split()) > 30:
                    all_documents.append({
                        "url": url, "title": title,
                        "content": main_content, "page_type": page_type,
                    })

                # ── Step 3: Dig into top articles ─────────────────────────────
                if headlines:
                    log.info("  Digging %d articles from %s", len(headlines), url)
                    for h in headlines:
                        try:
                            art = await asyncio.wait_for(
                                _scrape_url(h["url"],
                                    strategies=["trafilatura","newspaper3k","readability"]),
                                timeout=30
                            )
                            if art:
                                art_content = art.content or ""
                                art_title = art.title or h.get("title", h["url"])
                                if len(art_content.split()) > 50:
                                    all_documents.append({
                                        "url": h["url"], "title": art_title,
                                        "content": art_content, "page_type": "article",
                                    })
                        except Exception as e:
                            log.debug("Article %s failed: %s", h["url"], e)

            except Exception as e:
                log.warning("Failed to scrape %s: %s", url, e)
                import traceback as _tb
                log.debug("Scrape traceback: %s", _tb.format_exc())

        if not all_documents:
            store.finish_brief(today, _no_content_html(today), "", 0,
                               "No content scraped")
            return {"status": "no_content"}

        log.info("Ingesting %d documents into RAG", len(all_documents))

        # ── Step 4: Ingest into RAG ───────────────────────────────────────────
        rag_ctx = await ingest_documents(all_documents)
        session_id = rag_ctx.session_id
        log.info("RAG session: %s (%s mode, %d chunks)",
                 session_id, rag_ctx.mode, rag_ctx.chunk_count)

        # ── Step 5: Generate market overview ──────────────────────────────────
        log.info("Generating market overview")
        overview_result = await rag_query(
            session_id,
            "Provide a comprehensive morning market overview based on today's news. "
            "Cover: overall sentiment, top market-moving stories, sectors in focus, "
            "and key things to watch today.",
            top_k=10,
        )
        overview_html = overview_result.answer

        # ── Step 6: Portfolio insights ────────────────────────────────────────
        portfolio = store.get_portfolio()
        holding_insights = []

        for holding in portfolio:
            log.info("Analysing %s", holding["symbol"])
            try:
                result = await rag_query(
                    session_id,
                    f"What is today's news and market outlook for {holding['symbol']} "
                    f"({holding['name']}, {holding['sector']} sector)? "
                    f"Any direct news, sector tailwinds/headwinds, or macro factors?",
                    top_k=6,
                )
                holding_insights.append({
                    "holding": holding,
                    "insight": result.answer,
                    "sources": result.sources_used,
                })
            except Exception as e:
                holding_insights.append({
                    "holding": holding,
                    "insight": f"<p>Could not retrieve insight: {e}</p>",
                    "sources": [],
                })

        # ── Step 7: Assemble HTML brief ───────────────────────────────────────
        html = _build_brief_html(
            today, overview_html, holding_insights,
            site_summaries, len(all_documents), session_id
        )

        # ── Step 8: Save + send ────────────────────────────────────────────────
        # ── Step 7b: Default insight prompts ─────────────────────────────────────
        log.info("Running %d default insight prompts", len(DEFAULT_PROMPTS))
        print(f"[BRIEF] Running {len(DEFAULT_PROMPTS)} insight prompts", flush=True)
        for dp in DEFAULT_PROMPTS:
            try:
                print(f"[BRIEF] Prompt: {dp['key']}", flush=True)
                ins_result = await rag_query(session_id, dp["prompt"], top_k=8)
                store.save_insight(today, dp["key"], dp["label"],
                                   ins_result.answer, ins_result.sources_used)
                log.info("Insight done: %s", dp["key"])
            except Exception as e:
                log.warning("Insight failed %s: %s", dp["key"], e)
                store.save_insight(today, dp["key"], dp["label"],
                    f"<p style='color:#f87171'>Could not generate: {e}</p>")

        store.finish_brief(today, html, session_id, len(all_documents))

        subject = f"📊 Morning Market Brief — {datetime.now().strftime('%a %d %b %Y')}"
        await send_email(subject, html)

        log.info("Morning brief complete: %d articles, %d holdings",
                 len(all_documents), len(portfolio))
        return {
            "status": "done",
            "date": today,
            "session_id": session_id,
            "articles": len(all_documents),
            "holdings_analysed": len(portfolio),
        }

    except Exception as e:
        err = traceback.format_exc()
        print(f"[BRIEF] EXCEPTION: {err}", flush=True)
        log.error("Brief failed: %s", err)
        store.finish_brief(today, _error_html(today, str(e)), "", 0, str(e))
        return {"status": "failed", "error": str(e)}


# ── HTML builder ──────────────────────────────────────────────────────────────

def _build_brief_html(today, overview_html, holding_insights,
                      site_summaries, article_count, session_id) -> str:
    now_str = datetime.now().strftime("%A, %d %B %Y — %I:%M %p IST")
    holdings_html = ""
    for item in holding_insights:
        h = item["holding"]
        insight = item["insight"]
        pnl_color = "#4ade80" if h.get("avg_price", 0) > 0 else "#71717a"
        holdings_html += f"""
        <div class="holding-card">
          <div class="holding-header">
            <div class="holding-symbol">{h['symbol']}</div>
            <div class="holding-meta">
              <span>{h['name']}</span>
              <span class="holding-sector">{h.get('sector','')}</span>
              {f'<span>{h["exchange"]}</span>' if h.get('exchange') else ''}
            </div>
            {f'<div class="holding-qty">Qty: {h["qty"]} @ ₹{h["avg_price"]}</div>'
              if h.get('qty') else ''}
          </div>
          <div class="holding-insight">{insight}</div>
        </div>"""

    sites_html = "".join(
        f'<li><a href="{s["url"]}" target="_blank">{s["title"]}</a>'
        f' — {s["headline_count"]} headlines</li>'
        for s in site_summaries
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Morning Brief — {today}</title>
<style>
  :root {{
    --bg:#09090b; --surface:#111114; --surface2:#18181c;
    --border:#27272a; --text:#fafafa; --muted:#71717a;
    --accent:#22d3ee; --accent2:#a78bfa; --success:#4ade80;
    --warn:#fb923c; --err:#f87171;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Inter',system-ui,sans-serif; background:var(--bg);
          color:var(--text); line-height:1.7; padding:32px 24px; }}
  .brief-header {{ border-bottom:2px solid var(--accent);
                   padding-bottom:20px; margin-bottom:32px; }}
  .brief-title {{ font-size:1.8rem; font-weight:800; letter-spacing:-.03em; }}
  .brief-title span {{ color:var(--accent); }}
  .brief-date {{ color:var(--muted); font-size:.9rem; margin-top:4px; }}
  .brief-meta {{ font-size:.75rem; color:var(--muted); margin-top:8px; }}
  .brief-meta a {{ color:var(--accent); text-decoration:none; }}
  h3 {{ font-size:1.1rem; font-weight:700; color:var(--accent);
        margin:24px 0 12px; padding-bottom:6px;
        border-bottom:1px solid var(--border); }}
  h4 {{ font-size:.95rem; font-weight:600; color:var(--accent2); margin:16px 0 8px; }}
  p {{ margin:8px 0; font-size:.93rem; color:#e4e4e7; }}
  ul {{ margin:8px 0 8px 20px; }}
  li {{ margin:5px 0; font-size:.9rem; color:#d4d4d8; }}
  a {{ color:var(--accent); }}
  .section {{ background:var(--surface); border:1px solid var(--border);
              border-radius:12px; padding:24px; margin-bottom:24px; }}
  .section-title {{ font-size:1.2rem; font-weight:800; color:var(--text);
                    margin-bottom:16px; }}
  .holding-card {{ background:var(--surface2); border:1px solid var(--border);
                   border-radius:10px; padding:18px; margin-bottom:14px; }}
  .holding-header {{ display:flex; align-items:baseline; gap:14px;
                     flex-wrap:wrap; margin-bottom:12px; }}
  .holding-symbol {{ font-size:1.1rem; font-weight:800; color:var(--accent);
                     font-family:monospace; }}
  .holding-meta {{ font-size:.82rem; color:var(--muted); display:flex;
                   gap:10px; flex-wrap:wrap; }}
  .holding-sector {{ background:rgba(167,139,250,.12); color:var(--accent2);
                     padding:1px 8px; border-radius:4px; font-size:.72rem; }}
  .holding-qty {{ font-size:.78rem; color:var(--muted); margin-left:auto; }}
  .holding-insight {{ font-size:.88rem; line-height:1.75; color:#d4d4d8; }}
  .sources-list {{ font-size:.72rem; color:var(--muted); margin-top:8px; }}
  .status-badge {{ display:inline-block; padding:4px 12px; border-radius:6px;
                   font-size:.72rem; font-family:monospace;
                   background:rgba(74,222,128,.1); color:var(--success);
                   border:1px solid rgba(74,222,128,.25); margin-top:4px; }}
  .qa-link {{ display:inline-block; margin-top:16px; padding:10px 20px;
              background:var(--accent); color:#09090b; border-radius:8px;
              font-weight:700; font-size:.85rem; text-decoration:none; }}
  footer {{ margin-top:40px; padding-top:20px; border-top:1px solid var(--border);
            font-size:.72rem; color:var(--muted); }}
</style>
</head>
<body>

<div class="brief-header">
  <div class="brief-title">📊 Morning Market <span>Brief</span></div>
  <div class="brief-date">{now_str}</div>
  <div class="brief-meta">
    {article_count} articles analysed from {len(site_summaries)} sources ·
    <a href="/morning-brief?session_id={session_id}">Open for Q&amp;A</a>
  </div>
  <div class="status-badge">✓ Ready</div>
</div>

<div class="section">
  <div class="section-title">📈 Market Overview</div>
  {overview_html}
</div>

<div class="section">
  <div class="section-title">💼 Portfolio Insights</div>
  {holdings_html if holdings_html else '<p style="color:var(--muted)">No holdings configured. <a href="/portfolio">Add your portfolio →</a></p>'}
</div>

<div class="section">
  <div class="section-title">📰 Sources Scraped</div>
  <ul>{sites_html}</ul>
  <a class="qa-link" href="/qa?session_id={session_id}">
    Ask Questions About Today's News →
  </a>
</div>

<footer>
  Generated by WebPulse Morning Brief · Session: {session_id[:8]}… ·
  <a href="/morning-brief">View Archive</a>
</footer>
</body></html>"""


def _no_urls_html(today):
    return f"<html><body><h2>No daily URLs configured for {today}</h2>" \
           "<p>Go to <a href='/morning-brief'>Morning Brief settings</a> " \
           "and mark URLs as daily.</p></body></html>"

def _no_content_html(today):
    return f"<html><body><h2>No content scraped for {today}</h2>" \
           "<p>All daily URLs failed to return content.</p></body></html>"

def _error_html(today, error):
    return f"<html><body><h2>Brief failed for {today}</h2>" \
           f"<pre>{error}</pre></body></html>"