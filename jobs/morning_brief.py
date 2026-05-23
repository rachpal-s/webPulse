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
import re
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
# DEFAULT_PROMPTS moved to config.py as BRIEF_DEFAULT_PROMPTS

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

        # ── Step 6: Portfolio insights (optional) ────────────────────────────
        portfolio = store.get_portfolio()
        holding_insights = []

        if not cfg.portfolio_analysis_enabled:
            log.info("Portfolio analysis disabled (PORTFOLIO_ANALYSIS_ENABLED=false)")
        else:
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
        log.info("Running %d default insight prompts", len(_get_prompts()))
        print(f"[BRIEF] Running {len(_get_prompts())} insight prompts", flush=True)
        for dp in _get_prompts():
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

def _md_to_html(text: str) -> str:
    """Convert Markdown to HTML for email rendering."""
    if not text:
        return ""
    # Strip code fences
    text = re.sub(r"```(?:html)?\s*", "", text).strip()

    lines = text.split("\n")
    html = []
    in_ul = False

    for line in lines:
        s = line.strip()
        # Headings
        if s.startswith("### "):
            if in_ul: html.append("</ul>"); in_ul = False
            html.append(f'<h3 style="font-size:14px;font-weight:700;color:#22d3ee;margin:14px 0 6px;">{_inline_md(s[4:])}</h3>')
        elif s.startswith("## "):
            if in_ul: html.append("</ul>"); in_ul = False
            html.append(f'<h2 style="font-size:15px;font-weight:800;color:#fafafa;margin:16px 0 8px;">{_inline_md(s[3:])}</h2>')
        # Bullet
        elif re.match(r"^[\*\-\+] ", s):
            if not in_ul:
                html.append('<ul style="margin:6px 0 10px 20px;padding:0;">')
                in_ul = True
            html.append(f'<li style="font-size:13px;color:#d4d4d8;margin:4px 0;line-height:1.65;">{_inline_md(s[2:])}</li>')
        # Numbered list
        elif re.match(r"^\d+[.)]\s", s):
            if not in_ul:
                html.append('<ol style="margin:6px 0 10px 20px;padding:0;">')
                in_ul = True
            html.append(f'<li style="font-size:13px;color:#d4d4d8;margin:4px 0;line-height:1.65;">{_inline_md(re.sub(r"^\d+[.)]\s","",s))}</li>')
        elif not s:
            if in_ul: html.append("</ul>"); in_ul = False
            html.append('<div style="height:6px;"></div>')
        else:
            if in_ul: html.append("</ul>"); in_ul = False
            html.append(f'<p style="font-size:14px;line-height:1.7;color:#e4e4e7;margin:4px 0;">{_inline_md(s)}</p>')

    if in_ul:
        html.append("</ul>")
    return "\n".join(html)


def _inline_md(text: str) -> str:
    """Convert inline markdown (bold, italic, links) to HTML."""
    # Links [text](url)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<a href="\2" target="_blank" style="color:#22d3ee;text-decoration:none;font-size:0.9em;">[\1]</a>', text)
    # Bold **text**
    text = re.sub(r"\*\*(.+?)\*\*",
        r'<strong style="color:#fafafa;font-weight:600;">\1</strong>', text)
    # Italic *text*
    text = re.sub(r"\*(.+?)\*",
        r'<em style="color:#d4d4d8;">\1</em>', text)
    # Source citations [Source: title]
    text = re.sub(r"\[Source: ([^\]]+)\]",
        r'<span style="font-size:0.8em;color:#71717a;">[\1]</span>', text)
    return text


def _build_brief_html(today, overview_html, holding_insights,
                      site_summaries, article_count, session_id) -> str:
    now_str = datetime.now().strftime("%A, %d %B %Y — %I:%M %p IST")

    # ── Inline styles (email-safe — no external CSS, no CSS variables) ────────
    S = {
        "body":       "font-family:Arial,Helvetica,sans-serif;background:#09090b;color:#fafafa;margin:0;padding:0;",
        "wrap":       "max-width:680px;margin:0 auto;padding:24px 16px;",
        "header":     "border-bottom:2px solid #22d3ee;padding-bottom:20px;margin-bottom:28px;",
        "title":      "font-size:24px;font-weight:800;color:#fafafa;margin:0 0 4px;",
        "title_span": "color:#22d3ee;",
        "date":       "font-size:13px;color:#71717a;margin:0 0 6px;",
        "meta":       "font-size:12px;color:#71717a;margin:0;",
        "meta_a":     "color:#22d3ee;text-decoration:none;",
        "badge":      "display:inline-block;padding:3px 10px;border-radius:5px;font-size:11px;background:#0d2d1a;color:#4ade80;border:1px solid #166534;margin-top:8px;",
        "section":    "background:#111114;border:1px solid #27272a;border-radius:10px;padding:20px 24px;margin-bottom:20px;",
        "sec_title":  "font-size:16px;font-weight:800;color:#fafafa;margin:0 0 14px;border-bottom:1px solid #27272a;padding-bottom:8px;",
        "h3":         "font-size:15px;font-weight:700;color:#22d3ee;margin:18px 0 8px;",
        "h4":         "font-size:13px;font-weight:600;color:#a78bfa;margin:12px 0 6px;",
        "p":          "font-size:14px;line-height:1.7;color:#e4e4e7;margin:6px 0;",
        "ul":         "margin:8px 0 8px 20px;padding:0;",
        "li":         "font-size:13px;line-height:1.7;color:#d4d4d8;margin:4px 0;",
        "a":          "color:#22d3ee;text-decoration:none;",
        "hold_card":  "background:#18181c;border:1px solid #27272a;border-radius:8px;padding:14px 16px;margin-bottom:12px;",
        "hold_sym":   "font-size:16px;font-weight:800;color:#22d3ee;font-family:monospace;",
        "hold_name":  "font-size:12px;color:#71717a;margin-left:10px;",
        "hold_sect":  "display:inline-block;background:#1e1730;color:#a78bfa;padding:1px 7px;border-radius:4px;font-size:11px;margin-left:8px;",
        "hold_qty":   "font-size:11px;color:#71717a;",
        "hold_ins":   "font-size:13px;line-height:1.75;color:#d4d4d8;margin-top:8px;",
        "table":      "border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;",
        "th":         "background:#0e2a33;color:#22d3ee;padding:8px 12px;text-align:left;font-size:11px;text-transform:uppercase;border-bottom:2px solid #27272a;",
        "td":         "padding:8px 12px;border-bottom:1px solid #27272a;color:#e4e4e7;vertical-align:top;",
        "cta":        "display:inline-block;margin-top:14px;padding:10px 20px;background:#22d3ee;color:#09090b;border-radius:7px;font-weight:700;font-size:13px;text-decoration:none;",
        "footer":     "margin-top:32px;padding-top:16px;border-top:1px solid #27272a;font-size:11px;color:#71717a;text-align:center;",
        "src_li":     "font-size:12px;color:#d4d4d8;margin:4px 0;",
    }

    def clean_html_for_email(html: str) -> str:
        """Convert CSS-variable-based HTML to inline-styled email-safe HTML."""
        import re
        # Replace common block patterns with inline styles
        html = re.sub(r'<h3([^>]*)>', f'<h3\1 style="{S["h3"]}">', html)
        html = re.sub(r'<h4([^>]*)>', f'<h4\1 style="{S["h4"]}">', html)
        html = re.sub(r'<p([^>]*)>', f'<p\1 style="{S["p"]}">', html)
        html = re.sub(r'<ul([^>]*)>', f'<ul\1 style="{S["ul"]}">', html)
        html = re.sub(r'<ol([^>]*)>', f'<ol\1 style="{S["ul"]}">', html)
        html = re.sub(r'<li([^>]*)>', f'<li\1 style="{S["li"]}">', html)
        html = re.sub(r'<a ([^>]*href[^>]*)>', f'<a \1 style="{S["a"]}">', html)
        html = re.sub(r'<table([^>]*)>', f'<table\1 style="{S["table"]}" cellpadding="0" cellspacing="0">', html)
        html = re.sub(r'<th([^>]*)>', f'<th\1 style="{S["th"]}">', html)
        html = re.sub(r'<td([^>]*)>', f'<td\1 style="{S["td"]}">', html)
        html = re.sub(r'<strong([^>]*)>', '<strong\1 style="color:#fafafa;font-weight:600;">', html)
        # Strip CSS class attributes (email clients ignore them)
        html = re.sub(r' class="[^"]*"', '', html)
        return html

    holdings_html = ""
    for item in holding_insights:
        h = item["holding"]
        insight = clean_html_for_email(_md_to_html(item["insight"]))
        qty_line = f'<div style="{S["hold_qty"]}">Qty: {h["qty"]} @ ₹{h["avg_price"]}</div>' if h.get("qty") else ""
        sect = f'<span style="{S["hold_sect"]}">{h.get("sector","")}</span>' if h.get("sector") else ""
        holdings_html += f"""
        <div style="{S['hold_card']}">
          <div>
            <span style="{S['hold_sym']}">{h['symbol']}</span>
            <span style="{S['hold_name']}">{h['name']}</span>
            {sect}
            {qty_line}
          </div>
          <div style="{S['hold_ins']}">{insight}</div>
        </div>"""

    sites_html = "".join(
        f'<li style="{S["src_li"]}"><a href="{s["url"]}" style="{S["a"]}" target="_blank">{s["title"]}</a>'
        f' — {s["headline_count"]} headlines</li>'
        for s in site_summaries
    )

    ov = clean_html_for_email(_md_to_html(overview_html))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Morning Brief — {today}</title>
</head>
<body style="{S['body']}">
<div style="{S['wrap']}">

  <div style="{S['header']}">
    <div style="{S['title']}">📊 Morning Market <span style="{S['title_span']}">Brief</span></div>
    <div style="{S['date']}">{now_str}</div>
    <div style="{S['meta']}">{article_count} articles · {len(site_summaries)} sources ·
      <a href="http://localhost:8000/morning-brief" style="{S['meta_a']}">View in app</a>
    </div>
    <div style="{S['badge']}">✓ Ready</div>
  </div>

  <div style="{S['section']}">
    <div style="{S['sec_title']}">📈 Market Overview</div>
    {ov}
  </div>

  <div style="{S['section']}">
    <div style="{S['sec_title']}">💼 Portfolio Insights</div>
    {holdings_html if holdings_html else f'<p style="{S["p"]}">No holdings configured.</p>'}
  </div>

  <div style="{S['section']}">
    <div style="{S['sec_title']}">📰 Sources Scraped</div>
    <ul style="{S['ul']}">{sites_html}</ul>
    <a href="http://localhost:8000/qa?session_id={session_id}" style="{S['cta']}">
      Ask Questions About Today's News →
    </a>
  </div>

  <div style="{S['footer']}">
    Generated by WebPulse · Session: {session_id[:8]}… ·
    <a href="http://localhost:8000/morning-brief" style="{S['a']}">View Archive</a>
  </div>

</div>
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

def _get_prompts() -> list[dict]:
    """Return prompts from BRIEF_PROMPTS_JSON env if set, else config defaults."""
    if cfg.brief_prompts_json:
        try:
            import json as _j
            return _j.loads(cfg.brief_prompts_json)
        except Exception as e:
            log.warning("brief_prompts_json parse failed: %s — using defaults", e)
    return cfg.brief_default_prompts