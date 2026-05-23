"""
WebPulse — Multi-Strategy Scraper + Adaptive RAG
FastAPI app with Jinja2 templates
"""
import asyncio
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import get_settings
from rag.ollama import get_ollama_client
from rag.pipeline import ingest_documents, query as rag_query, get_store
from scraper.filter import apply_filter
from datetime import datetime as _dt
from scraper.cleaner import extract_tables_markdown
from scraper.detector import detect_page_type, extract_headlines
from scraper.engine import (
    ScrapeResult, StrategyResult,
    fetch_html, _metadata,
    _trafilatura, _newspaper3k, _readability, _goose3, _beautifulsoup, _playwright,
    _pick_best, clean_text, IS_BLOCKED,
)

cfg = get_settings()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Startup
    from jobs.scheduler import start_scheduler, startup_catchup
    start_scheduler()
    await startup_catchup()
    yield
    # Shutdown
    from jobs.scheduler import stop_scheduler
    stop_scheduler()

app = FastAPI(title=cfg.app_title, version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

def _datetimeformat(ts):
    if not ts: return "—"
    try: return _dt.fromtimestamp(float(ts)).strftime("%d %b %I:%M %p")
    except: return str(ts)

templates.env.filters["datetimeformat"] = _datetimeformat

# Cache-busting: inject a build version based on static file mtimes
import hashlib as _hashlib
def _static_version():
    static_dir = BASE / "static"
    h = _hashlib.md5()
    for f in sorted(static_dir.rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:8]

_BUILD_VER = _static_version()
templates.env.globals["ver"] = _BUILD_VER

STRATEGY_ORDER = ["trafilatura", "newspaper3k", "readability", "goose3",
                  "beautifulsoup", "playwright"]


# ── Helpers ───────────────────────────────────────────────────────────────────

# Per-strategy timeouts (seconds). Playwright gets extra time for JS rendering.
_STRATEGY_TIMEOUTS = {
    "trafilatura":   15,
    "newspaper3k":   20,
    "readability":   10,
    "goose3":        15,
    "beautifulsoup": 10,
    "playwright":    60,   # JS rendering + network wait
}


async def _run_with_timeout(coro_or_future, strategy: str) -> StrategyResult:
    """Wrap a strategy future with a timeout; return a failure result on expiry."""
    timeout = _STRATEGY_TIMEOUTS.get(strategy, 30)
    try:
        return await asyncio.wait_for(asyncio.wrap_future(coro_or_future), timeout=timeout)
    except asyncio.TimeoutError:
        return StrategyResult(
            strategy=strategy, success=False,
            error=f"Timed out after {timeout}s",
        )
    except Exception as e:
        return StrategyResult(strategy=strategy, success=False, error=str(e)[:200])


async def _run_strategies(
    html: str, url: str,
    strategies: list[str],
    wait_selector: Optional[str] = None,
    wait_seconds: float = 4.0,
) -> list[StrategyResult]:
    loop = asyncio.get_event_loop()

    # Run in defined order: Trafilatura first, Playwright last
    ordered = [s for s in STRATEGY_ORDER if s in strategies]

    tasks = []
    for s in ordered:
        if s == "trafilatura" and html:
            tasks.append((s, loop.run_in_executor(None, _trafilatura, html, url)))
        elif s == "newspaper3k":
            tasks.append((s, loop.run_in_executor(None, _newspaper3k, url)))
        elif s == "readability" and html:
            tasks.append((s, loop.run_in_executor(None, _readability, html)))
        elif s == "goose3" and html:
            tasks.append((s, loop.run_in_executor(None, _goose3, html, url)))
        elif s == "beautifulsoup" and html:
            tasks.append((s, loop.run_in_executor(None, _beautifulsoup, html)))
        elif s == "playwright":
            tasks.append((s, loop.run_in_executor(
                None, _playwright, url, wait_selector, wait_seconds
            )))

    return list(await asyncio.gather(*[_run_with_timeout(f, s) for s, f in tasks]))


async def _full_scrape(
    url: str,
    strategies: Optional[list[str]] = None,
    wait_selector: Optional[str] = None,
    wait_seconds: float = 4.0,
) -> ScrapeResult:
    if strategies is None:
        strategies = STRATEGY_ORDER

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    t_total = time.perf_counter()
    html = ""
    fetch_ms = 0.0

    # Fetch HTML for static strategies
    need_html = any(s in strategies for s in
                    ["trafilatura","readability","goose3","beautifulsoup"])
    if need_html:
        try:
            html, fetch_ms = await fetch_html(url)
        except Exception as e:
            if "playwright" not in strategies:
                # Raise HTTPException only inside a request context; use plain Exception otherwise
                import sys as _sys
                if "fastapi" in str(type(_sys.modules.get("starlette.requests", None))):
                    raise HTTPException(502, detail=str(e))
                raise Exception(f"Fetch failed: {e}") from e

    metadata = _metadata(html, url) if html else {"domain": url}

    # Detect page type
    signals = detect_page_type(html or "", url)
    page_type = signals.page_type

    # Run strategies
    results = await _run_strategies(html, url, strategies,
                                    wait_selector, wait_seconds)
    best = _pick_best(results)
    total_ms = (time.perf_counter() - t_total) * 1000

    # ── Post-run: get the best rendered HTML (prefer Playwright's version) ─────
    # Playwright executes JS, so its HTML has populated table rows.
    # Static html from httpx has empty tbodys on JS-rendered pages (Moneycontrol etc.)
    pw_result = next((r for r in results if r.strategy == "playwright"
                      and r.success and r.extra.get("rendered_html")), None)
    rendered_html = pw_result.extra["rendered_html"] if pw_result else html

    # ── Re-detect page type on rendered HTML if static detection was weak ─────
    if page_type in ("unknown", "article") and rendered_html and rendered_html != html:
        re_signals = detect_page_type(rendered_html, url)
        if re_signals.confidence > signals.confidence:
            page_type = re_signals.page_type
            signals = re_signals

    # ── Upgrade 'unknown' to 'data' if tables are present ────────────────────
    # Handles cases where URL path gives no clear signal but page has tables
    if page_type == "unknown" and signals.table_count >= 1:
        page_type = "data"

    # ── Extract tables ────────────────────────────────────────────────────────
    tables_md, table_count = ("", 0)
    if page_type in ("data", "mixed"):
        # Prefer rendered HTML (Playwright) over static HTML for table extraction
        source_html = rendered_html or html
        if source_html:
            tables_md, table_count = extract_tables_markdown(source_html)
        # If tables_md still empty but playwright embedded it in content, parse from there
        if not tables_md and best and best.strategy == "playwright":
            table_count = best.extra.get("tables_found", 0)
            if best.text and "## Table Data" in best.text:
                start = best.text.index("## Table Data") + len("## Table Data")
                section = best.text[start:].split("\n\n---\n\n")[0].strip()
                if "|" in section:
                    tables_md = section
        # If still empty, fall back to best non-playwright content
        if not tables_md and best:
            non_pw = next((r for r in results
                          if r.strategy != "playwright" and r.success and r.text
                          and len(r.text.split()) > 20), None)
            if non_pw:
                tables_md = non_pw.text
    elif best and best.strategy == "playwright":
        table_count = best.extra.get("tables_found", 0)

    # ── Extract headlines (all page types — mixed pages have both tables + news) ─
    headlines = []
    if html:
        headlines = extract_headlines(html, url, cfg.scraper_max_headlines)

    # Upgrade page_type to 'mixed' when we have both tables and headlines
    if page_type == "data" and len(headlines) >= 3:
        page_type = "mixed"
    elif page_type in ("unknown", "article") and len(headlines) >= 3:
        page_type = "homepage"

    return ScrapeResult(
        url=url,
        page_type=page_type,
        best_strategy=best.strategy if best else None,
        title=(best.title if best else None) or metadata.get("og_title"),
        content=best.text if best else None,
        word_count=best.word_count if best else 0,
        tables_md=tables_md or None,
        table_count=table_count,
        fetch_time_ms=round(fetch_ms, 1),
        total_time_ms=round(total_ms, 1),
        all_results=results,
        metadata=metadata,
        headlines=headlines,
    )


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "title": cfg.app_title,
        "strategies": STRATEGY_ORDER,
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    ollama = get_ollama_client()
    inf_status = await ollama.check_inference()
    emb_status = await ollama.check_embed()
    return templates.TemplateResponse(request, "settings.html", {
        "request": request,
        "title": cfg.app_title,
        "cfg": cfg,
        "inference_status": inf_status,
        "embed_status": emb_status,
    })


@app.get("/qa", response_class=HTMLResponse)
async def qa_page(request: Request, session_id: Optional[str] = None):
    session = None
    if session_id:
        store = get_store()
        session = store.get_session(session_id)
    return templates.TemplateResponse(request, "qa.html", {
        "request": request,
        "title": cfg.app_title,
        "session": session,
        "session_id": session_id,
    })


# ── API: Scrape ───────────────────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    url: str                                    # single URL or comma/semicolon separated list
    strategies: Optional[list[str]] = None
    wait_for_selector: Optional[str] = None
    wait_seconds: Optional[float] = 4.0
    filter_phrase: Optional[str] = None         # space-separated words to filter results


class ScrapeResponse(BaseModel):
    url: str
    page_type: str
    page_type_confidence: float = 0.0
    best_strategy: Optional[str]
    title: Optional[str]
    content: Optional[str]
    word_count: int
    tables_md: Optional[str]
    table_count: int
    fetch_time_ms: float
    total_time_ms: float
    headlines: list[dict]
    all_results: list[dict]
    metadata: dict
    # Multi-URL and filter fields
    multi_results: list[dict] = []              # results for each URL when multiple given
    filter_phrase: Optional[str] = None
    filter_applied: bool = False
    filter_insight: str = ""                    # GenAI insight about filtered results


def _parse_urls(raw: str) -> list[str]:
    """Split comma/semicolon/newline separated URLs and normalise each."""
    import re as _re
    parts = _re.split(r"[,;\n]+", raw)
    urls = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.startswith(("http://", "https://")):
            p = "https://" + p
        urls.append(p)
    return urls


def _scrape_result_to_dict(result: ScrapeResult) -> dict:
    return {
        "url": result.url,
        "page_type": result.page_type,
        "best_strategy": result.best_strategy,
        "title": result.title,
        "content": result.content,
        "word_count": result.word_count,
        "tables_md": result.tables_md,
        "table_count": result.table_count,
        "fetch_time_ms": result.fetch_time_ms,
        "total_time_ms": result.total_time_ms,
        "headlines": result.headlines,
        "all_results": [{
            "strategy": r.strategy,
            "success": r.success,
            "title": r.title,
            "word_count": r.word_count,
            "time_ms": round(r.time_ms, 1),
            "error": r.error,
            "extra": {k: v for k, v in r.extra.items() if k != "rendered_html"},
        } for r in result.all_results],
        "metadata": result.metadata,
    }


@app.post("/api/scrape", response_model=ScrapeResponse)
async def api_scrape(req: ScrapeRequest):
    t0 = time.perf_counter()
    urls = _parse_urls(req.url)
    store = get_store()
    filter_phrase = (req.filter_phrase or "").strip() or None

    # ── Single URL (existing behaviour) ──────────────────────────────────────
    if len(urls) == 1:
        result = await _full_scrape(
            urls[0],
            strategies=req.strategies,
            wait_selector=req.wait_for_selector,
            wait_seconds=req.wait_seconds or 4.0,
        )
        # Persist to history
        store.record_url(urls[0], result.title or "", result.page_type)

        # ── Apply GenAI filter ────────────────────────────────────────────────
        filter_res = await apply_filter(
            phrase=filter_phrase or "",
            page_type=result.page_type,
            headlines=result.headlines,
            tables_md=result.tables_md,
            content=result.content,
        ) if filter_phrase else None

        d = _scrape_result_to_dict(result)
        if filter_res and filter_res.filter_applied:
            d["headlines"] = filter_res.headlines
            if filter_res.content is not None:
                d["tables_md"] = filter_res.content

        return ScrapeResponse(
            **{k: d[k] for k in ScrapeResponse.model_fields if k in d},
            multi_results=[],
            filter_phrase=filter_phrase,
            filter_applied=bool(filter_phrase),
            filter_insight=filter_res.insight if filter_res else "",
        )

    # ── Multiple URLs — scrape in parallel ────────────────────────────────────
    # For multi-URL mode: exclude Playwright by default (too slow/CPU-heavy in parallel).
    # User can re-enable it via advanced options if needed for a specific URL set.
    requested_strategies = req.strategies or STRATEGY_ORDER
    multi_strategies = [s for s in requested_strategies if s != "playwright"]                        or ["trafilatura", "newspaper3k", "readability", "beautifulsoup"]

    # Per-URL timeout: cap each scrape so one slow site can't block the rest
    PER_URL_TIMEOUT = 45  # seconds

    async def _scrape_one(url: str) -> dict:
        try:
            r = await asyncio.wait_for(
                _full_scrape(
                    url,
                    strategies=multi_strategies,
                    wait_selector=req.wait_for_selector,
                    wait_seconds=min(req.wait_seconds or 4.0, 6.0),
                ),
                timeout=PER_URL_TIMEOUT,
            )
            store.record_url(url, r.title or "", r.page_type)
            return _scrape_result_to_dict(r)
        except asyncio.TimeoutError:
            return {"url": url, "error": f"Timed out after {PER_URL_TIMEOUT}s",
                    "page_type": "unknown", "title": None, "content": None,
                    "word_count": 0, "headlines": [], "tables_md": None,
                    "table_count": 0, "all_results": [], "metadata": {},
                    "best_strategy": None, "fetch_time_ms": 0, "total_time_ms": 0}
        except Exception as e:
            return {"url": url, "error": str(e), "page_type": "unknown",
                    "title": None, "content": None, "word_count": 0,
                    "headlines": [], "tables_md": None, "table_count": 0,
                    "all_results": [], "metadata": {}, "best_strategy": None,
                    "fetch_time_ms": 0, "total_time_ms": 0}

    multi_results = list(await asyncio.gather(*[_scrape_one(u) for u in urls]))

    # Apply GenAI filter across all results
    combined_insight = ""
    if filter_phrase:
        for mr in multi_results:
            fr = await apply_filter(
                phrase=filter_phrase,
                page_type=mr.get("page_type", "unknown"),
                headlines=mr.get("headlines"),
                tables_md=mr.get("tables_md"),
                content=mr.get("content"),
            )
            if fr.filter_applied:
                mr["headlines"] = fr.headlines
                if fr.content is not None:
                    mr["tables_md"] = fr.content
                if fr.insight:
                    combined_insight += f"**{mr.get('url','')}**: {fr.insight}\n\n"

    # Use first successful result as the primary response
    primary = next(
        (r for r in multi_results if r.get("content") or r.get("headlines")),
        multi_results[0],
    )
    total_ms = (time.perf_counter() - t0) * 1000

    return ScrapeResponse(
        url=", ".join(urls),
        page_type=primary.get("page_type", "unknown"),
        best_strategy=primary.get("best_strategy"),
        title=primary.get("title"),
        content=primary.get("content"),
        word_count=primary.get("word_count", 0),
        tables_md=primary.get("tables_md"),
        table_count=primary.get("table_count", 0),
        fetch_time_ms=primary.get("fetch_time_ms", 0),
        total_time_ms=round(total_ms, 1),
        headlines=primary.get("headlines", []),
        all_results=primary.get("all_results", []),
        metadata=primary.get("metadata", {}),
        multi_results=multi_results,
        filter_phrase=filter_phrase,
        filter_applied=bool(filter_phrase),
        filter_insight=combined_insight.strip(),
    )


# ── API: Async headline filter ───────────────────────────────────────────────

class FilterRequest(BaseModel):
    headlines: list[dict]        # [{title, url, section, summary}]
    phrase: str
    page_type: str = "homepage"
    tables_md: Optional[str] = None


class FilterResponse(BaseModel):
    scored: list[dict]           # original items with added _score and _matched fields
    insight: str = ""
    intent_type: str = ""
    refined_query: str = ""


_BATCH_SCORE_PROMPT = """You are a headline relevance scorer.

User filter: "{phrase}"
Required named entities (must be present): {named_entities}
Match condition: {condition}
Expanded terms: {refined_query}
Strict mode: {strict}

Scoring rules:
- If named_entities is non-empty and headline does NOT mention any of them → score 0 (hard rule)
- 9-10: Directly about the asked topic/place/event
- 6-8:  Clearly related — same region, event, theme
- 3-5:  Tangentially related — same broad topic, different angle  
- 0-2:  Unrelated or missing required named entity

Use BOTH title and section field to judge.
Geographic context: parent district/state of a city counts as related (score 5-6).

Headlines (title + section):
{headlines_json}

Reply with ONLY a JSON array of integers (0-10), same order as input.
Example: [9, 0, 5, 7]
"""


@app.post("/api/filter", response_model=FilterResponse)
async def api_filter(req: FilterRequest):
    """
    Two-phase async filter:
    Phase 1: instant regex scoring (returned immediately via first call)
    Phase 2: LLM batch scoring for semantic accuracy
    This endpoint does both and returns combined scores.
    """
    from scraper.filter import classify_intent, _regex_score, _expand_token, _SYNONYMS
    import json as _json

    phrase = req.phrase.strip()
    if not phrase:
        scored = [dict(h, _score=1.0, _matched=True) for h in req.headlines]
        return FilterResponse(scored=scored)

    # ── Phase 1: instant regex scoring ───────────────────────────────────────
    tokens = phrase.lower().split()

    def regex_score(item: dict) -> float:
        text = " ".join([
            item.get("title",""), item.get("url",""),
            item.get("section",""), item.get("summary","")
        ]).lower()
        hits = sum(1 for t in tokens
                   if any(syn in text for syn in _expand_token(t)))
        return hits / len(tokens) if tokens else 0.0

    regex_scores = {i: regex_score(h) for i, h in enumerate(req.headlines)}

    # ── Intent classification ─────────────────────────────────────────────────
    intent = await classify_intent(phrase)

    # ── Named Entity gate ─────────────────────────────────────────────────────
    named_entities = getattr(intent, "named_entities", []) or []
    ne_lower = [ne.lower() for ne in named_entities if ne]

    def passes_ne_gate(item: dict) -> bool:
        if not ne_lower:
            return True
        text = (item.get("title","") + " " + item.get("url","") + " " + item.get("section","")).lower()
        return any(ne in text for ne in ne_lower)

    # Only apply NE gate if at least ONE headline passes it.
    # If zero headlines contain the NE (e.g. "paytm" on a general market page),
    # the gate would kill everything — disable it and let LLM semantic scoring decide.
    ne_gate_active = ne_lower and any(passes_ne_gate(h) for h in req.headlines)

    if ne_gate_active:
        for i, h in enumerate(req.headlines):
            if not passes_ne_gate(h):
                regex_scores[i] = 0.0

    # ── Phase 2: LLM batch scoring ────────────────────────────────────────────
    llm_scores: dict[int, float] = {}
    try:
        ollama = get_ollama_client()
        headlines_json = _json.dumps([
            {"index": i, "title": h.get("title",""), "section": h.get("section","")}
            for i, h in enumerate(req.headlines)
        ], indent=2)

        ne_list = getattr(intent, "named_entities", []) or []
        prompt = _BATCH_SCORE_PROMPT.format(
            phrase=phrase,
            named_entities=", ".join(ne_list) if ne_list else "none",
            condition=intent.condition or f"content related to: {phrase}",
            refined_query=intent.refined_query,
            strict=getattr(intent, "strict", False),
            headlines_json=headlines_json,
        )
        response = await asyncio.wait_for(
            ollama.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            ),
            timeout=25.0,
        )
        import re as _re
        clean = _re.sub(r"```(?:json)?|```", "", response).strip()
        raw_scores = _json.loads(clean)
        if isinstance(raw_scores, list):
            for i, s in enumerate(raw_scores):
                if i < len(req.headlines):
                    score = float(s) / 10.0
                    # NE gate: only hard-zero if gate is active (NEs found in at least one headline)
                    if ne_gate_active and not passes_ne_gate(req.headlines[i]):
                        score = 0.0
                    llm_scores[i] = score
    except Exception:
        pass  # fall back to regex only

    # ── Combine scores ────────────────────────────────────────────────────────
    # LLM score weighted 70%, regex 30% when available; regex only otherwise
    scored_items = []
    for i, h in enumerate(req.headlines):
        rscore = regex_scores.get(i, 0.0)
        lscore = llm_scores.get(i, None)

        if lscore is not None:
            combined = rscore * 0.15 + lscore * 0.85
        else:
            combined = rscore

        matched = combined >= (0.45 if lscore is not None else 0.30)
        scored_items.append(dict(h, _score=round(combined, 3), _matched=matched))

    # Sort matched first, then by score descending
    scored_items.sort(key=lambda x: (not x["_matched"], -x["_score"]))

    # ── Generate insight for data pages ──────────────────────────────────────
    insight = ""
    if req.page_type == "data" and req.tables_md:
        from scraper.filter import filter_table_rows
        _, insight = await filter_table_rows(req.tables_md, intent)

    return FilterResponse(
        scored=scored_items,
        insight=insight,
        intent_type=intent.intent_type,
        refined_query=intent.refined_query,
    )


# ── API: URL history for autocomplete ────────────────────────────────────────

@app.get("/api/url-history")
async def url_history(q: str = "", limit: int = 20):
    store = get_store()
    return store.get_url_history(prefix=q, limit=limit)


# ── API: Dig into selected headlines ─────────────────────────────────────────

class DigRequest(BaseModel):
    urls: list[str]
    strategies: Optional[list[str]] = None


class DigResult(BaseModel):
    url: str
    title: Optional[str]
    content: Optional[str]
    word_count: int
    best_strategy: Optional[str]
    error: Optional[str] = None


@app.post("/api/dig")
async def api_dig(req: DigRequest) -> list[DigResult]:
    """Scrape multiple article URLs (from headline selection) concurrently."""
    strategies = req.strategies or ["trafilatura", "newspaper3k",
                                     "readability", "goose3", "beautifulsoup"]

    async def scrape_one(url: str) -> DigResult:
        try:
            result = await _full_scrape(url, strategies=strategies)
            return DigResult(
                url=url,
                title=result.title,
                content=result.content,
                word_count=result.word_count,
                best_strategy=result.best_strategy,
            )
        except Exception as e:
            return DigResult(url=url, title=None, content=None,
                             word_count=0, best_strategy=None,
                             error=str(e)[:200])

    results = await asyncio.gather(*[scrape_one(u) for u in req.urls[:20]])
    return list(results)


# ── API: Ingest for RAG ───────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    documents: list[dict]   # [{url, title, content, page_type}]


@app.post("/api/ingest")
async def api_ingest(req: IngestRequest):
    try:
        ctx = await ingest_documents(req.documents)
        return {
            "session_id": ctx.session_id,
            "mode": ctx.mode,
            "total_words": ctx.total_words,
            "chunk_count": ctx.chunk_count,
            "ready": ctx.ready,
            "error": ctx.error,
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── API: Q&A ──────────────────────────────────────────────────────────────────

class QARequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 6


@app.post("/api/qa")
async def api_qa(req: QARequest):
    try:
        result = await rag_query(req.session_id, req.question, req.top_k)
        return {
            "question": result.question,
            "answer": result.answer,
            "mode": result.mode,
            "sources_used": result.sources_used,
            "chunks_retrieved": result.chunks_retrieved,
            "latency_ms": round(result.latency_ms, 1),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── API: Q&A Streaming ────────────────────────────────────────────────────────

@app.get("/api/qa/stream")
async def api_qa_stream(session_id: str, question: str, top_k: int = 6):
    """SSE streaming endpoint for Q&A responses."""
    from rag.pipeline import get_store
    store = get_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    ollama = get_ollama_client()
    mode = session["mode"]

    # Build context (same logic as rag_query but streaming the answer)
    if mode == "small":
        chunks = store.get_all_chunks(session_id)
        context_text = "\n\n---\n\n".join(
            f"[{c['source_title'] or c['source_url']}]\n{c['content']}"
            for c in chunks
        )
    else:
        try:
            q_emb = await ollama.embed([question])
            relevant = store.similarity_search(session_id, q_emb[0], top_k=top_k)
            context_text = "\n\n---\n\n".join(
                f"[{r['source_title']} | {r.get('section','')}]\n{r['content']}"
                for r in relevant
            )
        except Exception:
            chunks = store.get_all_chunks(session_id)
            context_text = "\n\n".join(c["content"] for c in chunks[:top_k])

    messages = [
        {"role": "system", "content": (
            "You are a precise research assistant. Answer based strictly on the context provided.\n\n"
            "CITATION RULES:\n"
            "- Every factual claim MUST be followed by a source link.\n"
            "- Citation format: [Source Title](URL) — Markdown link syntax.\n"
            "- Extract the URL from [Source: title | URL: ...] markers in context.\n"
            "- Never make claims without citing a source from the provided context.\n\n"
            "FORMATTING RULES:\n"
            "- Always respond in clean HTML (no markdown, no triple backticks).\n"
            "- Use <table> with <thead>/<tbody>/<tr>/<th>/<td> for tabular data.\n"
            "- Use <ul>/<li> or <ol>/<li> for lists.\n"
            "- Use <strong> for bold, <em> for italic.\n"
            "- Use <p> for paragraphs. Use <h3> or <h4> for section headings.\n"
            "- Do NOT include <html>, <head>, <body>, <style>, or <script> tags.\n"
            "- Do NOT use inline styles or class attributes.\n"
            "- Keep the HTML clean and semantic."
        )},
        {"role": "user", "content":
         f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"},
    ]

    async def event_stream():
        async for token in ollama.chat_stream(messages):
            # SSE spec: multi-line data is sent as multiple "data:" lines per event.
            # The browser EventSource joins them with \n automatically.
            # This correctly preserves newlines WITHOUT any encoding.
            lines = token.split("\n")
            data_lines = "\n".join(f"data: {l}" for l in lines)
            yield f"{data_lines}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── API: Ollama health ────────────────────────────────────────────────────────

@app.get("/api/health/ollama")
async def ollama_health():
    ollama = get_ollama_client()
    return {
        "inference": await ollama.check_inference(),
        "embeddings": await ollama.check_embed(),
    }


# ── Portfolio routes ─────────────────────────────────────────────────────────

@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):
    store = get_store()
    holdings = store.get_portfolio()
    url_history = store.get_url_history(limit=50)
    return templates.TemplateResponse(request, "portfolio.html", {
        "request": request, "title": cfg.app_title,
        "holdings": holdings, "url_history": url_history,
    })

class HoldingRequest(BaseModel):
    symbol: str
    name: str
    exchange: str = "NSE"
    sector: str = ""
    qty: float = 0
    avg_price: float = 0
    notes: str = ""
    id: Optional[int] = None

@app.post("/api/portfolio")
async def save_holding(req: HoldingRequest):
    store = get_store()
    store.upsert_holding(
        req.symbol.upper(), req.name, req.qty, req.avg_price,
        req.sector, req.exchange, req.notes, req.id
    )
    return {"status": "ok"}

@app.delete("/api/portfolio/{holding_id}")
async def delete_holding(holding_id: int):
    get_store().delete_holding(holding_id)
    return {"status": "ok"}

@app.get("/api/portfolio")
async def get_portfolio():
    return get_store().get_portfolio()

class UrlDailyRequest(BaseModel):
    url: str
    is_daily: bool = False

@app.post("/api/url-daily")
async def set_url_daily(req: UrlDailyRequest):
    get_store().set_url_daily(req.url, req.is_daily)
    return {"status": "ok"}

# ── Morning brief routes ──────────────────────────────────────────────────────

@app.get("/morning-brief", response_class=HTMLResponse)
async def morning_brief_page(request: Request, date: Optional[str] = None,
                              session_id: Optional[str] = None):
    store = get_store()
    from datetime import date as _date
    view_date = date or _date.today().isoformat()
    today_brief = store.get_brief(view_date)
    insights = store.get_insights(view_date)
    archive = store.get_recent_briefs(limit=10)
    from jobs.scheduler import _scheduler
    next_run = None
    if _scheduler:
        job = _scheduler.get_job("morning_brief")
        if job and job.next_run_time:
            next_run = job.next_run_time.strftime("%a %d %b %I:%M %p %Z")
    return templates.TemplateResponse(request, "morning_brief.html", {
        "request": request, "title": cfg.app_title,
        "today_brief": today_brief, "insights": insights,
        "archive": archive, "next_run": next_run, "cfg": cfg,
        "auto_run": cfg.brief_auto_run,
    })

@app.post("/api/morning-brief/run")
async def trigger_brief():
    """Manually trigger the morning brief (runs in background)."""
    import asyncio, logging
    from jobs.morning_brief import run_morning_brief
    _log = logging.getLogger("morning_brief")

    async def _run():
        print("[BRIEF TASK] Starting _run()", flush=True)
        try:
            result = await run_morning_brief(force=True)
            print(f"[BRIEF TASK] Finished: {result}", flush=True)
            _log.info("Brief finished: %s", result)
        except Exception as e:
            import traceback
            print(f"[BRIEF TASK] EXCEPTION: {traceback.format_exc()}", flush=True)
            _log.error("Brief task exception:\n%s", traceback.format_exc())
            from datetime import date
            from rag.pipeline import get_store
            get_store().finish_brief(date.today().isoformat(), "", "", 0, str(e))

    print("[BRIEF] create_task called", flush=True)
    asyncio.create_task(_run())
    return {"status": "started"}

@app.get("/api/morning-brief/status")
async def brief_status():
    from datetime import date as _date
    store = get_store()
    today = _date.today().isoformat()
    brief = store.get_brief(today)
    return brief or {"status": "not_run", "date": today}

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "3.0.0",
            "strategies": STRATEGY_ORDER}