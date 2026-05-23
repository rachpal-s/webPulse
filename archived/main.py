"""
WebPulse Multi-Strategy Web Scraper API v3
- Fixed: NoneType crash in BS4 (iterator invalidation when parent decomposed)
- Fixed: Playwright gets proxy html -> graceful fallback
- New: XHR/fetch interception in Playwright to capture live API data
- New: Smart table extraction from JS-rendered pages
"""

import asyncio
import json
import re
import time
from typing import Optional
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup, Tag
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from goose3 import Goose
from newspaper import Article
from pydantic import BaseModel
from readability import Document

app = FastAPI(title="WebPulse Scraper", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Constants ─────────────────────────────────────────────────────────────────

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Platform": '"Windows"',
}
TIMEOUT = 28
ALL_STRATEGIES = ["playwright", "trafilatura", "readability", "goose3", "beautifulsoup", "newspaper3k"]

# ── Models ────────────────────────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    url: str
    strategies: Optional[list[str]] = None
    wait_for_selector: Optional[str] = None
    wait_seconds: Optional[float] = 4.0

class StrategyResult(BaseModel):
    strategy: str
    success: bool
    title: Optional[str] = None
    text: Optional[str] = None
    word_count: int = 0
    time_ms: float = 0
    error: Optional[str] = None
    extra: Optional[dict] = None  # captured XHR data, table counts, etc.

class ScrapeResponse(BaseModel):
    url: str
    best_strategy: Optional[str]
    title: Optional[str]
    content: Optional[str]
    word_count: int
    fetch_time_ms: float
    total_time_ms: float
    all_results: list[StrategyResult]
    metadata: dict

# ── Text cleaner ──────────────────────────────────────────────────────────────

JUNK_LINE = re.compile(
    r"(accept all cookies|cookie policy|privacy policy|subscribe now|sign up free|"
    r"log in to|advertisement|follow us on|share this article|you might also like|"
    r"trending now|buy now|shop now|click here|sponsored by|promoted content|"
    r"newsletter signup|enable javascript|please enable js|loading\.\.\.|"
    r"please wait|fetching data)",
    re.I,
)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or len(s) < 3:
            continue
        if len(s) < 120 and JUNK_LINE.search(s):
            continue
        lines.append(s)
    result = "\n".join(lines).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result

# ── HTML fetcher ──────────────────────────────────────────────────────────────

IS_PROXY_ERROR = re.compile(r"host not in allowlist|access denied|403 forbidden", re.I)

async def fetch_html(url: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    async with httpx.AsyncClient(
        headers=BROWSER_HEADERS, follow_redirects=True, timeout=TIMEOUT
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
    html = r.text
    if IS_PROXY_ERROR.search(html[:300]):
        raise ValueError(f"Blocked by proxy/firewall: {html[:100]}")
    return html, (time.perf_counter() - t0) * 1000

# ── BS4 safe extractor (fixes NoneType crash) ─────────────────────────────────

NOISE_TAGS = [
    "script", "style", "nav", "header", "footer", "aside", "noscript",
    "form", "iframe", "picture", "svg", "button", "input", "select",
    "textarea", "ins", "figure", "figcaption",
]
AD_PAT = re.compile(
    r"(^ad[-_]|[-_]ad$|^ads$|advertisement|sponsor|promo|banner|cookie-|"
    r"popup|modal|overlay|^sidebar|widget-ad|social-share|share-bar|"
    r"newsletter|subscribe-box|taboola|outbrain|gpt-ad|dfp-|"
    r"interstitial|sticky-ad|floating-ad|lightbox)",
    re.I,
)

def safe_bs4_extract(html: str, strategy_name: str = "beautifulsoup") -> StrategyResult:
    """
    Bug-fixed BeautifulSoup extractor.
    Key fix: use list() on find_all() before decomposing, and check isinstance(el, Tag)
    to avoid NoneType crash when parent elements are decomposed mid-iteration.
    """
    t0 = time.perf_counter()
    try:
        soup = BeautifulSoup(html, "lxml")

        # Check for proxy/firewall block pages
        body_text = soup.get_text()[:300]
        if IS_PROXY_ERROR.search(body_text):
            return StrategyResult(
                strategy=strategy_name, success=False,
                error="Blocked by proxy or firewall",
                time_ms=(time.perf_counter() - t0) * 1000,
            )

        # Remove noise tags
        for tag in soup(NOISE_TAGS):
            tag.decompose()

        # ── THE FIX: list() prevents iterator invalidation ──────────────────
        # Without list(), decomposing a parent element corrupts the iterator,
        # causing child elements to become detached (parent=None), and then
        # el.get('class') raises AttributeError: 'NoneType' has no attribute 'get'
        for el in list(soup.find_all(True)):
            if not isinstance(el, Tag):       # skip NavigableString & other non-Tag nodes
                continue
            if el.parent is None:             # already decomposed (detached from tree)
                continue
            cls = " ".join(el.get("class") or [])
            eid = el.get("id") or ""
            if AD_PAT.search(cls) or AD_PAT.search(eid):
                el.decompose()
        # ────────────────────────────────────────────────────────────────────

        # Resolve title safely
        h1 = soup.find("h1")
        title_tag = soup.find("title")
        title = (h1.get_text(strip=True) if h1 else None) or (title_tag.get_text(strip=True) if title_tag else None)

        # Score all candidate containers by raw text length; pick the longest
        best_el: Optional[Tag] = None
        best_len = 0
        for c in soup.find_all(["article", "main", "section", "div"]):
            if not isinstance(c, Tag) or c.parent is None:
                continue
            t = c.get_text(separator=" ", strip=True)
            if len(t) > best_len:
                best_len = len(t)
                best_el = c

        content_el = best_el or soup.body
        if content_el is None:
            return StrategyResult(
                strategy=strategy_name, success=False,
                error="No content container found",
                time_ms=(time.perf_counter() - t0) * 1000,
            )

        text = clean_text(content_el.get_text(separator="\n"))
        if not text or len(text.split()) < 10:
            return StrategyResult(
                strategy=strategy_name, success=False,
                error="Extracted text too short or empty",
                time_ms=(time.perf_counter() - t0) * 1000,
            )

        return StrategyResult(
            strategy=strategy_name, success=True,
            title=title, text=text, word_count=len(text.split()),
            time_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return StrategyResult(
            strategy=strategy_name, success=False, error=str(e),
            time_ms=(time.perf_counter() - t0) * 1000,
        )

# ── Table extractor ───────────────────────────────────────────────────────────

def extract_tables_as_markdown(html: str) -> tuple[str, int]:
    """Extract all data tables from HTML as clean Markdown. Returns (text, table_count)."""
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    out = []
    table_count = 0
    for table in tables:
        if not isinstance(table, Tag):
            continue
        # Header
        thead = table.find("thead")
        headers = []
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]

        # Rows
        tbody = table.find("tbody") or table
        rows = []
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)

        if not rows:
            continue

        table_count += 1
        # Try to infer a caption/title
        caption = table.find("caption")
        prev_heading = None
        for sib in table.find_all_previous(["h1", "h2", "h3", "h4"]):
            prev_heading = sib.get_text(strip=True)
            break
        if caption:
            out.append(f"### {caption.get_text(strip=True)}")
        elif prev_heading:
            out.append(f"### {prev_heading}")

        # Normalise row widths
        max_cols = max(len(r) for r in rows)
        if headers and len(headers) != max_cols:
            headers = headers[:max_cols] or [f"Col{i+1}" for i in range(max_cols)]

        if headers:
            out.append("| " + " | ".join(headers) + " |")
            out.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for r in rows:
                padded = r + [""] * (len(headers) - len(r))
                out.append("| " + " | ".join(padded[:len(headers)]) + " |")
        else:
            for r in rows:
                out.append("| " + " | ".join(r) + " |")
        out.append("")

    return "\n".join(out).strip(), table_count

# ── Playwright strategy ───────────────────────────────────────────────────────

def run_playwright(url: str, wait_selector: Optional[str], extra_wait: float) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

        captured_xhr: list[dict] = []          # intercept JSON API responses
        captured_json_texts: list[str] = []    # raw JSON from XHR

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-web-security",
                ],
            )
            ctx = browser.new_context(
                user_agent=UA,
                viewport={"width": 1440, "height": 900},
                locale="en-US",
                timezone_id="Asia/Kolkata",
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Sec-Ch-Ua": '"Chromium";v="124"',
                    "Sec-Ch-Ua-Platform": '"Windows"',
                },
            )
            page = ctx.new_page()

            # Stealth patches
            page.add_init_script("""
                try {
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    window.chrome = { runtime: {} };
                    Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
                    Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
                } catch(e) {}
            """)

            # ── XHR/fetch interception ────────────────────────────────────────
            def handle_response(response):
                try:
                    ct = response.headers.get("content-type", "")
                    u = response.url
                    # Capture JSON API responses that look like data feeds
                    if (
                        response.status == 200
                        and "json" in ct
                        and any(k in u.lower() for k in [
                            "api", "indices", "market", "stock", "quote",
                            "data", "feed", "live", "price", "rate", "chart",
                        ])
                        and "google" not in u
                        and "analytics" not in u
                        and "ad" not in u.lower().split("/")[-1][:3]
                    ):
                        try:
                            body = response.body()
                            if body and len(body) > 50:
                                text = body.decode("utf-8", errors="replace")
                                captured_xhr.append({"url": u, "size": len(body)})
                                captured_json_texts.append(text)
                        except Exception:
                            pass
                except Exception:
                    pass

            page.on("response", handle_response)
            # ─────────────────────────────────────────────────────────────────

            resp = page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT * 1000)

            # Check for proxy/firewall block
            status = resp.status if resp else 0
            if status in (403, 407, 451):
                browser.close()
                return StrategyResult(
                    strategy="playwright", success=False,
                    error=f"Server blocked the request (HTTP {status}). "
                          "Site may require cookies/login or uses aggressive bot detection.",
                    time_ms=(time.perf_counter() - t0) * 1000,
                )

            # Wait for specific element if given
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=12000)
                except PWTimeout:
                    pass

            # Wait for JS rendering
            if extra_wait and extra_wait > 0:
                page.wait_for_timeout(int(extra_wait * 1000))

            # Try to settle network
            try:
                page.wait_for_load_state("networkidle", timeout=6000)
            except PWTimeout:
                pass

            # Scroll to trigger lazy-loaded content
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                page.wait_for_timeout(1000)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(800)
            except Exception:
                pass

            html = page.content()
            title = page.title() or ""
            browser.close()

        # ── Check if we got a proxy block page ───────────────────────────────
        if IS_PROXY_ERROR.search(html[:400]):
            return StrategyResult(
                strategy="playwright", success=False,
                error="Blocked by network proxy or firewall. "
                      "The target site is not reachable from this server.",
                time_ms=(time.perf_counter() - t0) * 1000,
            )

        # ── Build content from multiple sources ───────────────────────────────
        content_parts = []
        extra_info: dict = {}

        # 1. Captured XHR/JSON API data (best source for financial sites)
        if captured_json_texts:
            extra_info["xhr_captured"] = len(captured_xhr)
            extra_info["xhr_urls"] = [x["url"][:80] for x in captured_xhr[:5]]
            json_content = format_captured_json(captured_json_texts, url)
            if json_content:
                content_parts.append("## Live Data (from API calls)\n\n" + json_content)

        # 2. Tables from rendered HTML (second-best for data pages)
        table_md, table_count = extract_tables_as_markdown(html)
        if table_md and table_count > 0:
            extra_info["tables_found"] = table_count
            # Only add if tables have actual data (not just headers)
            if len(table_md.split("\n")) > 4:
                content_parts.append("## Table Data\n\n" + table_md)

        # 3. Text content via trafilatura on rendered HTML
        traf = trafilatura.extract(html, include_tables=True, favor_recall=True, include_links=False)
        traf_clean = clean_text(traf) if traf else ""
        if traf_clean and len(traf_clean.split()) > 20:
            content_parts.append("## Page Content\n\n" + traf_clean)

        # 4. Fallback to BS4
        if not content_parts:
            bs_result = safe_bs4_extract(html, strategy_name="playwright-bs4")
            if bs_result.success and bs_result.text:
                content_parts.append(bs_result.text)

        if not content_parts:
            return StrategyResult(
                strategy="playwright", success=False,
                error="Page rendered but no extractable content found. "
                      "May require login, CAPTCHA, or further interaction.",
                time_ms=(time.perf_counter() - t0) * 1000,
                extra=extra_info,
            )

        combined = "\n\n---\n\n".join(content_parts)
        return StrategyResult(
            strategy="playwright", success=True,
            title=title or None,
            text=combined,
            word_count=len(combined.split()),
            time_ms=(time.perf_counter() - t0) * 1000,
            extra=extra_info,
        )

    except ImportError:
        return StrategyResult(
            strategy="playwright", success=False,
            error="playwright not installed. Run: pip install playwright && playwright install chromium",
            time_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return StrategyResult(
            strategy="playwright", success=False,
            error=str(e)[:300],
            time_ms=(time.perf_counter() - t0) * 1000,
        )


def format_captured_json(json_texts: list[str], source_url: str) -> str:
    """Turn captured XHR JSON payloads into readable text/tables."""
    out = []
    for raw in json_texts[:8]:
        try:
            data = json.loads(raw)
        except Exception:
            continue

        # Flatten common financial API response shapes
        rows = None
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            # Try common wrappers: data.data, data.result, data.indices, etc.
            for key in ["data", "result", "results", "indices", "items", "records",
                        "globalIndices", "indexData", "list", "response"]:
                val = data.get(key)
                if isinstance(val, list) and val:
                    rows = val
                    break
                if isinstance(val, dict):
                    # one more level
                    for k2 in ["data", "list", "items", "records"]:
                        v2 = val.get(k2)
                        if isinstance(v2, list) and v2:
                            rows = v2
                            break
                    if rows:
                        break

        if rows and isinstance(rows, list) and all(isinstance(r, dict) for r in rows[:3]):
            # Build markdown table
            # Use only columns with short scalar values (skip nested objects/arrays)
            sample = rows[0]
            cols = [k for k, v in sample.items()
                    if isinstance(v, (str, int, float, bool)) and len(str(v)) < 60][:12]
            if cols:
                out.append("| " + " | ".join(cols) + " |")
                out.append("| " + " | ".join(["---"] * len(cols)) + " |")
                for row in rows[:100]:
                    vals = [str(row.get(c, "")) for c in cols]
                    out.append("| " + " | ".join(vals) + " |")
                out.append("")
        elif isinstance(data, dict) and not rows:
            # Flat key-value dict — format as definition list
            for k, v in list(data.items())[:30]:
                if isinstance(v, (str, int, float)):
                    out.append(f"**{k}**: {v}")

    return "\n".join(out)

# ── Other strategies ──────────────────────────────────────────────────────────

def run_trafilatura(html: str, url: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        if IS_PROXY_ERROR.search(html[:300]):
            return StrategyResult(strategy="trafilatura", success=False,
                                  error="Blocked by proxy", time_ms=(time.perf_counter()-t0)*1000)
        text = trafilatura.extract(
            html, url=url, include_tables=True, include_links=False,
            include_images=False, no_fallback=False, favor_recall=True,
        )
        meta = trafilatura.extract_metadata(html, default_url=url)
        title = meta.title if meta else None
        text = clean_text(text)
        if not text or len(text.split()) < 10:
            return StrategyResult(strategy="trafilatura", success=False,
                                  error="No content extracted", time_ms=(time.perf_counter()-t0)*1000)
        return StrategyResult(strategy="trafilatura", success=True, title=title,
                              text=text, word_count=len(text.split()),
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult(strategy="trafilatura", success=False, error=str(e),
                              time_ms=(time.perf_counter()-t0)*1000)


def run_newspaper3k(url: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        article = Article(url, browser_user_agent=UA, request_timeout=TIMEOUT)
        article.download()
        article.parse()
        text = clean_text(article.text)
        if not text or len(text.split()) < 10:
            return StrategyResult(strategy="newspaper3k", success=False,
                                  error="No content extracted", time_ms=(time.perf_counter()-t0)*1000)
        return StrategyResult(strategy="newspaper3k", success=True, title=article.title,
                              text=text, word_count=len(text.split()),
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult(strategy="newspaper3k", success=False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


def run_readability(html: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        if IS_PROXY_ERROR.search(html[:300]):
            return StrategyResult(strategy="readability", success=False,
                                  error="Blocked by proxy", time_ms=(time.perf_counter()-t0)*1000)
        doc = Document(html)
        title = doc.title()
        soup = BeautifulSoup(doc.summary(), "lxml")
        text = clean_text(soup.get_text(separator="\n"))
        if not text or len(text.split()) < 10:
            return StrategyResult(strategy="readability", success=False,
                                  error="No content extracted", time_ms=(time.perf_counter()-t0)*1000)
        return StrategyResult(strategy="readability", success=True, title=title,
                              text=text, word_count=len(text.split()),
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult(strategy="readability", success=False, error=str(e),
                              time_ms=(time.perf_counter()-t0)*1000)


def run_goose3(url: str, html: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        if IS_PROXY_ERROR.search(html[:300]):
            return StrategyResult(strategy="goose3", success=False,
                                  error="Blocked by proxy", time_ms=(time.perf_counter()-t0)*1000)
        g = Goose({"browser_user_agent": UA})
        article = g.extract(url=url, raw_html=html)
        text = clean_text(article.cleaned_text)
        if not text or len(text.split()) < 10:
            return StrategyResult(strategy="goose3", success=False,
                                  error="No content extracted", time_ms=(time.perf_counter()-t0)*1000)
        return StrategyResult(strategy="goose3", success=True, title=article.title,
                              text=text, word_count=len(text.split()),
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult(strategy="goose3", success=False, error=str(e),
                              time_ms=(time.perf_counter()-t0)*1000)

# ── Metadata ──────────────────────────────────────────────────────────────────

def extract_metadata(html: str, url: str) -> dict:
    try:
        soup = BeautifulSoup(html, "lxml")
        meta = {}
        for tag in soup.find_all("meta"):
            if not isinstance(tag, Tag):
                continue
            prop = tag.get("property") or tag.get("name", "")
            val = tag.get("content", "")
            if prop and val:
                meta[prop] = val
        parsed = urlparse(url)
        return {
            "domain": parsed.netloc,
            "og_title": meta.get("og:title"),
            "og_description": meta.get("og:description"),
            "og_image": meta.get("og:image"),
            "og_type": meta.get("og:type"),
            "author": meta.get("author") or meta.get("article:author"),
            "published_time": meta.get("article:published_time"),
        }
    except Exception:
        return {"domain": urlparse(url).netloc}

# ── Pick best ─────────────────────────────────────────────────────────────────

def pick_best(results: list[StrategyResult]) -> Optional[StrategyResult]:
    successful = [r for r in results if r.success and r.text and len(r.text.split()) >= 10]
    if not successful:
        return None
    # Playwright wins if it got live XHR data or tables (bonus weight)
    def score(r: StrategyResult) -> float:
        base = r.word_count
        if r.strategy == "playwright" and r.extra:
            if r.extra.get("xhr_captured", 0) > 0:
                base *= 3   # XHR data is gold
            if r.extra.get("tables_found", 0) > 0:
                base *= 1.5
        return base
    return max(successful, key=score)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    with open("index.html") as f:
        return f.read()

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape(req: ScrapeRequest):
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    t_total = time.perf_counter()
    strategies = req.strategies or ALL_STRATEGIES

    # Fetch HTML for static strategies
    html = ""
    fetch_ms = 0.0
    need_html = any(s in strategies for s in ["trafilatura", "readability", "goose3", "beautifulsoup"])
    if need_html:
        try:
            html, fetch_ms = await fetch_html(url)
        except Exception as e:
            html = ""
            fetch_ms = 0.0
            if "playwright" not in strategies:
                raise HTTPException(status_code=502, detail=str(e))

    metadata = extract_metadata(html, url) if html else {"domain": urlparse(url).netloc}

    loop = asyncio.get_event_loop()
    futures = []

    if "playwright" in strategies:
        futures.append(loop.run_in_executor(
            None, run_playwright, url, req.wait_for_selector, req.wait_seconds or 4.0
        ))
    if html:
        if "trafilatura" in strategies:
            futures.append(loop.run_in_executor(None, run_trafilatura, html, url))
        if "readability" in strategies:
            futures.append(loop.run_in_executor(None, run_readability, html))
        if "goose3" in strategies:
            futures.append(loop.run_in_executor(None, run_goose3, url, html))
        if "beautifulsoup" in strategies:
            futures.append(loop.run_in_executor(None, safe_bs4_extract, html, "beautifulsoup"))
    if "newspaper3k" in strategies:
        futures.append(loop.run_in_executor(None, run_newspaper3k, url))

    results: list[StrategyResult] = list(await asyncio.gather(*futures))
    best = pick_best(results)
    total_ms = (time.perf_counter() - t_total) * 1000

    return ScrapeResponse(
        url=url,
        best_strategy=best.strategy if best else None,
        title=best.title if best else metadata.get("og_title"),
        content=best.text if best else None,
        word_count=best.word_count if best else 0,
        fetch_time_ms=round(fetch_ms, 1),
        total_time_ms=round(total_ms, 1),
        all_results=results,
        metadata=metadata,
    )

@app.get("/scrape", response_model=ScrapeResponse)
async def scrape_get(url: str, wait_seconds: float = 4.0):
    return await scrape(ScrapeRequest(url=url, wait_seconds=wait_seconds))

@app.get("/health")
async def health():
    try:
        from playwright.sync_api import sync_playwright
        pw_ok = True
    except ImportError:
        pw_ok = False
    return {"status": "ok", "playwright_available": pw_ok, "strategies": ALL_STRATEGIES, "version": "3.0.0"}
