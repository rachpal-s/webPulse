"""scraper/engine.py — Multi-strategy scraper, Trafilatura-first."""
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup, Tag
from goose3 import Goose
from newspaper import Article
from readability import Document

from config import get_settings

cfg = get_settings()

# ── Browser headers ───────────────────────────────────────────────────────────

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
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

IS_BLOCKED = re.compile(r"host not in allowlist|access denied|403 forbidden", re.I)

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    strategy: str
    success: bool
    title: Optional[str] = None
    text: Optional[str] = None
    word_count: int = 0
    time_ms: float = 0.0
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ScrapeResult:
    url: str
    page_type: str
    best_strategy: Optional[str]
    title: Optional[str]
    content: Optional[str]
    word_count: int
    tables_md: Optional[str]       # Markdown tables if data page
    table_count: int
    fetch_time_ms: float
    total_time_ms: float
    all_results: list[StrategyResult]
    metadata: dict
    headlines: list[dict] = field(default_factory=list)   # for homepage


# ── HTML fetcher ──────────────────────────────────────────────────────────────

async def fetch_html(url: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    async with httpx.AsyncClient(
        headers=HEADERS, follow_redirects=True,
        timeout=cfg.scraper_timeout,
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
    html = r.text
    if IS_BLOCKED.search(html[:300]):
        raise ValueError(f"Blocked: {html[:120]}")
    return html, (time.perf_counter() - t0) * 1000


# ── Text cleaner ──────────────────────────────────────────────────────────────

JUNK_LINE = re.compile(
    r"(accept all cookies|cookie policy|subscribe now|sign up|log in|"
    r"advertisement|follow us on|share this|you might also like|"
    r"trending now|buy now|click here|sponsored|promoted content|"
    r"enable javascript|please wait|fetching data|loading\.\.\.)",
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
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ── Noise tags and ad patterns ────────────────────────────────────────────────

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


# ── Strategy 1: Trafilatura ───────────────────────────────────────────────────

def _trafilatura(html: str, url: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        text = trafilatura.extract(
            html, url=url,
            include_tables=True, include_links=False,
            include_images=False, no_fallback=False,
            favor_recall=True,
        )
        meta = trafilatura.extract_metadata(html, default_url=url)
        title = meta.title if meta else None
        text = clean_text(text)
        wc = len(text.split()) if text else 0
        if not text or wc < 20:
            return StrategyResult("trafilatura", False,
                                  error="Insufficient content",
                                  time_ms=(time.perf_counter()-t0)*1000)
        return StrategyResult("trafilatura", True, title=title,
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult("trafilatura", False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


# ── Strategy 2: Newspaper3k ───────────────────────────────────────────────────

def _newspaper3k(url: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        article = Article(url, browser_user_agent=UA,
                          request_timeout=cfg.scraper_timeout)
        article.download()
        article.parse()
        text = clean_text(article.text)
        wc = len(text.split()) if text else 0
        if not text or wc < 20:
            return StrategyResult("newspaper3k", False,
                                  error="Insufficient content",
                                  time_ms=(time.perf_counter()-t0)*1000)
        extra = {}
        if article.authors:
            extra["authors"] = article.authors
        if article.publish_date:
            extra["publish_date"] = str(article.publish_date)
        return StrategyResult("newspaper3k", True, title=article.title,
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter()-t0)*1000,
                              extra=extra)
    except Exception as e:
        return StrategyResult("newspaper3k", False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


# ── Strategy 3: Readability ───────────────────────────────────────────────────

def _readability(html: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "lxml")
        text = clean_text(soup.get_text(separator="\n"))
        wc = len(text.split()) if text else 0
        if not text or wc < 20:
            return StrategyResult("readability", False,
                                  error="Insufficient content",
                                  time_ms=(time.perf_counter()-t0)*1000)
        return StrategyResult("readability", True, title=doc.title(),
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult("readability", False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


# ── Strategy 4: Goose3 ────────────────────────────────────────────────────────

def _goose3(html: str, url: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        g = Goose({"browser_user_agent": UA})
        article = g.extract(url=url, raw_html=html)
        text = clean_text(article.cleaned_text)
        wc = len(text.split()) if text else 0
        if not text or wc < 20:
            return StrategyResult("goose3", False,
                                  error="Insufficient content",
                                  time_ms=(time.perf_counter()-t0)*1000)
        return StrategyResult("goose3", True, title=article.title,
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult("goose3", False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


# ── Strategy 5: BeautifulSoup ─────────────────────────────────────────────────

def _beautifulsoup(html: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        soup = BeautifulSoup(html, "lxml")
        if IS_BLOCKED.search(soup.get_text()[:300]):
            return StrategyResult("beautifulsoup", False,
                                  error="Blocked by proxy",
                                  time_ms=(time.perf_counter()-t0)*1000)

        for tag in soup(NOISE_TAGS):
            tag.decompose()

        # Safe iteration — list() prevents iterator invalidation on decompose
        for el in list(soup.find_all(True)):
            if not isinstance(el, Tag) or el.parent is None:
                continue
            cls = " ".join(el.get("class") or [])
            eid = el.get("id") or ""
            if AD_PAT.search(cls) or AD_PAT.search(eid):
                el.decompose()

        h1 = soup.find("h1")
        title_tag = soup.find("title")
        title = (h1.get_text(strip=True) if h1 else None) or \
                (title_tag.get_text(strip=True) if title_tag else None)

        best_el, best_len = None, 0
        for c in soup.find_all(["article", "main", "section", "div"]):
            if not isinstance(c, Tag) or c.parent is None:
                continue
            t = c.get_text(separator=" ", strip=True)
            if len(t) > best_len:
                best_len = len(t)
                best_el = c

        content_el = best_el or soup.body
        if content_el is None:
            return StrategyResult("beautifulsoup", False,
                                  error="No content container",
                                  time_ms=(time.perf_counter()-t0)*1000)

        text = clean_text(content_el.get_text(separator="\n"))
        wc = len(text.split()) if text else 0
        if not text or wc < 20:
            return StrategyResult("beautifulsoup", False,
                                  error="Insufficient content",
                                  time_ms=(time.perf_counter()-t0)*1000)

        return StrategyResult("beautifulsoup", True, title=title,
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult("beautifulsoup", False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


# ── Strategy 6: Playwright (fallback for JS-heavy pages) ─────────────────────

# Chromium launch args that reduce bot-detection surface
_PW_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--disable-extensions",
    "--disable-features=IsolateOrigins,site-per-process",
    "--window-size=1440,900",
]

# Realistic browser fingerprint headers
_PW_EXTRA_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Inline stealth JS — hides every known automation fingerprint
_STEALTH_JS = """
(function() {
    // 1. Hide webdriver flag
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

    // 2. Fake plugins (real Chrome has plugins; headless has none)
    Object.defineProperty(navigator, 'plugins', {
        get: () => {
            const arr = [
                { name:'Chrome PDF Plugin', filename:'internal-pdf-viewer', description:'Portable Document Format' },
                { name:'Chrome PDF Viewer', filename:'mhjfbmdgcfjbbpaeojofohoefgiehjai', description:'' },
                { name:'Native Client', filename:'internal-nacl-plugin', description:'' }
            ];
            arr.item = i => arr[i]; arr.namedItem = n => arr.find(p=>p.name===n) || null;
            arr.refresh = ()=>{};
            return arr;
        }
    });

    // 3. Real language lists
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });

    // 4. Fake chrome runtime (headless has no window.chrome by default)
    if (!window.chrome) {
        window.chrome = {
            app: { isInstalled: false, getDetails: ()=>{}, getIsInstalled: ()=>{}, runningState: ()=>'cannot_run' },
            runtime: { PlatformOs: {MAC:'mac',WIN:'win'}, PlatformArch: {ARM:'arm',X86_32:'x86-32',X86_64:'x86-64'},
                       PlatformNaclArch: {ARM:'arm',X86_32:'x86-32',X86_64:'x86-64'},
                       RequestUpdateCheckStatus: {THROTTLED:'throttled',NO_UPDATE:'no_update',UPDATE_AVAILABLE:'update_available'} },
            loadTimes: function() { return {
                commitLoadTime: Date.now()/1000 - Math.random()*2,
                connectionInfo:'h2', finishDocumentLoadTime:0, finishLoadTime:0,
                firstPaintAfterLoadTime:0, firstPaintTime:0, navigationType:'Other',
                npnNegotiatedProtocol:'h2', requestTime:Date.now()/1000 - Math.random()*3,
                startLoadTime:Date.now()/1000 - Math.random()*2, wasAlternateProtocolAvailable:false,
                wasFetchedViaSpdy:true, wasNpnNegotiated:true
            }},
            csi: function() { return { onloadT: Date.now(), pageT: Math.random()*5000, startE: Date.now()-3000, tran: 15 } }
        };
    }

    // 5. Correct permissions API
    const origQuery = window.navigator.permissions && window.navigator.permissions.query;
    if (origQuery) {
        window.navigator.permissions.query = params =>
            params.name === 'notifications'
                ? Promise.resolve({ state: Notification.permission })
                : origQuery(params);
    }

    // 6. Realistic screen dimensions
    Object.defineProperty(screen, 'availWidth', { get: () => 1440 });
    Object.defineProperty(screen, 'availHeight', { get: () => 900 });

    // 7. Prevent iframe detection
    Object.defineProperty(HTMLIFrameElement.prototype, 'contentWindow', {
        get: function() { return window; }
    });

    // 8. Realistic hardware concurrency and memory
    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
    if ('deviceMemory' in navigator)
        Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
})();
"""


def _launch_browser_stealth(p, headless: bool):
    """Launch Chromium with stealth settings. Applies playwright-stealth if available."""
    try:
        from playwright_stealth import Stealth
        Stealth().hook_playwright_context(p)
    except Exception:
        pass  # continue without stealth plugin if unavailable

    return p.chromium.launch(
        headless=headless,
        args=_PW_ARGS,
    )


def _playwright(url: str, wait_selector: Optional[str] = None,
                extra_wait: float = 4.0) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        import json as _json

        captured_json: list[str] = []
        captured_xhr: list[dict] = []

        with sync_playwright() as p:
            browser = _launch_browser_stealth(p, cfg.playwright_headless)
            ctx = browser.new_context(
                user_agent=UA,
                viewport={"width": 1440, "height": 900},
                locale="en-US",
                timezone_id="Asia/Kolkata",
                extra_http_headers=_PW_EXTRA_HEADERS,
            )
            page = ctx.new_page()

            # Apply comprehensive stealth JS before any page loads
            page.add_init_script(_STEALTH_JS)

            def _on_response(resp):
                try:
                    ct = resp.headers.get("content-type", "")
                    u = resp.url
                    if resp.status == 200 and "json" in ct and any(
                        k in u.lower() for k in ["api","data","feed","market","stock","indices","quote","rate","price","finance"]
                    ) and "analytics" not in u and "gtm" not in u                     and not _NAV_URL_PATTERNS.search(u):
                        body = resp.body()
                        if body and 50 < len(body) < 500_000:
                            captured_json.append(body.decode("utf-8", errors="replace"))
                            captured_xhr.append({"url": u, "size": len(body)})
                except Exception:
                    pass

            page.on("response", _on_response)
            resp = page.goto(url, wait_until="domcontentloaded",
                             timeout=cfg.scraper_timeout * 1000)

            nav_status = resp.status if resp else 0

            # On 403/407: still continue — some sites return 403 on first load
            # then redirect to a challenge page. Check content before giving up.
            if nav_status in (407, 451):
                browser.close()
                return StrategyResult("playwright", False,
                                      error=f"HTTP {nav_status} — proxy/legal block",
                                      time_ms=(time.perf_counter()-t0)*1000)

            # Wait for dynamic content
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=12000)
                except PWTimeout:
                    pass

            page.wait_for_timeout(int(extra_wait * 1000))
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except PWTimeout:
                pass

            # Scroll to trigger lazy-loaded rows
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                page.wait_for_timeout(1000)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(800)
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(500)
            except Exception:
                pass

            # Wait a bit more for any deferred XHR after scroll
            page.wait_for_timeout(1500)

            html = page.content()
            pw_title = page.title()
            browser.close()

        if IS_BLOCKED.search(html[:400]):
            return StrategyResult("playwright", False,
                                  error="Blocked by network proxy",
                                  time_ms=(time.perf_counter()-t0)*1000)

        parts = []
        # Store rendered HTML so main.py can re-run table extraction / page detection on it
        extra = {"rendered_html": html}

        # XHR JSON data (best for financial/data pages)
        if captured_json:
            json_content = _format_xhr_json(captured_json)
            if json_content:
                extra["xhr_captured"] = len(captured_xhr)
                extra["xhr_urls"] = [x["url"][:80] for x in captured_xhr[:5]]
                parts.append("## Live API Data\n\n" + json_content)
            # If ALL captured JSON was nav/schema noise, don't count it as captured

        # Tables from rendered HTML
        from scraper.cleaner import extract_tables_markdown
        table_md, tc = extract_tables_markdown(html)
        if table_md and tc > 0:
            extra["tables_found"] = tc
            parts.append("## Table Data\n\n" + table_md)

        # Text via trafilatura on rendered HTML
        traf = trafilatura.extract(html, include_tables=True, favor_recall=True)
        if traf:
            traf_clean = clean_text(traf)
            if len(traf_clean.split()) > 20:
                parts.append("## Page Content\n\n" + traf_clean)

        if not parts:
            bs = _beautifulsoup(html)
            if bs.success and bs.text:
                parts.append(bs.text)

        if not parts:
            return StrategyResult("playwright", False,
                                  error="Rendered but no extractable content",
                                  time_ms=(time.perf_counter()-t0)*1000,
                                  extra=extra)

        combined = "\n\n---\n\n".join(parts)
        return StrategyResult("playwright", True,
                              title=pw_title or None,
                              text=combined,
                              word_count=len(combined.split()),
                              time_ms=(time.perf_counter()-t0)*1000,
                              extra=extra)

    except ImportError:
        return StrategyResult("playwright", False,
                              error="Not installed. Run: pip install playwright && playwright install chromium",
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult("playwright", False, error=str(e)[:300],
                              time_ms=(time.perf_counter()-t0)*1000)


# Column names that indicate a navigation/menu payload — not market data
_NAV_COL_NAMES = frozenset({
    "url", "shorturl", "longurl", "href", "link", "path", "slug", "uri",
    "navitem", "menuitem", "l1", "l2", "l3", "l3navmenuitem", "l2navmenuitem",
    "seotitle", "seo_title", "meta_title", "canonical", "permalink",
    "category_slug", "subcategory", "breadcrumb", "anchor", "target",
    "pagetype", "page_type_nav", "templatetype",
})

# URL path patterns that suggest navigation/config APIs (not data feeds)
_NAV_URL_PATTERNS = re.compile(
    r"/(nav|menu|sitemap|breadcrumb|sidebar|header|footer|widget|"
    r"config|setting|layout|template|taxonomy|category-tree|l[123]menu)",
    re.I,
)

# Type-name strings used in API schema descriptors
_TYPE_NAMES = frozenset({
    "string", "str", "int", "integer", "float", "double", "decimal",
    "bool", "boolean", "number", "object", "array", "null", "none",
    "date", "datetime", "timestamp", "text", "varchar", "bigint",
})


def _is_schema_row(rows: list) -> bool:
    """Return True if this list looks like schema/navigation/config data, not real market data."""
    if not rows or not isinstance(rows[0], dict):
        return False
    sample = rows[0]
    col_names = set(k.lower() for k in sample.keys())

    # Single-column tables are almost always metadata (e.g. {"length": 16990})
    if len(col_names) <= 1:
        return True

    # Classic schema shape: {name/field/column + type} columns
    has_type_col = "type" in col_names or "dtype" in col_names or "datatype" in col_names
    has_name_col = bool(col_names & {"name", "field", "column", "key", "attribute", "param"})
    if has_type_col and has_name_col:
        return True

    # Values that are all type-name strings
    vals = [str(v).lower().strip() for v in sample.values()]
    type_hits = sum(1 for v in vals if v in _TYPE_NAMES)
    if type_hits / max(len(vals), 1) > 0.5:
        return True

    # Navigation/menu payload: columns look like URL routing data
    nav_hits = col_names & _NAV_COL_NAMES
    if len(nav_hits) >= 2:
        return True

    # All values look like URL paths (start with /)
    url_path_vals = sum(1 for v in vals if str(v).startswith("/") or str(v).startswith("http"))
    if url_path_vals / max(len(vals), 1) > 0.5:
        return True

    return False


def _score_section(rows: list) -> float:
    """Score a list of row-dicts by data richness. Returns -1 to skip."""
    if not rows or not isinstance(rows[0], dict):
        return -1.0
    if _is_schema_row(rows):
        return -1.0

    sample = rows[0]
    score = 0.0

    # Row count (capped) — more rows = better
    score += min(len(rows), 200) * 1.5

    # Column count — more columns = richer
    score += len(sample) * 4.0

    # Numeric values (prices, changes) — key signal for market data
    numeric = sum(
        1 for v in sample.values()
        if isinstance(v, (int, float))
        or (isinstance(v, str)
            and v.strip().replace(".", "").replace("-", "").replace("+", "").replace(",", "").isdigit()
            and v.strip() not in ("", "0"))
    )
    score += numeric * 8.0

    # Known market/financial column names
    _MARKET_COLS = {
        "name", "symbol", "ticker", "ltp", "price", "lastprice", "last",
        "chg", "change", "net_change", "chgper", "percent_change", "pctchange",
        "open", "high", "low", "close", "prevclose", "prev_close",
        "volume", "vol", "marketstate", "market_state", "time", "updated_at",
    }
    col_lower = {k.lower() for k in sample.keys()}
    score += len(col_lower & _MARKET_COLS) * 12.0

    return score



def _extract_json_sections(data, depth: int = 0) -> list:
    """
    Recursively extract (section_title, list_of_row_dicts) from any JSON shape.
    Specifically tuned for Moneycontrol Global Indices.
    """
    if depth > 4:
        return []
    results = []

    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], dict):
            # Check if items themselves contain list values (category containers)
            cat_key = next(
                (k for item in data[:3] for k, v in item.items()
                 if isinstance(v, list) and v and isinstance(v[0], dict)),
                None,
            )
            if cat_key:
                for item in data:
                    # 🚨 WE FOUND THE SECRET WORD: "heading" 🚨
                    title = str(
                        item.get("heading") or item.get("category") or 
                        item.get("name") or item.get("section") or 
                        item.get("type") or ""
                    )
                    
                    if title == "header": 
                        continue # Skip useless metadata
                        
                    rows = item.get(cat_key, [])
                    if rows and isinstance(rows[0], dict):
                        results.append((title, rows))
            else:
                results.append(("", data))

    elif isinstance(data, dict):
        # 1. Try standard wrapper keys first (The X-ray showed 'data' and 'success')
        for key in ["data", "result", "results", "indices", "items", "records", "list", "response", "globalIndices", "payload"]:
            val = data.get(key)
            if val is None:
                continue
            sub = _extract_json_sections(val, depth + 1)
            if sub:
                return sub

        # 2. dict-of-lists pattern
        list_children = {
            k: v for k, v in data.items()
            if isinstance(v, list) and v and isinstance(v[0], dict)
        }
        if list_children:
            for category, rows in list_children.items():
                if category == "header":
                    continue 
                    
                has_nested = any(isinstance(vv, list) for item in rows[:3] for vv in item.values())
                if has_nested:                     
                    sub = _extract_json_sections(rows, depth + 1)                     
                    if sub:                         
                        for sub_title, sub_rows in sub:
                            final_title = sub_title if sub_title else category
                            if final_title != "header":
                                results.append((final_title, sub_rows))          
                else:                     
                    results.append((category, rows))
            if results:
                return results

        # 3. dict-of-dicts: recurse one level
        dict_results = []
        for k, v in data.items():
            if isinstance(v, dict):
                sub = _extract_json_sections(v, depth + 1)
                if sub:
                    for sub_title, sub_rows in sub:
                        if sub_title == "header":
                            continue
                            
                        # If inner key is generic, map it back to the parent 
                        if sub_title in ["dataList", "data", ""]:
                            dict_results.append((k, sub_rows))
                        else:
                            dict_results.append((f"{k}_{sub_title}", sub_rows))
                            
        if dict_results:
            results.extend(dict_results)
            return results

    return results

def _format_xhr_json(json_texts: list[str]) -> str:
    """
    Convert captured XHR JSON payloads to clean Markdown tables.
    Handles flat lists, dict-wrapped lists, and dict-of-lists (US/EUROPE/ASIA).
    Deduplicates by column fingerprint to avoid showing same data twice.
    """
    import json as _json
    out = []
    seen_fingerprints: set = set()

    # Parse all payloads and collect scored sections
    scored: list[tuple[float, str, list, int]] = []  # (score, title, rows, payload_id)

    for payload_id, raw in enumerate(json_texts[:30]):
        try:
            data = _json.loads(raw)
        except Exception:
            continue

        # Try force_extract_markets first (handles array-row format from Moneycontrol)
        extracted = force_extract_markets(data)
        if extracted:
            for section_title, rows in extracted:
                s = _score_section(rows)
                if s > 0:
                    scored.append((s, section_title, rows, payload_id))
        else:
            # Fallback to generic recursive extractor for dict/list JSON shapes
            for section_title, rows in _extract_json_sections(data):
                s = _score_section(rows)
                if s > 0:
                    scored.append((s, section_title, rows, payload_id))

    # Sort by score descending — richest data sections first
    scored.sort(key=lambda x: x[0], reverse=True)

    for score, section_title, rows, payload_id in scored:
        sample = rows[0]
        cols = [
            k for k, v in sample.items()
            if isinstance(v, (str, int, float, bool)) and len(str(v)) < 80
        ][:14]
        if not cols:
            continue

        # Deduplicate: same cols + same payload + same title = true duplicate → skip
        # same cols + same payload + different title = different region → keep (US/Europe/Asia)
        # same cols + different payload = same API called twice → skip
        fingerprint = (frozenset(cols), payload_id, section_title)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)

        if section_title:
            out.append(f"### {section_title}")

        out.append("| " + " | ".join(cols) + " |")
        out.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in rows[:300]:
            vals = [str(row.get(c, "")) for c in cols]
            out.append("| " + " | ".join(vals) + " |")
        out.append("")

    return "\n".join(out)

def force_extract_markets(obj, results=None):
    if results is None:
        results = []

    if isinstance(obj, dict):
        # 🎯 HEAT-SEEKER: Find the market data
        if "heading" in obj and "data" in obj and isinstance(obj["data"], list):
            region = obj["heading"]
            raw_stocks = obj["data"]
            clean_stocks = []

            # Detect header row (first array row where values are column name strings)
            headers = None
            for row in raw_stocks:
                if isinstance(row, list):
                    # First list row = column headers (e.g. ['symbol','name','ltp',...])
                    if headers is None:
                        headers = [str(h).lower() for h in row]
                        continue
                    # Subsequent list rows = data rows — map using detected headers
                    if len(row) >= 2:
                        clean_stock = {
                            headers[i] if i < len(headers) else f"col{i}": row[i]
                            for i in range(min(len(row), len(headers) if headers else len(row)))
                        }
                        # Skip any row whose values look like the header itself
                        if clean_stock.get("name") in ("name", "symbol", "index", ""):
                            continue
                        clean_stocks.append(clean_stock)
                elif isinstance(row, dict):
                    # Already a dict row — skip if it's a schema descriptor
                    if row.get("name") in ("symbol", "name", ""):
                        continue
                    clean_stocks.append(row)

            # Save the cleanly mapped data!
            if len(clean_stocks) > 0:
                results.append((region, clean_stocks))

        # Recursively dig deeper into all dictionary values
        for value in obj.values():
            force_extract_markets(value, results)

    elif isinstance(obj, list):
        # Recursively dig into all lists
        for item in obj:
            force_extract_markets(item, results)

    return results

# ── Pick best result ──────────────────────────────────────────────────────────

def _pick_best(results: list[StrategyResult]) -> Optional[StrategyResult]:
    successful = [r for r in results if r.success and r.text and len(r.text.split()) >= 20]
    if not successful:
        return None
    def score(r: StrategyResult) -> float:
        base = float(r.word_count)
        if r.strategy == "playwright":
            if r.extra.get("xhr_captured", 0) > 0:
                base *= 3.0
            if r.extra.get("tables_found", 0) > 0:
                base *= 1.5
        return base
    return max(successful, key=score)


# ── Metadata extractor ────────────────────────────────────────────────────────

def _metadata(html: str, url: str) -> dict:
    try:
        soup = BeautifulSoup(html, "lxml")
        meta: dict = {}
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
            "author": meta.get("author") or meta.get("article:author"),
            "published_time": meta.get("article:published_time"),
        }
    except Exception:
        return {"domain": urlparse(url).netloc}