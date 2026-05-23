# WebPulse v2 — Multi-Strategy Web Scraper

6-engine content extraction with **Playwright headless Chrome** for JS-rendered pages.

## Quick Start

```bash
pip install -r requirements.txt
playwright install chromium
uvicorn main:app --reload --port 8000
```
Open → http://localhost:8000

---

## Strategies

| Strategy | How it works | Best for |
|---|---|---|
| **🎭 Playwright** | Real headless Chrome, executes JS, waits for dynamic data | Market data, SPAs, Moneycontrol, NSE, Bloomberg |
| **Trafilatura** | ML boilerplate removal | News articles, blogs |
| **Readability** | Mozilla Reader Mode algorithm | Articles, editorial |
| **Goose3** | Gravity/Goose extractor | News |
| **BeautifulSoup** | Custom heuristic, kills ad/nav/script elements | General |
| **Newspaper3k** | NLP extractor, pulls authors/dates | News |

Winner = highest word count (most complete extraction).

---

## API

### POST /scrape
```json
{
  "url": "https://moneycontrol.com/markets/global-indices/",
  "strategies": ["playwright", "trafilatura"],
  "wait_for_selector": "table.mktIndTbl",
  "wait_seconds": 4
}
```

### GET /scrape?url=...&wait_seconds=3

### GET /health

---

## Tips for JS-heavy sites

- Set `wait_for_selector` to a CSS selector that appears only after data loads  
  e.g. `"table.mktIndTbl"` for Moneycontrol indices tables
- Increase `wait_seconds` (3–8) if data loads slowly
- Use only `["playwright"]` strategy for fastest results on SPA sites

## Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```
Use 2 workers max (Playwright is memory-heavy).
