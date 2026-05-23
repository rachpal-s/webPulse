"""
scraper/filter.py — GenAI-first intelligent filtering.

Pipeline:
  1. Intent classification  — Ollama decides if phrase is a question, topic, or instruction
  2. Content-type routing   — headlines vs table rows handled differently
  3. Semantic scoring       — embeddings for headlines; LLM row evaluation for tables
  4. GenAI insight          — for data pages, generate a natural-language summary of matches
  5. Regex fallback         — if Ollama unreachable, falls back to substring matching

All Ollama calls are non-blocking with timeouts; Ollama failure = graceful regex fallback.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Intent classification ─────────────────────────────────────────────────────

@dataclass
class FilterIntent:
    phrase: str
    intent_type: str          # "question" | "topic" | "instruction" | "unknown"
    refined_query: str        # normalised version for embedding / LLM prompts
    condition: Optional[str]  # e.g. "chg < 0 AND region = Asia" for table rows
    confidence: float         # 0-1
    strict: bool = False      # user said only/just/exclusively
    key_terms: list = None    # extracted key terms
    named_entities: list = None  # proper nouns — strict presence required


_INTENT_PROMPT = """\
You are a search intent analyser for a news headline filter.

User filter phrase: "{phrase}"

Respond with ONLY a JSON object (no markdown):
{{
  "intent_type": "<question|topic|instruction>",
  "refined_query": "<expanded query: synonyms, related places, alternate terms>",
  "key_terms": ["<main non-proper search terms>"],
  "named_entities": ["<proper nouns ONLY: specific people/places/orgs — exact spelling>"],
  "condition": "<plain English: what a matching headline looks like>",
  "strict": <true if user said only/just/exclusively, false otherwise>,
  "confidence": <0.0-1.0>
}}

named_entities RULES (critical):
- Include ONLY proper nouns (specific named people, cities, organisations)
- These are MANDATORY — headline must contain them or their synonyms to match
- Also include the parent region for places (nakodar → ["nakodar","jalandhar"])
- Do NOT include common words like "news","market","stocks" here

Examples:
- "nakodar news only" → named_entities=["nakodar","jalandhar"], strict=true, refined="nakodar jalandhar punjab doaba"
- "modi cabinet meeting" → named_entities=["modi"], strict=true, refined="modi cabinet ministers pm"
- "declining Asian stocks" → named_entities=[], strict=false, refined="declining falling Asian market indices"
- "budget 2025" → named_entities=[], strict=false, refined="union budget 2025 india fiscal policy"
"""


async def classify_intent(phrase: str) -> FilterIntent:
    """Ask Ollama to classify the filter intent. Falls back to 'topic' on failure."""
    default = FilterIntent(
        phrase=phrase,
        intent_type="topic",
        refined_query=phrase,
        condition=None,
        confidence=0.5,
    )
    try:
        from rag.ollama import get_ollama_client
        client = get_ollama_client()
        response = await asyncio.wait_for(
            client.chat(
                [{"role": "user", "content": _INTENT_PROMPT.format(phrase=phrase)}],
                temperature=0.0,
                max_tokens=256,
            ),
            timeout=12.0,
        )
        # Strip any markdown fences
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        data = json.loads(clean)
        return FilterIntent(
            phrase=phrase,
            intent_type=data.get("intent_type", "topic"),
            refined_query=data.get("refined_query", phrase),
            condition=data.get("condition"),
            confidence=float(data.get("confidence", 0.7)),
            strict=bool(data.get("strict", False)),
            key_terms=data.get("key_terms", []),
            named_entities=data.get("named_entities", []),
        )
    except Exception:
        return default


# ── Headline filtering ────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# Financial / directional synonyms for better regex fallback when Ollama unavailable
_SYNONYMS: dict[str, list[str]] = {
    "declining": ["drop", "fall", "fell", "slide", "slip", "down", "loss", "loss", "weak", "tumble", "plunge", "sink", "negative", "red", "lower"],
    "rising":    ["rise", "rose", "gain", "rally", "up", "surge", "jump", "climb", "advance", "positive", "green", "higher", "record"],
    "asian":     ["asia", "nikkei", "sensex", "hang seng", "kospi", "shanghai", "india", "japan", "china", "korea", "hong kong"],
    "european":  ["europe", "ftse", "dax", "cac", "stoxx", "uk", "germany", "france"],
    "us":        ["wall street", "dow", "nasdaq", "s&p", "nyse"],
    "market":    ["stock", "index", "indices", "equity", "shares", "bourse"],
    "volatile":  ["volatility", "swing", "fluctuat", "uncertain", "turbul"],
}

def _expand_token(token: str) -> list[str]:
    """Return token + any synonyms for regex fallback."""
    t = token.lower()
    return [t] + _SYNONYMS.get(t, [])


def _regex_score(text: str, tokens: list[str]) -> float:
    text_l = text.lower()
    hits = 0
    for t in tokens:
        expanded = _expand_token(t)
        if any(syn in text_l for syn in expanded):
            hits += 1
    return hits / len(tokens) if tokens else 0.0


async def filter_headlines(
    items: list[dict],
    intent: FilterIntent,
    field_keys: list[str],
    semantic_threshold: float = 0.30,
    regex_threshold: float = 0.25,
) -> list[dict]:
    """
    Filter headline items using semantic embeddings (primary) + regex (fallback).
    Returns items ranked by relevance score.
    """
    if not items:
        return items

    tokens = intent.refined_query.lower().split()
    query_text = intent.refined_query

    # ── Regex scores (always, instant) ───────────────────────────────────────
    def _item_text(item: dict) -> str:
        return " ".join(str(item.get(k, "") or "") for k in field_keys)

    regex_scores: dict[int, float] = {
        i: _regex_score(_item_text(item), tokens)
        for i, item in enumerate(items)
    }

    # ── Semantic scores via Ollama embeddings ─────────────────────────────────
    semantic_scores: dict[int, float] = {}
    try:
        from rag.ollama import get_ollama_client
        client = get_ollama_client()

        texts = [_item_text(item) for item in items]
        phrase_emb, *item_embs = await asyncio.wait_for(
            client.embed([query_text] + texts),
            timeout=30.0,
        )
        for i, emb in enumerate(item_embs):
            sim = _cosine(phrase_emb, emb)
            semantic_scores[i] = sim

    except Exception:
        pass  # fall back to regex only

    # ── Combine: union of matches, weighted score ─────────────────────────────
    combined: dict[int, float] = {}
    for i in range(len(items)):
        rscore = regex_scores.get(i, 0.0)
        sscore = semantic_scores.get(i, 0.0)

        # An item passes if EITHER score clears its threshold
        if rscore >= regex_threshold or sscore >= semantic_threshold:
            # Weighted combination: semantic carries more weight when available
            if semantic_scores:
                combined[i] = rscore * 0.35 + sscore * 0.65
            else:
                combined[i] = rscore

    if not combined:
        return []

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [items[i] for i, _ in ranked]


# ── Table row filtering ───────────────────────────────────────────────────────

_ROW_EVAL_PROMPT = """\
You are a data filter assistant. Given a user's filter condition and a batch of data rows, \
decide which rows match.

Filter condition: {condition}

Rows (JSON):
{rows_json}

Respond with ONLY a JSON array of the indices (0-based) of rows that match the condition. \
No explanation, no markdown. Example: [0, 2, 5]
If none match, respond: []
"""

_INSIGHT_PROMPT = """\
You are a financial data analyst. The user filtered market data with: "{phrase}"

These rows matched:
{rows_text}

Write a concise 1-3 sentence insight about these results. Be specific — mention names, \
numbers, trends. Do not use bullet points. Do not repeat the filter phrase verbatim.
"""


async def filter_table_rows(
    content_md: str,
    intent: FilterIntent,
    max_rows_per_batch: int = 40,
) -> tuple[str, str]:
    """
    Filter markdown table content to only rows matching the intent condition.
    Returns (filtered_markdown, insight_text).

    For each table section (### heading + rows), sends rows to LLM in batches
    to decide which match, then generates a unified insight.
    """
    if not content_md or not intent.condition:
        # No structured condition — return full content with no insight
        return content_md, ""

    lines = content_md.split("\n")
    output_sections: list[str] = []
    all_matched_rows: list[dict] = []

    # ── Parse markdown into sections ──────────────────────────────────────────
    i = 0
    while i < len(lines):
        line = lines[i]

        # Section heading
        if line.startswith("### "):
            heading = line
            i += 1
            # Collect header + separator + data rows
            header_row: list[str] = []
            data_rows: list[list[str]] = []

            while i < len(lines) and (lines[i].startswith("|") or lines[i].strip() == ""):
                r = lines[i]
                if not r.strip():
                    i += 1
                    continue
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                if all(set(c) <= set("- ") for c in cells):
                    # separator row — skip
                    pass
                elif not header_row:
                    header_row = cells
                else:
                    data_rows.append(cells)
                i += 1

            if not header_row or not data_rows:
                output_sections.append(heading)
                continue

            # Convert rows to dicts for LLM evaluation
            row_dicts = [
                {header_row[j]: cells[j] if j < len(cells) else ""
                 for j in range(len(header_row))}
                for cells in data_rows
            ]

            # LLM batch evaluation
            matched_indices = await _eval_rows_batch(
                row_dicts, intent.condition, max_rows_per_batch
            )

            if matched_indices:
                matched_dicts = [row_dicts[idx] for idx in matched_indices if idx < len(row_dicts)]
                all_matched_rows.extend(matched_dicts)

                # Rebuild markdown for matched rows only
                sep = "| " + " | ".join(["---"] * len(header_row)) + " |"
                md_lines = [
                    heading,
                    "| " + " | ".join(header_row) + " |",
                    sep,
                ]
                for idx in matched_indices:
                    if idx < len(data_rows):
                        md_lines.append("| " + " | ".join(data_rows[idx]) + " |")
                md_lines.append("")
                output_sections.append("\n".join(md_lines))
        else:
            # Non-table line (prose, ## heading etc.) — keep as-is
            output_sections.append(line)
            i += 1

    filtered_md = "\n".join(output_sections).strip()

    # ── Generate insight across ALL matched rows ───────────────────────────────
    insight = ""
    if all_matched_rows:
        insight = await _generate_insight(all_matched_rows, intent.phrase)

    return filtered_md, insight


async def _eval_rows_batch(
    row_dicts: list[dict],
    condition: str,
    batch_size: int,
) -> list[int]:
    """Send rows to LLM in batches; collect matching indices."""
    matched: list[int] = []

    # Regex pre-filter as a fast first pass to reduce LLM load
    # (Only skip obviously irrelevant rows; LLM makes final call)
    try:
        from rag.ollama import get_ollama_client
        client = get_ollama_client()

        for batch_start in range(0, len(row_dicts), batch_size):
            batch = row_dicts[batch_start: batch_start + batch_size]
            rows_json = json.dumps(batch, indent=2)
            prompt = _ROW_EVAL_PROMPT.format(
                condition=condition,
                rows_json=rows_json,
            )
            response = await asyncio.wait_for(
                client.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                ),
                timeout=20.0,
            )
            clean = re.sub(r"```(?:json)?|```", "", response).strip()
            batch_indices = json.loads(clean)
            # Offset back to global index
            matched.extend(batch_start + idx for idx in batch_indices
                           if isinstance(idx, int) and 0 <= idx < len(batch))

    except Exception:
        # LLM failed — fall back to simple regex on stringified rows
        condition_tokens = condition.lower().split()
        for i, row in enumerate(row_dicts):
            row_text = " ".join(str(v) for v in row.values()).lower()
            if any(t in row_text for t in condition_tokens):
                matched.append(i)

    return matched


async def _generate_insight(matched_rows: list[dict], phrase: str) -> str:
    """Ask Ollama to write a concise insight about the matched rows."""
    try:
        from rag.ollama import get_ollama_client
        client = get_ollama_client()

        # Summarise rows as readable text (avoid sending huge JSON)
        rows_text = "\n".join(
            ", ".join(f"{k}: {v}" for k, v in row.items() if v)
            for row in matched_rows[:50]   # cap at 50 rows for prompt size
        )
        prompt = _INSIGHT_PROMPT.format(phrase=phrase, rows_text=rows_text)
        insight = await asyncio.wait_for(
            client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=150,
            ),
            timeout=20.0,
        )
        return insight.strip()
    except Exception:
        # Build a basic insight from the data itself
        n = len(matched_rows)
        names = [r.get("name", r.get("symbol", "")) for r in matched_rows[:5] if r.get("name") or r.get("symbol")]
        if names:
            return f"Found {n} matching {'row' if n==1 else 'rows'}: {', '.join(names)}{'…' if n > 5 else ''}."
        return f"Found {n} matching {'row' if n==1 else 'rows'}."


# ── Unified entry point ───────────────────────────────────────────────────────

@dataclass
class FilterResult:
    headlines: list[dict] = field(default_factory=list)
    content: Optional[str] = None          # filtered content (tables_md or text)
    insight: str = ""                      # GenAI insight (data pages only)
    intent: Optional[FilterIntent] = None
    matched_count: int = 0
    filter_applied: bool = False


async def apply_filter(
    phrase: str,
    page_type: str,
    headlines: Optional[list[dict]] = None,
    tables_md: Optional[str] = None,
    content: Optional[str] = None,
) -> FilterResult:
    """
    Unified filter entry point. Routes to the right strategy based on page_type.

    Args:
        phrase:     raw filter phrase from user
        page_type:  "homepage" | "data" | "article" | "unknown"
        headlines:  list of {title, url, section, summary} dicts
        tables_md:  markdown table string (data pages)
        content:    plain text content (articles)

    Returns:
        FilterResult with filtered headlines/content and optional insight
    """
    if not phrase or not phrase.strip():
        return FilterResult(
            headlines=headlines or [],
            content=tables_md or content,
            filter_applied=False,
        )

    # Step 1: classify intent (with timeout fallback)
    intent = await classify_intent(phrase)

    result = FilterResult(intent=intent, filter_applied=True)

    # Step 2: route by page type
    if page_type in ("homepage", "unknown") and headlines:
        filtered = await filter_headlines(
            headlines, intent,
            field_keys=["title", "url", "summary", "section"],
        )
        result.headlines = filtered
        result.matched_count = len(filtered)

    elif page_type == "data" and tables_md:
        filtered_md, insight = await filter_table_rows(tables_md, intent)
        result.content = filtered_md
        result.insight = insight
        # Count matched rows (lines starting with | that aren't headers/separators)
        result.matched_count = sum(
            1 for l in filtered_md.split("\n")
            if l.startswith("|") and "---" not in l and
            not any(h in l for h in ["NAME", "SYMBOL", "name", "symbol"])
        )

    else:
        # article or fallback — return unchanged (per user's decision)
        result.headlines = headlines or []
        result.content = content or tables_md
        result.matched_count = 0

    return result