"""
jobs/queue_crawler.py — Background job that crawls daily URLs and
pre-populates url_queue with relevance-scored articles.

Runs on a configurable schedule so the morning brief can consume
from the queue instead of crawling at brief time.
"""
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from config import get_settings
from rag.pipeline import get_store

log = logging.getLogger("queue_crawler")
cfg = get_settings()


async def refresh_queue(force: bool = False) -> dict:
    """
    For each category, crawl its daily URLs and populate url_queue.
    Returns summary dict.
    """
    store = get_store()
    tz = ZoneInfo(cfg.brief_timezone)
    now = datetime.now(tz)
    hour = now.hour

    if not force:
        # Check time window
        if hour < cfg.crawl_queue_start_hour or hour >= cfg.crawl_queue_end_hour:
            log.info("Outside crawl window (%d-%d) — skipping",
                     cfg.crawl_queue_start_hour, cfg.crawl_queue_end_hour)
            return {"status": "outside_window", "hour": hour}

    # Clean stale entries first
    cleaned = store.clear_old_queue(max_age_hours=24)
    if cleaned:
        log.info("Cleared %d stale queue entries", cleaned)

    from scraper.crawler import discover_recent_pages
    from scraper.relevance import build_category_intent_vector, filter_by_relevance
    from jobs.morning_brief import _get_prompts
    from rag.ollama import get_ollama_client

    ollama = get_ollama_client()
    categories = store.get_categories()
    total_queued = 0
    summary = []

    for cat in categories:
        cat_id = cat["id"]
        cat_name = cat["name"]

        # Get this category's daily URLs
        daily_urls = store.get_urls_by_category(cat_id)
        if not daily_urls:
            continue

        log.info("Queue refresh: %s (%d URLs)", cat_name, len(daily_urls))

        # Build prompts for this category
        cat_prompts_raw = store.get_category_prompts(cat_id)
        cat_prompts = [
            {"key": p["prompt_key"], "label": p["label"], "prompt": p["prompt_text"]}
            for p in cat_prompts_raw
            if p.get("prompt_key") not in ("undefined", "unknown", "")
            and p.get("prompt_text")
        ] if cat_prompts_raw else []

        prompts = cat_prompts if cat_prompts else _get_prompts()

        # Build intent vector
        try:
            intent_vector = await build_category_intent_vector(prompts, ollama)
        except Exception as e:
            log.warning("Intent vector failed for %s: %s", cat_name, e)
            intent_vector = []

        cat_queued = 0
        for url_rec in daily_urls:
            domain_url = url_rec["url"]
            try:
                # Discover recent pages
                crawled = await asyncio.wait_for(
                    discover_recent_pages(
                        domain_url,
                        window_hours=cfg.crawl_window_hours,
                        max_results=cfg.crawl_max_results,
                    ),
                    timeout=30
                )

                if not crawled:
                    log.info("  %s: no recent pages", domain_url)
                    continue

                log.info("  %s: %d pages via %s",
                         domain_url, len(crawled), crawled[0].source)

                # Relevance filter
                if intent_vector:
                    relevant, had_match = await filter_by_relevance(
                        crawled, intent_vector, ollama,
                        min_score=cfg.crawl_min_score,
                        top_n=cfg.crawl_top_n,
                    )
                    if not had_match:
                        log.info("  %s: no pages above threshold %.2f",
                                 domain_url, cfg.crawl_min_score)
                        continue
                else:
                    # No intent vector — queue top N by recency
                    relevant = crawled[:cfg.crawl_top_n]

                # Upsert into queue
                pages_to_queue = [
                    {
                        "url": p.url,
                        "title": p.title,
                        "summary": p.summary,
                        "score": p.score,
                    }
                    for p in relevant
                ]
                store.upsert_queue_urls(cat_id, domain_url, pages_to_queue)
                cat_queued += len(pages_to_queue)
                log.info("  %s: queued %d articles", domain_url, len(pages_to_queue))

            except Exception as e:
                log.warning("  Queue crawl failed %s: %s", domain_url, e)

        total_queued += cat_queued
        summary.append({"category": cat_name, "queued": cat_queued})
        log.info("Queue refresh: %s → %d articles queued", cat_name, cat_queued)

    log.info("Queue refresh complete: %d total articles queued", total_queued)
    return {
        "status": "done",
        "total_queued": total_queued,
        "categories": summary,
        "timestamp": now.isoformat(),
    }