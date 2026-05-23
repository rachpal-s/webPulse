"""
jobs/scheduler.py — APScheduler for morning brief.

Runs Mon-Fri at BRIEF_START_TIME (IST).
On startup: checks if today's brief was missed and runs immediately if so.
"""
import logging
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import get_settings

log = logging.getLogger("scheduler")
cfg = get_settings()
_scheduler: AsyncIOScheduler | None = None


async def _run_brief_job():
    """Wrapper called by APScheduler."""
    from jobs.morning_brief import run_morning_brief
    log.info("Scheduled morning brief triggered")
    result = await run_morning_brief()
    log.info("Scheduled brief result: %s", result.get("status"))


def _should_run_today() -> bool:
    """Return True if today is Mon-Fri and brief hasn't run yet."""
    tz = ZoneInfo(cfg.brief_timezone)
    now = datetime.now(tz)
    if now.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    from rag.pipeline import get_store
    today = date.today().isoformat()
    store = get_store()
    brief = store.get_brief(today)
    return brief is None or brief["status"] not in ("done", "running")


async def startup_catchup():
    """
    Called at app startup. If auto_run enabled and scheduled run was missed
    (machine was down), run immediately.
    """
    if not cfg.brief_auto_run:
        log.info("Startup catchup skipped — BRIEF_AUTO_RUN=false")
        return

    tz = ZoneInfo(cfg.brief_timezone)
    now = datetime.now(tz)

    if now.weekday() >= 5:
        log.info("Weekend — no brief needed")
        return

    # Parse scheduled time
    h, m = map(int, cfg.brief_start_time.split(":"))
    scheduled_today = now.replace(hour=h, minute=m, second=0, microsecond=0)

    if now < scheduled_today:
        log.info("Before scheduled time (%s IST) — brief not due yet",
                 cfg.brief_start_time)
        return

    if _should_run_today():
        log.info("Startup catchup: missed brief detected — running now")
        from jobs.morning_brief import run_morning_brief
        await run_morning_brief()
    else:
        log.info("Startup catchup: brief already done for today")


def start_scheduler():
    global _scheduler

    tz = cfg.brief_timezone
    _scheduler = AsyncIOScheduler(timezone=tz)

    if cfg.brief_auto_run:
        h, m = map(int, cfg.brief_start_time.split(":"))
        _scheduler.add_job(
            _run_brief_job,
            CronTrigger(day_of_week="mon-fri", hour=h, minute=m, timezone=tz),
            id="morning_brief",
            name="Morning Market Brief",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        log.info("Scheduler started — auto brief at %s IST Mon-Fri", cfg.brief_start_time)
    else:
        log.info("Scheduler started — auto brief DISABLED (BRIEF_AUTO_RUN=false)")

    _scheduler.start()
    return _scheduler


def stop_scheduler():
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)