"""
test_email.py — Send today's actual morning brief as a test email.
Pulls real content from DB (brief + insights) and sends via SMTP config.
Usage: python test_email.py
"""
import asyncio
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import get_settings
from rag.pipeline import get_store

cfg = get_settings()


async def send_test():
    print("=" * 55)
    print("WebPulse Email Test — using today's brief content")
    print("=" * 55)
    print(f"  Host    : {cfg.smtp_host}:{cfg.smtp_port}")
    print(f"  From    : {cfg.smtp_from}")
    print(f"  To      : {cfg.smtp_to}")
    print(f"  Enabled : {cfg.smtp_enabled}")
    print()

    if not cfg.smtp_enabled:
        print("⚠  SMTP_ENABLED=false — set true in .env to send")
        return

    # ── Fetch today's brief from DB ───────────────────────────────────────────
    store = get_store()
    today = date.today().isoformat()
    brief = store.get_brief(today)

    if not brief:
        print(f"✗  No brief found for {today}.")
        print("   Run the morning brief first via /morning-brief → Run Now")
        return

    if not brief.get("html_content"):
        print(f"✗  Brief for {today} has no HTML content (status: {brief.get('status')})")
        return

    print(f"  Brief date   : {brief['brief_date']}")
    print(f"  Status       : {brief['status']}")
    print(f"  Articles     : {brief['articles_scraped']}")
    print(f"  HTML size    : {len(brief['html_content'])} chars")
    print()

    # ── Build message ─────────────────────────────────────────────────────────
    try:
        import aiosmtplib
    except ImportError:
        print("✗  pip install aiosmtplib")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📊 WebPulse Morning Brief — {today} [TEST]"
    msg["From"] = cfg.smtp_from
    msg["To"] = cfg.smtp_to
    msg.attach(MIMEText(
        f"WebPulse Morning Brief for {today}.\n"
        f"Articles: {brief['articles_scraped']}\n"
        f"View online: http://192.168.29.56:8000/morning-brief",
        "plain"
    ))
    msg.attach(MIMEText(brief["html_content"], "html"))

    use_tls = cfg.smtp_port == 465
    print(f"Sending via {'SSL' if use_tls else 'STARTTLS'}...")

    try:
        await aiosmtplib.send(
            msg,
            hostname=cfg.smtp_host,
            port=cfg.smtp_port,
            username=cfg.smtp_user,
            password=cfg.smtp_password,
            use_tls=use_tls,
            start_tls=not use_tls,
        )
        print(f"✓  Sent to {cfg.smtp_to}")
        print("   Check inbox and share screenshot of any rendering issues.")
    except Exception as e:
        print(f"✗  {e}")


asyncio.run(send_test())