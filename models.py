"""
list_ollama_models.py — List all models available on the configured Ollama server.

Shows:
  - Local models (downloaded and ready)
  - Model details: size, family, quantization, context length
  - Which models are currently configured in WebPulse

Usage:
  python list_ollama_models.py
  python list_ollama_models.py --json       # raw JSON output
  python list_ollama_models.py --running    # show only currently loaded models
"""
import asyncio
import sys
import json
import httpx
from datetime import datetime


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def _ago(modified_at: str) -> str:
    try:
        dt = datetime.fromisoformat(modified_at.replace("Z", "+00:00"))
        from datetime import timezone
        delta = datetime.now(timezone.utc) - dt
        days = delta.days
        if days == 0:
            return "today"
        if days == 1:
            return "yesterday"
        if days < 30:
            return f"{days}d ago"
        if days < 365:
            return f"{days // 30}mo ago"
        return f"{days // 365}y ago"
    except Exception:
        return modified_at[:10]


async def fetch_models(base_url: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{base_url}/api/tags")
        r.raise_for_status()
        return r.json()


async def fetch_running(base_url: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{base_url}/api/ps")
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"models": []}


async def fetch_model_info(base_url: str, name: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.post(f"{base_url}/api/show",
                                   json={"name": name})
            r.raise_for_status()
            return r.json()
        except Exception:
            return {}


async def main():
    import sys
    sys.path.insert(0, ".")
    try:
        from config import get_settings
        cfg = get_settings()
        inference_url = cfg.ollama_inference_url
        embed_url = cfg.ollama_url
        inference_model = cfg.ollama_inference_model
        embed_model = cfg.ollama_embed_model
    except Exception:
        inference_url = "https://ollama.com"
        embed_url = "http://localhost:11434"
        inference_model = "unknown"
        embed_model = "unknown"

    raw_json = "--json" in sys.argv
    running_only = "--running" in sys.argv

    print(f"\n{'='*60}")
    print(f"  Ollama Model Browser")
    print(f"{'='*60}")
    print(f"  Inference URL : {inference_url}")
    print(f"  Embed URL     : {embed_url}")
    print(f"  Active model  : {inference_model}")
    print(f"  Embed model   : {embed_model}")
    print(f"{'='*60}\n")

    # Fetch from both endpoints (may be same or different servers)
    endpoints = list({inference_url, embed_url})

    for url in endpoints:
        print(f"📡 Server: {url}")
        print("-" * 60)

        try:
            data = await fetch_models(url)
            running_data = await fetch_running(url)
        except Exception as e:
            print(f"  ✗ Cannot connect: {e}\n")
            continue

        models = data.get("models", [])
        running = {m["name"] for m in running_data.get("models", [])}

        if raw_json:
            print(json.dumps(data, indent=2))
            continue

        if not models:
            print("  No models found.\n")
            continue

        if running_only:
            models = [m for m in models if m["name"] in running]

        # Sort by size descending
        models.sort(key=lambda m: m.get("size", 0), reverse=True)

        # Column widths
        name_w = max(len(m["name"]) for m in models) + 2

        print(f"  {'MODEL':<{name_w}} {'SIZE':>8}  {'FAMILY':<14} {'QUANT':<8} {'MODIFIED':<12} {'STATUS'}")
        print(f"  {'-'*name_w} {'-'*8}  {'-'*14} {'-'*8} {'-'*12} {'-'*10}")

        for m in models:
            name = m["name"]
            size = _human_size(m.get("size", 0))
            modified = _ago(m.get("modified_at", ""))
            details = m.get("details", {})
            family = details.get("family", "—")[:13]
            quant = details.get("quantization_level", "—")[:7]

            status_parts = []
            if name in running:
                status_parts.append("🟢 loaded")
            if name == inference_model or name.split(":")[0] == inference_model.split(":")[0]:
                status_parts.append("⚡ active")
            if name == embed_model or name.split(":")[0] == embed_model.split(":")[0]:
                status_parts.append("🔢 embed")
            status = "  ".join(status_parts) if status_parts else ""

            print(f"  {name:<{name_w}} {size:>8}  {family:<14} {quant:<8} {modified:<12} {status}")

        print(f"\n  Total: {len(models)} model(s)\n")

    print("\nTo switch inference model, update OLLAMA_INFERENCE_MODEL in .env")
    print("To switch embed model, update OLLAMA_EMBED_MODEL in .env\n")


asyncio.run(main())