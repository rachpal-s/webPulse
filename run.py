"""run.py — WebPulse startup script"""
import uvicorn
from config import get_settings
import socket
import argparse
cfg = get_settings()
def get_ip():
    """
    Returns the local IP address of the machine (LAN IP).
    Works on Windows, Linux, and macOS.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # We don't actually need to send data or have internet access.
        # connecting to a public IP tells the OS to figure out 
        # which interface to use, revealing the correct local IP.
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        # Fallback if there is no network connection
        local_ip = "127.0.0.1"
    finally:
        s.close()
    
    return local_ip

async def _check_ollama():
    import httpx
    cfg_inner = get_settings()
    checks = [
        ("Embed   ", cfg_inner.ollama_embed_url),
        ("Inference", cfg_inner.ollama_inference_url),
    ]
    all_ok = True
    print("  Checking Ollama endpoints...")
    for label, base_url in checks:
        url = base_url.rstrip("/") + "/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(url)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                print(f"  OK  {label}: {base_url} ({len(models)} models: {', '.join(models[:3])})")
            else:
                print(f"  ERR {label}: {base_url} HTTP {r.status_code}")
                all_ok = False
        except Exception as e:
            print(f"  ERR {label}: {base_url} UNREACHABLE — {e.__class__.__name__}")
            all_ok = False
    if not all_ok:
        print("  WARNING: Ollama endpoint(s) down — no semantic search, keyword fallback only")
    else:
        print("  All Ollama endpoints reachable — semantic search enabled")
    print()

if __name__ == "__main__":
    PORT = cfg.app_port
    HOST = cfg.app_host
    parser = argparse.ArgumentParser(description="Run app with own port")
    
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"Port to run the application (default: {PORT})"
    )

    args = parser.parse_args()

    PORT = args.port

    import asyncio as _aio
    _aio.run(_check_ollama())

    print(f"""
  ██╗    ██╗███████╗██████╗ ██████╗ ██╗   ██╗██╗      ███████╗███████╗
  ██║    ██║██╔════╝██╔══██╗██╔══██╗██║   ██║██║      ██╔════╝██╔════╝
  ██║ █╗ ██║█████╗  ██████╔╝██████╔╝██║   ██║██║      ███████╗█████╗
  ██║███╗██║██╔══╝  ██╔══██╗██╔═══╝ ██║   ██║██║      ╚════██║██╔══╝
  ╚███╔███╔╝███████╗██████╔╝██║     ╚██████╔╝███████╗ ███████║███████╗
   ╚══╝╚══╝ ╚══════╝╚═════╝ ╚═╝      ╚═════╝ ╚══════╝ ╚══════╝╚══════╝
  v3.0  Multi-Strategy Scraper + Adaptive RAG
    """)
    print(f"  → App        : http://{get_ip()}:{PORT}")
    print(f"  → Inference  : {cfg.ollama_inference_url}  ({cfg.ollama_inference_model})")
    print(f"  → Embeddings : {cfg.ollama_embed_url}  ({cfg.ollama_embed_model})")
    print(f"  → Auth       : {'✓ API key set' if cfg.ollama_inference_api_key else '— none (open endpoint)'}")
    print(f"  → DB         : {cfg.db_path}")
    print(f"  → Debug      : {cfg.app_debug}\n")

    uvicorn.run(
        "main:app",
        host=cfg.app_host,
        port=cfg.app_port,
        reload=cfg.app_debug,
        log_level="info",
    )