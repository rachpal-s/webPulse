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