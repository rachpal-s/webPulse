"""
test_embeddings.py — Test embedding provider connectivity and semantic search.

Usage:
  python test_embeddings.py              # test current provider from .env
  python test_embeddings.py --all        # test all configured providers
  python test_embeddings.py --search     # also test semantic search on latest session
"""
import asyncio
import sys
import os
import sqlite3
import struct

sys.path.insert(0, ".")


async def test_provider(provider: str, ollama_client) -> dict:
    """Test a specific provider with a sample text."""
    import os as _os
    orig = _os.environ.get("EMBED_PROVIDER")
    _os.environ["EMBED_PROVIDER"] = provider

    result = {"provider": provider, "ok": False, "dims": 0, "error": ""}
    try:
        embs = await ollama_client.embed(["Indian stock market Nifty Sensex"])
        if embs and embs[0]:
            result["ok"] = True
            result["dims"] = len(embs[0])
    except Exception as e:
        result["error"] = str(e)[:120]
    finally:
        if orig is None:
            _os.environ.pop("EMBED_PROVIDER", None)
        else:
            _os.environ["EMBED_PROVIDER"] = orig

    return result


async def test_semantic_search(session_id: str, store) -> dict:
    """Test similarity search on a given session."""
    from rag.ollama import get_ollama_client
    ollama = get_ollama_client()

    try:
        embs = await ollama.embed(["buy sell stock recommendation"])
        if not embs or not embs[0]:
            return {"ok": False, "error": "Embedding returned empty"}

        results = store.similarity_search(session_id, embs[0], top_k=5)
        return {
            "ok": True,
            "results": len(results),
            "top_score": results[0].get("similarity", 0) if results else 0,
            "top_source": results[0].get("source_title", "")[:60] if results else "",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:120]}


async def main():
    from config import get_settings
    from rag.pipeline import get_store
    from rag.ollama import get_ollama_client, OllamaClient

    cfg = get_settings()
    store = get_store()
    ollama = get_ollama_client()

    print("\n" + "=" * 60)
    print("  WebPulse Embedding Test")
    print("=" * 60)

    # ── Current config ────────────────────────────────────────────
    current = os.environ.get("EMBED_PROVIDER", cfg.embed_provider).lower()
    print(f"\n  Current EMBED_PROVIDER : {current.upper()}")
    print(f"  Ollama embed URL       : {cfg.ollama_embed_url}")
    print(f"  Ollama embed model     : {cfg.ollama_embed_model}")
    print(f"  Google embed model     : {cfg.google_embed_model}")
    print(f"  Embed dimensions       : {cfg.embed_dimensions}")

    # ── DB stats ──────────────────────────────────────────────────
    print("\n  Database:")
    conn = sqlite3.connect(cfg.db_path)
    chunks_n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    try:
        emb_n = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
    except Exception:
        emb_n = "table missing"

    sessions = conn.execute(
        "SELECT session_id, embed_provider, embed_model, "
        "(SELECT COUNT(*) FROM chunks WHERE session_id=s.session_id) as cc "
        "FROM sessions s ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    conn.close()

    print(f"    chunks         : {chunks_n:,}")
    print(f"    chunk_embeddings: {emb_n:,}")
    print(f"\n  Recent sessions (newest first):")
    for sid, ep, em, cc in sessions:
        emb_count = 0
        try:
            db = sqlite3.connect(cfg.db_path)
            emb_count = db.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE session_id=?)", (sid,)
            ).fetchone()[0]
            db.close()
        except Exception:
            pass
        print(f"    {sid[:12]}…  provider={ep or 'ollama':10}  "
              f"chunks={cc:4}  embeddings={emb_count:4}  model={em or 'unknown'}")

    # ── Test current provider ─────────────────────────────────────
    print(f"\n  Testing current provider ({current.upper()})...")
    result = await test_provider(current, ollama)
    if result["ok"]:
        print(f"  ✓ {current.upper():10} → {result['dims']} dims")
    else:
        print(f"  ✗ {current.upper():10} → FAILED: {result['error']}")

    # ── Test all providers if --all ────────────────────────────────
    if "--all" in sys.argv:
        print("\n  Testing all configured providers:")
        providers_to_test = []
        if cfg.ollama_embed_url:
            providers_to_test.append("ollama")
        if cfg.google_api_key:
            providers_to_test.append("google")
        if cfg.openai_api_key:
            providers_to_test.append("openai")
        if cfg.cohere_api_key:
            providers_to_test.append("cohere")

        for p in providers_to_test:
            if p == current:
                continue
            r = await test_provider(p, ollama)
            status = f"✓ {r['dims']} dims" if r["ok"] else f"✗ {r['error'][:60]}"
            print(f"  {'✓' if r['ok'] else '✗'} {p.upper():10} → {status}")

    # ── Test semantic search if --search ──────────────────────────
    if "--search" in sys.argv and sessions:
        latest_sid = sessions[0][0]
        print(f"\n  Testing semantic search on session {latest_sid[:12]}…")
        sr = await test_semantic_search(latest_sid, store)
        if sr["ok"]:
            print(f"  ✓ similarity_search → {sr['results']} results")
            if sr["results"]:
                print(f"    top score  : {sr['top_score']:.4f}")
                print(f"    top source : {sr['top_source']}")
        else:
            print(f"  ✗ similarity_search failed: {sr['error']}")

    print("\n" + "=" * 60 + "\n")


asyncio.run(main())