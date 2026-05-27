"""rag/ollama.py — Ollama client for remote inference + local embeddings."""
import asyncio
import json
import time
from typing import AsyncGenerator, Optional

import httpx

from config import get_settings

cfg = get_settings()


def _auth_headers(api_key: str) -> dict:
    """Return Authorization header dict if api_key is set, else empty dict."""
    if api_key and api_key.strip():
        return {"Authorization": f"Bearer {api_key.strip()}"}
    return {}


class OllamaClient:
    """
    Unified client for Ollama inference (remote) and embeddings (local/remote).

    Auth: if OLLAMA_INFERENCE_API_KEY / OLLAMA_EMBED_API_KEY are set in .env,
    they are sent as Bearer tokens. Leave blank for standard self-hosted Ollama
    (no auth required by default).
    """

    def __init__(self):
        self.inference_url = cfg.ollama_inference_url.rstrip("/")
        self.embed_url = cfg.ollama_embed_url.rstrip("/")
        self.inference_model = cfg.ollama_inference_model
        self.embed_model = cfg.ollama_embed_model
        self._inf_headers = _auth_headers(cfg.ollama_inference_api_key)
        self._emb_headers = _auth_headers(cfg.ollama_embed_api_key)

    # ── Health checks ─────────────────────────────────────────────────────────

    async def check_inference(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(
                    f"{self.inference_url}/api/tags",
                    headers=self._inf_headers,
                )
                models = r.json().get("models", [])
                names = [m["name"] for m in models]
                return {
                    "ok": True,
                    "models": names,
                    "current": self.inference_model,
                    "available": any(self.inference_model in n for n in names),
                    "auth": bool(self._inf_headers),
                }
        except Exception as e:
            return {"ok": False, "error": str(e), "auth": bool(self._inf_headers)}

    async def check_embed(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(
                    f"{self.embed_url}/api/tags",
                    headers=self._emb_headers,
                )
                models = r.json().get("models", [])
                names = [m["name"] for m in models]
                return {
                    "ok": True,
                    "models": names,
                    "current": self.embed_model,
                    "available": any(self.embed_model in n for n in names),
                    "auth": bool(self._emb_headers),
                }
        except Exception as e:
            return {"ok": False, "error": str(e), "auth": bool(self._emb_headers)}

    # ── Embeddings ────────────────────────────────────────────────────────────

    # ── Provider-specific rate limit config ──────────────────────────────────
    # ── Provider → default embedding dimensions ──────────────────────────────
    _PROVIDER_DIMS = {
        "ollama": 768,
        "google": 768,
        "jina":   1024,
        "openai": 1536,
        "cohere": 1024,
    }

    @classmethod
    def dims_for_provider(cls, provider: str) -> int:
        return cls._PROVIDER_DIMS.get(provider.lower(), 768)
    
    # Tuned to stay safely under each provider's free tier limits
    _PROVIDER_CONFIG = {
        "google": {
            # gemini-embedding-001: 90 RPM, 950 RPD — use batch + sleep
            "batch_size": 20,
            "inter_batch_sleep": 0.8,   # ~75 req/min → under 90 RPM limit
            "retry_sleep_base": 5,       # longer backoff on 429
        },
        "openai": {
            # text-embedding-3-small: 3000 RPM — very generous
            "batch_size": 100,
            "inter_batch_sleep": 0.0,
            "retry_sleep_base": 2,
        },
        "cohere": {
            # embed-english-v3: 20 RPM — very tight
            "batch_size": 10,
            "inter_batch_sleep": 3.5,   # ~17 req/min → under 20 RPM
            "retry_sleep_base": 10,
        },
        "ollama": {
            # Local — CPU/thermal throttle protection
            "batch_size": 4,
            "inter_batch_sleep": 0.5,
            "retry_sleep_base": 5,
        },
        "jina": {
            # jina-embeddings-v3: 100 RPM, no daily limit, 10M free tokens
            "batch_size": 50,          # Jina supports up to 2048 per request
            "inter_batch_sleep": 0.6,  # ~100 RPM safe zone
            "retry_sleep_base": 3,
        },
    }

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings — routes to provider set in EMBED_PROVIDER."""
        # Read fresh settings — clears lru_cache to pick up .env changes
        provider = self._live_cfg().embed_provider.lower()
        import logging as _lg
        _lg.getLogger("rag").info("embed() called with provider=%s", provider)

        if provider == "google":
            return await self._embed_google(texts)
        elif provider == "openai":
            return await self._embed_openai(texts)
        elif provider == "cohere":
            return await self._embed_cohere(texts)
        elif provider == "jina":
            return await self._embed_jina(texts)
        else:
            return await self._embed_ollama(texts)

    def _live_cfg(self):
        """Get settings — reads EMBED_PROVIDER from os.environ for live switching."""
        import os as _os
        from config import get_settings as _gs
        _c = _gs()
        # Allow live override via os.environ (populated by load_dotenv in main.py)
        provider = _os.environ.get("EMBED_PROVIDER", _c.embed_provider)
        if provider != _c.embed_provider:
            # Create a simple namespace with overridden provider
            import types as _t
            cfg_copy = _t.SimpleNamespace(**{
                k: getattr(_c, k) for k in dir(_c) if not k.startswith('_')
            })
            cfg_copy.embed_provider = provider
            return cfg_copy
        return _c

    async def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        """Ollama embedding — batch first, legacy fallback. Respects thermal throttle config."""
        import asyncio as _aio
        cfg_p = self._PROVIDER_CONFIG["ollama"]
        BATCH = cfg_p["batch_size"]
        SLEEP = cfg_p["inter_batch_sleep"]

        all_embeddings = []
        async with httpx.AsyncClient(timeout=60) as c:
            for i in range(0, len(texts), BATCH):
                batch = texts[i:i + BATCH]
                try:
                    r = await c.post(
                        f"{self.embed_url}/api/embed",
                        json={"model": self.embed_model, "input": batch},
                        headers=self._emb_headers,
                    )
                    if r.status_code == 200:
                        all_embeddings.extend(r.json().get("embeddings", []))
                        if i + BATCH < len(texts):
                            await _aio.sleep(SLEEP)
                        continue
                except Exception:
                    pass
                # Legacy fallback — one at a time
                for text in batch:
                    r = await c.post(
                        f"{self.embed_url}/api/embeddings",
                        json={"model": self.embed_model, "prompt": text},
                        headers=self._emb_headers,
                    )
                    r.raise_for_status()
                    all_embeddings.append(r.json()["embedding"])
                if i + BATCH < len(texts):
                    await _aio.sleep(SLEEP)
        return all_embeddings

    async def _embed_google(self, texts: list[str]) -> list[list[float]]:
        """Google Gemini batch embedding with provider-aware rate limiting."""
        _c = self._live_cfg()
        model = _c.google_embed_model
        cfg_p = self._PROVIDER_CONFIG["google"]
        BATCH = cfg_p["batch_size"]
        SLEEP = cfg_p["inter_batch_sleep"]
        RETRY_BASE = cfg_p["retry_sleep_base"]

        headers = {"Content-Type": "application/json",
                   "x-goog-api-key": _c.google_api_key}
        batch_url = (f"https://generativelanguage.googleapis.com/v1beta/models"
                     f"/{model}:batchEmbedContents")
        all_embeddings = []
        import asyncio as _aio
        import logging as _lg
        _log = _lg.getLogger("rag")

        async with httpx.AsyncClient(timeout=120) as c:
            for i in range(0, len(texts), BATCH):
                batch = texts[i:i + BATCH]
                payload = {
                    "requests": [
                        {
                            "model": f"models/{model}",
                            "content": {"parts": [{"text": t}]},
                            "taskType": "RETRIEVAL_DOCUMENT",
                            "outputDimensionality": _c.embed_dimensions
                        }
                        for t in batch
                    ]
                }
                for attempt in range(3):  # max 3 attempts per batch
                    try:
                        r = await c.post(batch_url, json=payload, headers=headers)
                        if r.status_code == 429:
                            # Check if daily quota exhausted (won't recover with retries)
                            err = r.json().get("error", {})
                            status_str = err.get("status", "")
                            if "RESOURCE_EXHAUSTED" in str(err) and attempt >= 1:
                                _log.error(
                                    "Google daily quota exhausted — "
                                    "falling back to Ollama for remaining chunks"
                                )
                                # Fall back to Ollama for remaining texts
                                remaining = texts[i:]
                                fallback = await self._embed_ollama(remaining)
                                all_embeddings.extend(fallback)
                                return all_embeddings
                            wait = RETRY_BASE * (2 ** attempt)
                            _log.warning("Google 429 — waiting %ds (attempt %d)", wait, attempt+1)
                            await _aio.sleep(wait)
                            continue
                        r.raise_for_status()
                        for emb in r.json().get("embeddings", []):
                            all_embeddings.append(emb["values"])
                        if i + BATCH < len(texts):
                            await _aio.sleep(SLEEP)
                        break
                    except Exception as _e:
                        if attempt == 2:
                            _log.error("Google embed batch %d failed — falling back to Ollama", i//BATCH)
                            remaining = texts[i:]
                            fallback = await self._embed_ollama(remaining)
                            all_embeddings.extend(fallback)
                            return all_embeddings
                        await _aio.sleep(RETRY_BASE)
        return all_embeddings

    async def _embed_jina(self, texts: list[str]) -> list[list[float]]:
        """Jina AI embeddings — jina-embeddings-v3, 100 RPM, no daily limit."""
        _c = self._live_cfg()
        cfg_p = self._PROVIDER_CONFIG["jina"]
        BATCH = cfg_p["batch_size"]
        SLEEP = cfg_p["inter_batch_sleep"]
        RETRY_BASE = cfg_p["retry_sleep_base"]

        url = "https://api.jina.ai/v1/embeddings"
        headers = {"Authorization": f"Bearer {_c.jina_api_key}",
                   "Content-Type": "application/json",
                   "Accept": "application/json"}
        all_embeddings = []
        import asyncio as _aio
        import logging as _lg
        _log = _lg.getLogger("rag")

        async with httpx.AsyncClient(timeout=120) as c:
            for i in range(0, len(texts), BATCH):
                batch = texts[i:i + BATCH]
                # Filter empty/whitespace-only texts before sending
                clean_batch = [t.strip() for t in batch]
                clean_batch = [t if t else "." for t in clean_batch]  # replace empty with placeholder

                payload = {
                    "model": _c.jina_embed_model,
                    "input": clean_batch,
                    "task": "retrieval.passage",  # optimised for RAG document indexing
                    # Note: "dimensions" only supported on jina-embeddings-v3 with MRL
                    # omit to get model default (1024 for v3)
                }
                for attempt in range(5):
                    try:
                        r = await c.post(url, json=payload, headers=headers)
                        if r.status_code == 429:
                            wait = RETRY_BASE * (2 ** attempt)
                            _log.warning("Jina 429 — waiting %ds (attempt %d)", wait, attempt+1)
                            await _aio.sleep(wait)
                            continue
                        r.raise_for_status()
                        data = r.json()
                        # Response: {"data": [{"embedding": [...], "index": N}]}
                        for item in sorted(data["data"], key=lambda x: x["index"]):
                            all_embeddings.append(item["embedding"])
                        if i + BATCH < len(texts):
                            await _aio.sleep(SLEEP)
                        break
                    except Exception as _e:
                        if attempt == 4:
                            raise
                        await _aio.sleep(RETRY_BASE * (2 ** attempt))
        return all_embeddings

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """OpenAI embeddings API."""
        _c = self._live_cfg()
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {_c.openai_api_key}",
                   "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(url, headers=headers,
                json={"model": _c.openai_embed_model, "input": texts})
            r.raise_for_status()
            data = r.json()
            return [item["embedding"] for item in sorted(
                data["data"], key=lambda x: x["index"])]

    async def _embed_cohere(self, texts: list[str]) -> list[list[float]]:
        """Cohere embeddings — 20 RPM free tier, batch with sleep."""
        _c = self._live_cfg()
        cfg_p = self._PROVIDER_CONFIG["cohere"]
        BATCH = cfg_p["batch_size"]
        SLEEP = cfg_p["inter_batch_sleep"]
        RETRY_BASE = cfg_p["retry_sleep_base"]

        url = "https://api.cohere.com/v2/embed"
        headers = {"Authorization": f"Bearer {_c.cohere_api_key}",
                   "Content-Type": "application/json"}
        all_embeddings = []
        import asyncio as _aio
        async with httpx.AsyncClient(timeout=60) as c:
            for i in range(0, len(texts), BATCH):
                batch = texts[i:i + BATCH]
                for attempt in range(5):
                    try:
                        r = await c.post(url, headers=headers,
                            json={"model": _c.cohere_embed_model, "texts": batch,
                                  "input_type": "search_document",
                                  "embedding_types": ["float"]})
                        if r.status_code == 429:
                            wait = RETRY_BASE * (2 ** attempt)
                            await _aio.sleep(wait)
                            continue
                        r.raise_for_status()
                        all_embeddings.extend(r.json()["embeddings"]["float"])
                        if i + BATCH < len(texts):
                            await _aio.sleep(SLEEP)
                        break
                    except Exception as _e:
                        if attempt == 4:
                            raise
                        await _aio.sleep(RETRY_BASE)
        return all_embeddings

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self.embed(texts))

    # ── Chat inference ────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": self.inference_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                f"{self.inference_url}/api/chat",
                json=payload,
                headers=self._inf_headers,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.inference_model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120) as c:
            async with c.stream(
                "POST",
                f"{self.inference_url}/api/chat",
                json=payload,
                headers=self._inf_headers,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except Exception:
                        continue

    # ── LangChain embeddings wrapper ──────────────────────────────────────────

    def as_langchain_embeddings(self):
        """Return a LangChain OllamaEmbeddings object, with auth headers if configured."""
        from langchain_ollama import OllamaEmbeddings
        kwargs = dict(base_url=self.embed_url, model=self.embed_model)
        # LangChain OllamaEmbeddings supports custom headers via client_kwargs
        if self._emb_headers:
            kwargs["client_kwargs"] = {"headers": self._emb_headers}
        return OllamaEmbeddings(**kwargs)


# Singleton
_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client