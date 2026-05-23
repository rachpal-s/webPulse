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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        async with httpx.AsyncClient(timeout=60) as c:
            for text in texts:
                r = await c.post(
                    f"{self.embed_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                    headers=self._emb_headers,
                )
                r.raise_for_status()
                embeddings.append(r.json()["embedding"])
        return embeddings

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