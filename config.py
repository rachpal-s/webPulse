"""config.py — centralised settings loaded from .env"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama
    ollama_inference_url: str = "http://localhost:11434"
    ollama_embed_url: str = "http://localhost:11434"
    ollama_inference_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_inference_api_key: str = ""   # blank = no auth (default self-hosted Ollama)
    ollama_embed_api_key: str = ""       # blank = no auth; set if embed endpoint is gated

    # RAG thresholds
    rag_small_threshold: int = 2000
    rag_medium_threshold: int = 15000

    # Chunking
    chunk_breakpoint_type: str = "percentile"
    chunk_breakpoint_threshold: int = 95
    chunk_min_size: int = 150
    chunk_max_size: int = 800

    # SQLite
    db_path: str = "data/webpulse.db"
    embed_dimensions: int = 768

    # Scraper
    scraper_timeout: int = 28
    scraper_max_headlines: int = 50
    playwright_wait_seconds: float = 4.0
    playwright_headless: bool = True

    # Morning brief batch
    brief_auto_run: bool = True          # set false to disable auto morning run
    brief_start_time: str = "08:15"
    brief_articles_per_site: int = 10
    brief_timezone: str = "Asia/Kolkata"

    # SMTP
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_to: str = ""
    smtp_enabled: bool = False
    portfolio_symbols: str = ""              # fallback if DB portfolio empty e.g. "RELIANCE,TCS"
    portfolio_analysis_enabled: bool = True  # set false to skip per-holding Q&A in brief
    brief_prompts_json: str = ""             # JSON override for DEFAULT_PROMPTS; empty = use defaults

    @property
    def brief_default_prompts(self) -> list[dict]:
        """Default insight prompts for morning brief. Override via BRIEF_PROMPTS_JSON."""
        return [
            {
                "key": "trending_news",
                "label": "Top Trending News",
                "prompt": (
                    "What are the top 10 trending news stories today that may impact market dynamics? "
                    "List them as an HTML numbered list with a one-line explanation of potential market impact for each."
                ),
            },
            {
                "key": "market_outlook",
                "label": "India Market Outlook",
                "prompt": (
                    "Based on today's news context, how is the Indian stock market (Sensex/Nifty) likely to behave today? "
                    "Consider global cues, FII/DII activity, sector trends, and macro factors. "
                    "Give a clear directional view: bullish, bearish, or range-bound, with key reasons."
                ),
            },
            {
                "key": "stock_calls",
                "label": "Expert Stock Recommendations",
                "prompt": (
                    "Based on the news context, which specific stocks have been explicitly recommended by analysts or experts? "
                    "Create an HTML table with columns: Stock, Recommendation (BUY/SELL/HOLD), Target Price (if mentioned), "
                    "Analyst/Source, and Key Reason. Only include stocks with explicit recommendations in the news."
                ),
            },
            {
                "key": "focus_areas",
                "label": "Focus Areas Today",
                "prompt": (
                    "Based on today's news, what are the key focus areas, themes, or sectors that investors should watch today? "
                    "Include: sectors in spotlight, key events or data releases, geopolitical factors, and any earnings announcements. "
                    "Format as an HTML table with columns: Area, Why It Matters, Likely Impact."
                ),
            },
            {
                "key": "risk_factors",
                "label": "Risk Factors & Caution Zones",
                "prompt": (
                    "Based on today's news context, what are the key risk factors or caution zones for the market today? "
                    "Include global risks, domestic concerns, overvalued sectors, or stocks facing headwinds. "
                    "Be specific and factual. Format as a concise HTML bulleted list."
                ),
            },
        ]

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = True
    app_title: str = "WebPulse"


@lru_cache
def get_settings() -> Settings:
    return Settings()