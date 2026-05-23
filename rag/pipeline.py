"""
rag/pipeline.py — Adaptive RAG pipeline.

Mode selection based on word count:
  small  (< RAG_SMALL_THRESHOLD)  → direct LLM context, no chunking/embedding
  medium (< RAG_MEDIUM_THRESHOLD) → semantic chunking + SQLite-vec search
  large  (≥ RAG_MEDIUM_THRESHOLD) → full RAG: metadata-aware chunking + vector search
"""
import asyncio
import hashlib
import json
import re
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import get_settings
from rag.ollama import get_ollama_client

cfg = get_settings()


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    source_url: str
    source_title: str
    content: str
    word_count: int
    chunk_index: int
    total_chunks: int
    section: str = ""
    page_type: str = ""
    embedding: Optional[list[float]] = None


@dataclass
class RAGContext:
    session_id: str
    mode: str                           # "small" | "medium" | "large"
    total_words: int
    chunk_count: int
    sources: list[dict]                 # [{url, title}]
    ready: bool = False
    error: Optional[str] = None


@dataclass
class QAResult:
    question: str
    answer: str
    mode: str
    sources_used: list[dict]
    latency_ms: float
    chunks_retrieved: int = 0


# ── SQLite vector store ───────────────────────────────────────────────────────

class VectorStore:
    """SQLite + sqlite-vec for chunk storage and similarity search."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        import sqlite_vec
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate_db(self, conn):
        """Add columns/tables that may be missing in older DB versions."""
        migrations = [
            "ALTER TABLE url_history ADD COLUMN is_daily INTEGER DEFAULT 0",
            "ALTER TABLE url_history ADD COLUMN last_briefed TEXT DEFAULT ''",
            """CREATE TABLE IF NOT EXISTS brief_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brief_date TEXT NOT NULL,
                prompt_key TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                answer_html TEXT DEFAULT '',
                generated_at REAL,
                UNIQUE(brief_date, prompt_key)
            )""",
            "ALTER TABLE brief_insights ADD COLUMN sources_json TEXT DEFAULT '{}'",
        ]
        for sql in migrations:
            try:
                conn.execute(sql)
            except Exception:
                pass  # column already exists — ignore
        conn.commit()

    def _init_db(self):
        conn = self._connect()
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL,
                mode TEXT,
                total_words INTEGER,
                sources TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                source_url TEXT,
                source_title TEXT,
                content TEXT NOT NULL,
                word_count INTEGER,
                chunk_index INTEGER,
                total_chunks INTEGER,
                section TEXT DEFAULT '',
                page_type TEXT DEFAULT '',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[{cfg.embed_dimensions}]
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_session
                ON chunks(session_id);

            CREATE TABLE IF NOT EXISTS url_history (
                url TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                last_scraped REAL NOT NULL,
                scrape_count INTEGER DEFAULT 1,
                page_type TEXT DEFAULT '',
                is_daily INTEGER DEFAULT 0,      -- 1 = include in morning brief
                last_briefed TEXT DEFAULT ''     -- date of last morning brief run
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                qty REAL DEFAULT 0,
                avg_price REAL DEFAULT 0,
                sector TEXT DEFAULT '',
                exchange TEXT DEFAULT 'NSE',
                notes TEXT DEFAULT '',
                active INTEGER DEFAULT 1,
                created_at REAL,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS morning_briefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brief_date TEXT NOT NULL UNIQUE,  -- YYYY-MM-DD
                status TEXT DEFAULT 'pending',    -- pending/running/done/failed
                started_at REAL,
                completed_at REAL,
                rag_session_id TEXT DEFAULT '',
                articles_scraped INTEGER DEFAULT 0,
                html_content TEXT DEFAULT '',
                error_msg TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_briefs_date
                ON morning_briefs(brief_date);

            CREATE TABLE IF NOT EXISTS brief_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brief_date TEXT NOT NULL,
                prompt_key TEXT NOT NULL,   -- e.g. "trending_news", "market_outlook"
                prompt_text TEXT NOT NULL,
                answer_html TEXT DEFAULT '',
                generated_at REAL,
                UNIQUE(brief_date, prompt_key)
            );
        """)
        conn.commit()
        conn.close()
        # Run migrations on a fresh connection after schema init
        conn2 = self._connect()
        self._migrate_db(conn2)
        conn2.close()

    def save_session(self, ctx: RAGContext):
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO sessions
            (session_id, created_at, mode, total_words, sources)
            VALUES (?, ?, ?, ?, ?)
        """, (ctx.session_id, time.time(), ctx.mode,
              ctx.total_words, json.dumps(ctx.sources)))
        conn.commit()
        conn.close()

    def save_chunks(self, session_id: str, chunks: list[Chunk]):
        conn = self._connect()
        for chunk in chunks:
            conn.execute("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, session_id, source_url, source_title, content,
                 word_count, chunk_index, total_chunks, section, page_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (chunk.chunk_id, session_id, chunk.source_url,
                  chunk.source_title, chunk.content, chunk.word_count,
                  chunk.chunk_index, chunk.total_chunks,
                  chunk.section, chunk.page_type))

            if chunk.embedding:
                serialized = struct.pack(
                    f"{len(chunk.embedding)}f", *chunk.embedding
                )
                conn.execute("""
                    INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding)
                    VALUES (?, ?)
                """, (chunk.chunk_id, serialized))

        conn.commit()
        conn.close()

    def similarity_search(
        self,
        session_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        conn = self._connect()
        serialized = struct.pack(f"{len(query_embedding)}f", *query_embedding)
        rows = conn.execute(f"""
            SELECT c.chunk_id, c.content, c.source_url, c.source_title,
                   c.section, c.chunk_index, c.word_count,
                   vec_distance_cosine(cv.embedding, ?) AS distance
            FROM chunk_vectors cv
            JOIN chunks c ON cv.chunk_id = c.chunk_id
            WHERE c.session_id = ?
            ORDER BY distance ASC
            LIMIT ?
        """, (serialized, session_id, top_k)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_all_chunks(self, session_id: str) -> list[dict]:
        conn = self._connect()
        rows = conn.execute("""
            SELECT * FROM chunks WHERE session_id = ?
            ORDER BY source_url, chunk_index
        """, (session_id,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        conn.close()
        if row:
            d = dict(row)
            d["sources"] = json.loads(d["sources"])
            return d
        return None

    def delete_session(self, session_id: str):
        conn = self._connect()
        conn.execute("DELETE FROM chunk_vectors WHERE chunk_id IN "
                     "(SELECT chunk_id FROM chunks WHERE session_id = ?)", (session_id,))
        conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

    # ── URL history ───────────────────────────────────────────────────────────

    # ── Portfolio methods ─────────────────────────────────────────────────────

    def get_portfolio(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM portfolio WHERE active=1 ORDER BY symbol"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def upsert_holding(self, symbol: str, name: str, qty: float,
                       avg_price: float, sector: str = "",
                       exchange: str = "NSE", notes: str = "",
                       holding_id: int = None):
        import time as _time
        conn = self._connect()
        now = _time.time()
        if holding_id:
            conn.execute("""
                UPDATE portfolio SET symbol=?,name=?,qty=?,avg_price=?,
                sector=?,exchange=?,notes=?,updated_at=? WHERE id=?
            """, (symbol,name,qty,avg_price,sector,exchange,notes,now,holding_id))
        else:
            conn.execute("""
                INSERT INTO portfolio (symbol,name,qty,avg_price,sector,
                exchange,notes,active,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,1,?,?)
            """, (symbol,name,qty,avg_price,sector,exchange,notes,now,now))
        conn.commit()
        conn.close()

    def delete_holding(self, holding_id: int):
        conn = self._connect()
        conn.execute("UPDATE portfolio SET active=0 WHERE id=?", (holding_id,))
        conn.commit()
        conn.close()

    # ── Morning brief methods ──────────────────────────────────────────────────

    def get_brief(self, date_str: str) -> Optional[dict]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM morning_briefs WHERE brief_date=?", (date_str,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_recent_briefs(self, limit: int = 10) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT id,brief_date,status,started_at,completed_at,articles_scraped "
            "FROM morning_briefs ORDER BY brief_date DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def start_brief(self, date_str: str) -> int:
        import time as _time
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO morning_briefs
            (brief_date,status,started_at,html_content,error_msg)
            VALUES (?,'running',?,'','')
        """, (date_str, _time.time()))
        conn.commit()
        row = conn.execute(
            "SELECT id FROM morning_briefs WHERE brief_date=?", (date_str,)
        ).fetchone()
        conn.close()
        return row[0]

    def finish_brief(self, date_str: str, html: str, session_id: str,
                     articles: int, error: str = ""):
        import time as _time
        status = "done" if not error else "failed"
        conn = self._connect()
        conn.execute("""
            UPDATE morning_briefs SET status=?,completed_at=?,
            rag_session_id=?,articles_scraped=?,html_content=?,error_msg=?
            WHERE brief_date=?
        """, (status, _time.time(), session_id, articles, html, error, date_str))
        conn.commit()
        conn.close()

    def save_insight(self, date_str: str, prompt_key: str,
                     prompt_text: str, answer_html: str, sources: list = None):
        import time as _time, json as _json
        conn = self._connect()
        # Add sources_json column if missing
        try:
            conn.execute("ALTER TABLE brief_insights ADD COLUMN sources_json TEXT DEFAULT '{}'")
            conn.commit()
        except Exception:
            pass
        sources_json = _json.dumps({s.get("url",""): s.get("title","") for s in (sources or []) if s.get("url")})
        conn.execute("""
            INSERT OR REPLACE INTO brief_insights
            (brief_date, prompt_key, prompt_text, answer_html, generated_at, sources_json)
            VALUES (?,?,?,?,?,?)
        """, (date_str, prompt_key, prompt_text, answer_html, _time.time(), sources_json))
        conn.commit()
        conn.close()

    def get_insights(self, date_str: str) -> list[dict]:
        conn = self._connect()
        rows = conn.execute("""
            SELECT prompt_key, prompt_text, answer_html, generated_at,
                   COALESCE(sources_json, '{}') as sources_json
            FROM brief_insights WHERE brief_date=?
            ORDER BY id
        """, (date_str,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]



    def set_url_daily(self, url: str, is_daily: bool):
        import time as _time
        conn = self._connect()
        # Upsert — row may not exist if URL was never scraped via the app
        conn.execute("""
            INSERT INTO url_history (url, last_scraped, scrape_count, is_daily)
            VALUES (?, ?, 0, ?)
            ON CONFLICT(url) DO UPDATE SET is_daily=excluded.is_daily
        """, (url, _time.time(), 1 if is_daily else 0))
        conn.commit()
        conn.close()

    def get_daily_urls(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM url_history WHERE is_daily=1 ORDER BY last_scraped DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── URL history ───────────────────────────────────────────────────────────

    def record_url(self, url: str, title: str = "", page_type: str = ""):
        """Upsert a URL into history for autocomplete."""
        conn = self._connect()
        conn.execute("""
            INSERT INTO url_history (url, title, last_scraped, scrape_count, page_type)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(url) DO UPDATE SET
                title = CASE WHEN excluded.title != '' THEN excluded.title ELSE url_history.title END,
                last_scraped = excluded.last_scraped,
                scrape_count = url_history.scrape_count + 1,
                page_type = CASE WHEN excluded.page_type != '' THEN excluded.page_type ELSE url_history.page_type END
        """, (url, title or "", time.time(), page_type or ""))
        conn.commit()
        conn.close()

    def get_url_history(self, prefix: str = "", limit: int = 20) -> list[dict]:
        """Return URLs matching prefix, ordered by most recently used."""
        conn = self._connect()
        # Explicitly include is_daily with COALESCE fallback for old DBs
        select = """
            SELECT url, title, last_scraped, scrape_count, page_type,
                   COALESCE(is_daily, 0) as is_daily,
                   COALESCE(last_briefed, '') as last_briefed
            FROM url_history
        """
        if prefix:
            rows = conn.execute(
                select + " WHERE url LIKE ? OR title LIKE ? ORDER BY last_scraped DESC LIMIT ?",
                (f"%{prefix}%", f"%{prefix}%", limit)
            ).fetchall()
        else:
            rows = conn.execute(
                select + " ORDER BY last_scraped DESC LIMIT ?", (limit,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


# ── Chunker ────────────────────────────────────────────────────────────────────

class AdaptiveChunker:
    """
    Two-phase chunking:
      medium → SemanticChunker (embedding-based sentence grouping)
      large  → SemanticChunker + metadata-enriched splits
    Falls back to RecursiveCharacterTextSplitter if embeddings unavailable.
    """

    def __init__(self):
        self._lc_embeddings = None

    def _get_lc_embeddings(self):
        if self._lc_embeddings is None:
            client = get_ollama_client()
            self._lc_embeddings = client.as_langchain_embeddings()
        return self._lc_embeddings

    def chunk_document(
        self,
        content: str,
        source_url: str,
        source_title: str,
        page_type: str,
        mode: str,
    ) -> list[Chunk]:
        """Split a document into semantically coherent chunks with metadata."""

        # ── Try semantic chunking ─────────────────────────────────────────────
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            embeddings = self._get_lc_embeddings()
            splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type=cfg.chunk_breakpoint_type,
                breakpoint_threshold_amount=cfg.chunk_breakpoint_threshold,
            )
            docs = splitter.create_documents([content])
            raw_chunks = [d.page_content for d in docs]
        except Exception:
            # Fallback: recursive character splitter
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg.chunk_max_size * 5,  # chars
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " "],
            )
            raw_chunks = splitter.split_text(content)

        # ── Filter too-small chunks ───────────────────────────────────────────
        raw_chunks = [c.strip() for c in raw_chunks
                      if len(c.split()) >= cfg.chunk_min_size]
        if not raw_chunks:
            raw_chunks = [content]

        # ── Build Chunk objects with metadata ─────────────────────────────────
        total = len(raw_chunks)
        chunks: list[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            section = _infer_section(text)
            chunk_id = _make_chunk_id(source_url, idx)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_url=source_url,
                source_title=source_title,
                content=text,
                word_count=len(text.split()),
                chunk_index=idx,
                total_chunks=total,
                section=section,
                page_type=page_type,
            ))
        return chunks


def _infer_section(text: str) -> str:
    """Extract first heading-like line as section label."""
    for line in text.split("\n")[:5]:
        line = line.strip()
        if line.startswith("##"):
            return re.sub(r"^#+\s*", "", line)[:80]
        if len(line) > 10 and len(line) < 100 and line[0].isupper():
            return line[:80]
    return ""


def _make_chunk_id(url: str, idx: int) -> str:
    h = hashlib.md5(f"{url}::{idx}".encode()).hexdigest()[:12]
    return f"chunk_{h}_{idx}"


def _make_session_id(urls: list[str]) -> str:
    combined = "|".join(sorted(urls))
    return hashlib.md5(combined.encode()).hexdigest()[:16]


# ── Main RAG pipeline ─────────────────────────────────────────────────────────

_store: Optional[VectorStore] = None
_chunker: Optional[AdaptiveChunker] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(cfg.db_path)
    return _store


def get_chunker() -> AdaptiveChunker:
    global _chunker
    if _chunker is None:
        _chunker = AdaptiveChunker()
    return _chunker


async def ingest_documents(
    documents: list[dict],   # [{url, title, content, page_type}]
) -> RAGContext:
    """
    Ingest a list of scraped documents into the RAG pipeline.
    Automatically selects mode based on total word count.
    """
    total_words = sum(len(d.get("content","").split()) for d in documents)
    session_id = _make_session_id([d["url"] for d in documents])
    sources = [{"url": d["url"], "title": d.get("title", d["url"])} for d in documents]

    # Determine mode
    if total_words < cfg.rag_small_threshold:
        mode = "small"
    elif total_words < cfg.rag_medium_threshold:
        mode = "medium"
    else:
        mode = "large"

    ctx = RAGContext(
        session_id=session_id,
        mode=mode,
        total_words=total_words,
        chunk_count=0,
        sources=sources,
    )

    store = get_store()
    store.save_session(ctx)

    # Small mode: store content as single chunks (no embeddings needed)
    if mode == "small":
        chunks: list[Chunk] = []
        for doc in documents:
            chunk_id = _make_chunk_id(doc["url"], 0)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_url=doc["url"],
                source_title=doc.get("title", ""),
                content=doc.get("content", ""),
                word_count=len(doc.get("content","").split()),
                chunk_index=0, total_chunks=1,
                page_type=doc.get("page_type",""),
            ))
        store.save_chunks(session_id, chunks)
        ctx.chunk_count = len(chunks)
        ctx.ready = True
        return ctx

    # Medium / Large: semantic chunking + embeddings
    chunker = get_chunker()
    ollama = get_ollama_client()

    all_chunks: list[Chunk] = []
    for doc in documents:
        content = doc.get("content","")
        if not content:
            continue
        doc_chunks = await asyncio.get_event_loop().run_in_executor(
            None,
            chunker.chunk_document,
            content, doc["url"], doc.get("title",""),
            doc.get("page_type",""), mode,
        )
        all_chunks.extend(doc_chunks)

    # Generate embeddings in batches
    BATCH_SIZE = 16
    texts = [c.content for c in all_chunks]
    try:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            embs = await ollama.embed(batch)
            all_embeddings.extend(embs)

        for chunk, emb in zip(all_chunks, all_embeddings):
            chunk.embedding = emb
    except Exception as e:
        # If embeddings fail, still save chunks for keyword fallback
        ctx.error = f"Embeddings failed: {e}. Using keyword fallback."

    store.save_chunks(session_id, all_chunks)
    ctx.chunk_count = len(all_chunks)
    ctx.ready = True
    return ctx


async def query(
    session_id: str,
    question: str,
    top_k: int = 6,
) -> QAResult:
    """Answer a question using the ingested RAG context."""
    t0 = time.perf_counter()
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        return QAResult(
            question=question, answer="Session not found. Please re-ingest the content.",
            mode="unknown", sources_used=[], latency_ms=0,
        )

    mode = session["mode"]
    ollama = get_ollama_client()
    sources_used: list[dict] = []
    context_text = ""
    chunks_used = 0

    if mode == "small":
        # Direct context: concatenate all chunks
        chunks = store.get_all_chunks(session_id)
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['source_title'] or c['source_url']}]\n{c['content']}"
            for c in chunks
        )
        sources_used = session["sources"]
        chunks_used = len(chunks)

    else:
        # Vector search
        try:
            # Retry once — Ollama may have unloaded after long embedding session
            for _attempt in range(2):
                try:
                    q_emb = await ollama.embed([question])
                    break
                except Exception as _e:
                    if _attempt == 0:
                        import asyncio as _aio
                        await _aio.sleep(3)   # give Ollama time to reload
                    else:
                        raise
            relevant = store.similarity_search(session_id, q_emb[0], top_k=top_k)
            context_parts = []
            seen: dict = {}   # url -> title
            for row in relevant:
                title = row['source_title'] or row['source_url']
                url = row['source_url'] or ''
                if url:
                    seen[url] = title
                # Format: cite as [title](url) so LLM can copy it directly
                cite = f"[{title}]({url})" if url else title
                context_parts.append(
                    f"[CITE AS: {cite} | Section: {row.get('section','—')}]\n{row['content']}"
                )
            context_text = "\n\n---\n\n".join(context_parts)
            sources_used = [{"url": u, "title": t} for u, t in seen.items()]
            chunks_used = len(relevant)
        except Exception:
            # Fallback to keyword matching if vector search fails
            chunks = store.get_all_chunks(session_id)
            keywords = set(question.lower().split())
            scored = []
            for idx, c in enumerate(chunks):
                text_lower = c["content"].lower()
                score = sum(1 for kw in keywords if kw in text_lower)
                if score > 0:
                    scored.append((score, idx, c))  # idx breaks ties
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [c for _, _, c in scored[:top_k]]
            context_text = "\n\n---\n\n".join(
                f"[Source: {c['source_title']}]\n{c['content']}" for c in top
            )
            sources_used = session.get("sources") or []
            chunks_used = len(top)

    # ── Build prompt ──────────────────────────────────────────────────────────
    system_prompt = (
        "You are a precise research assistant. Answer questions strictly based on "
        "the provided context. If the answer is not in the context, say so clearly. "
        "Be concise, factual, and always cite your sources.\n\n"
        "CITATION RULES:\n"
        "- Every factual claim MUST be followed by a source link.\n"
        "- Citation format: [Source Title](URL) — Markdown link syntax.\n"
        "- Extract the URL from [Source: title | URL: ...] markers in context.\n"
        "- In tables, add a Source column with Markdown links.\n"
        "- Never make claims without citing a source from the provided context.\n\n"
        "FORMATTING RULES:\n"
        "- Always respond in clean HTML (no markdown, no triple backticks).\n"
        "- Use <table> with <thead>/<tbody>/<tr>/<th>/<td> for tabular data.\n"
        "- Use <ul>/<li> or <ol>/<li> for lists.\n"
        "- Use <strong> for bold, <em> for italic.\n"
        "- Use <p> for paragraphs. Use <h3> or <h4> for section headings.\n"
        "- Do NOT include <html>, <head>, <body>, <style>, or <script> tags.\n"
        "- Do NOT use inline styles or class attributes.\n"
        "- Keep the HTML clean and semantic."
    )
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the context above:"
    )

    answer = await ollama.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    return QAResult(
        question=question,
        answer=answer,
        mode=mode,
        sources_used=sources_used,
        latency_ms=(time.perf_counter() - t0) * 1000,
        chunks_retrieved=chunks_used,
    )