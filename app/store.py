"""Document store with hybrid search, all in one SQLite file.

- Semantic search: each chunk's embedding is stored as a BLOB; at query time we
  compare the question's embedding to every chunk (cosine similarity, NumPy).
- Keyword search: SQLite's built-in FTS5 full-text index ranks chunks with BM25,
  which catches exact terms (names, codes, numbers) that embeddings can miss.
- The two rankings are merged with Reciprocal Rank Fusion (RRF).

Brute-force vector search is fast for thousands of chunks; for millions you
would swap in a vector database (pgvector, Qdrant, ...) behind the same search().
"""

import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.embeddings import Embedder
from app.ingest import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    pages       INTEGER,
    chunks      INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        INTEGER,
    position    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text);
"""

RRF_K = 60  # standard RRF constant: dampens the advantage of the very top ranks
CANDIDATES = 20  # how many results each method contributes before fusion
MIN_VECTOR_SCORE = 0.2  # below this, a chunk is only kept if it also matched keywords

STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its
me my of on or our should so that the their there this to was we what when where which who
why will with you your""".split())


@dataclass
class SearchHit:
    chunk_id: int
    document_id: int
    document: str
    page: int | None
    text: str
    score: float  # fused RRF score, used for ordering
    vector_score: float  # cosine similarity, -1..1


class DocumentStore:
    def __init__(self, db_path: Path, embedder: Embedder):
        self.db_path = db_path
        self.embedder = embedder
        self._lock = threading.Lock()
        self._matrix_cache: tuple[np.ndarray, np.ndarray] | None = None  # (chunk ids, vectors)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # --- Writing -----------------------------------------------------------

    def add_document(self, name: str, chunks: list[Chunk], pages: int | None = None) -> dict:
        vectors = self.embedder.embed([c.text for c in chunks])
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            doc_id = conn.execute(
                "INSERT INTO documents (name, pages, chunks, created_at) VALUES (?, ?, ?, ?)",
                (name, pages, len(chunks), created)).lastrowid
            for position, (chunk, vector) in enumerate(zip(chunks, vectors)):
                chunk_id = conn.execute(
                    "INSERT INTO chunks (document_id, page, position, text, embedding) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, chunk.page, position, chunk.text, vector.astype(np.float32).tobytes())).lastrowid
                conn.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (chunk_id, chunk.text))
            self._matrix_cache = None
        return {"id": doc_id, "name": name, "pages": pages, "chunks": len(chunks), "created_at": created}

    def delete_document(self, document_id: int) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id = ?)",
                         (document_id,))
            deleted = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,)).rowcount
            self._matrix_cache = None
        return deleted > 0

    # --- Reading -----------------------------------------------------------

    def list_documents(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT id, name, pages, chunks, created_at FROM documents ORDER BY id")]

    def get_chunks(self, chunk_ids: list[int]) -> dict[int, SearchHit]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT c.id, c.document_id, d.name, c.page, c.text
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders})""", chunk_ids)
            return {r["id"]: SearchHit(r["id"], r["document_id"], r["name"], r["page"], r["text"], 0.0, 0.0)
                    for r in rows}

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        vector_ranked = self._vector_search(query)
        keyword_ranked = self._keyword_search(query)
        vector_scores = dict(vector_ranked)

        # Reciprocal Rank Fusion: a chunk ranked high by either method scores high.
        fused: dict[int, float] = {}
        for ranking in (vector_ranked, keyword_ranked):
            for rank, (chunk_id, _) in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        keyword_ids = {cid for cid, _ in keyword_ranked}
        relevant = [cid for cid in fused
                    if cid in keyword_ids or vector_scores.get(cid, -1.0) >= MIN_VECTOR_SCORE]
        best = sorted(relevant, key=lambda cid: fused[cid], reverse=True)[:top_k]

        chunks = self.get_chunks(best)
        hits = []
        for cid in best:
            hit = chunks[cid]
            hit.score = round(fused[cid], 5)
            hit.vector_score = round(vector_scores.get(cid, self._similarity(query, cid)), 3)
            hits.append(hit)
        return hits

    def _load_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """All chunk vectors as one matrix, cached until documents change."""
        with self._lock:
            if self._matrix_cache is None:
                with self._connect() as conn:
                    rows = conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
                ids = np.array([r["id"] for r in rows], dtype=np.int64)
                matrix = (np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                          if rows else np.zeros((0, 0), dtype=np.float32))
                self._matrix_cache = (ids, matrix)
            return self._matrix_cache

    def _vector_search(self, query: str) -> list[tuple[int, float]]:
        ids, matrix = self._load_matrix()
        if len(ids) == 0:
            return []
        scores = matrix @ self.embedder.embed([query])[0]  # cosine similarity (vectors are normalised)
        top = np.argsort(-scores)[:CANDIDATES]
        return [(int(ids[i]), float(scores[i])) for i in top]

    def _keyword_search(self, query: str) -> list[tuple[int, float]]:
        terms = [t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS]
        if not terms:
            return []
        # Quote each term so user input can't use FTS5 query syntax; OR them together.
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY rank LIMIT ?", (fts_query, CANDIDATES)).fetchall()
        return [(r["rowid"], -r["rank"]) for r in rows]  # bm25() is lower-is-better

    def _similarity(self, query: str, chunk_id: int) -> float:
        ids, matrix = self._load_matrix()
        index = int(np.searchsorted(ids, chunk_id))
        return float(matrix[index] @ self.embedder.embed([query])[0])
