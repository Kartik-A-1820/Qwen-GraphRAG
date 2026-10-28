"""Canonical ingest ledger; vector/BM25/graph indexes are repairable projections."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from langchain_core.documents import Document


class IngestCatalog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS files(hash TEXT PRIMARY KEY, ready INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks(
                    id TEXT PRIMARY KEY, file_hash TEXT NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS graph_cache(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY CHECK(id=1), dirty INTEGER NOT NULL);
                INSERT OR IGNORE INTO state VALUES(1, 0);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def empty(self):
        with self.connection() as db:
            return db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0

    def bind_index_profile(self, profile):
        with self.connection() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='index_profile'").fetchone()
            if row and row[0] != profile:
                raise ValueError("Embedding/chunking configuration changed. Use a fresh data directory or an explicit migration.")
            db.execute("INSERT OR IGNORE INTO metadata VALUES('index_profile', ?)", (profile,))

    def ready(self, file_hash):
        with self.connection() as db:
            row = db.execute("SELECT ready FROM files WHERE hash=?", (file_hash,)).fetchone()
            return bool(row and row[0])

    def dirty(self):
        with self.connection() as db:
            return bool(db.execute("SELECT dirty FROM state WHERE id=1").fetchone()[0])

    def prepare(self, file_hash, documents):
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO files VALUES(?, 0)", (file_hash,))
            for doc_id, doc in documents.items():
                db.execute("INSERT OR IGNORE INTO chunks VALUES(?, ?, ?, ?)",
                           (doc_id, file_hash, doc.page_content, json.dumps(doc.metadata)))
            db.execute("UPDATE state SET dirty=1 WHERE id=1")

    def documents(self, pending_only=False):
        query = "SELECT c.id, c.text, c.metadata FROM chunks c JOIN files f ON c.file_hash=f.hash"
        if pending_only:
            query += " WHERE f.ready=0"
        query += " ORDER BY c.id"
        with self.connection() as db:
            return {key: Document(page_content=text, metadata=json.loads(metadata))
                    for key, text, metadata in db.execute(query)}

    def cached_graph(self, key):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM graph_cache WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def cache_graph(self, key, payload):
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO graph_cache VALUES(?, ?)", (key, json.dumps(payload)))
            db.execute("UPDATE state SET dirty=1 WHERE id=1")

    def profile(self):
        with self.connection() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='profile'").fetchone()
            return row[0] if row else None

    def finish(self, profile):
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES('profile', ?)", (profile,))
            db.execute("UPDATE files SET ready=1")
            db.execute("UPDATE state SET dirty=0 WHERE id=1")


def chunk_id(file_hash, index, text):
    return hashlib.sha256(f"{file_hash}:{index}:{text}".encode()).hexdigest()
