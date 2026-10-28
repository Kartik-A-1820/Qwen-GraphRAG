import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("RAG_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_dir: Path = Path(os.getenv("RAG_CHROMA_DIR", PROJECT_ROOT / "data" / "chroma"))
    upload_dir: Path = Path(os.getenv("RAG_UPLOAD_DIR", PROJECT_ROOT / "data" / "uploads"))
    offload_dir: Path = Path(os.getenv("RAG_OFFLOAD_DIR", PROJECT_ROOT / "data" / "offload"))
    graph_path: Path = Path(os.getenv("RAG_GRAPH_PATH", PROJECT_ROOT / "data" / "graph_index.json"))
    graph_db_path: Path = Path(os.getenv("RAG_GRAPH_DB_PATH", PROJECT_ROOT / "data" / "kuzu_graph"))
    graph_backend: str = os.getenv("RAG_GRAPH_BACKEND", "kuzu")
    graph_buffer_pool_mb: int = int(os.getenv("RAG_GRAPH_BUFFER_POOL_MB", "128"))
    graph_extraction_mode: str = os.getenv("RAG_GRAPH_EXTRACTION_MODE", "rules")
    graph_extractor_model: str = os.getenv("RAG_GRAPH_EXTRACTOR_MODEL", os.getenv("RAG_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
    graph_extraction_max_new_tokens: int = int(os.getenv("RAG_GRAPH_EXTRACTION_MAX_NEW_TOKENS", "512"))
    graph_extraction_max_chunk_chars: int = int(os.getenv("RAG_GRAPH_EXTRACTION_MAX_CHUNK_CHARS", "2200"))
    collection_name: str = os.getenv("RAG_COLLECTION", "document_vector_collection")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "180"))
    dense_k: int = int(os.getenv("RAG_DENSE_K", "8"))
    sparse_k: int = int(os.getenv("RAG_SPARSE_K", "8"))
    graph_k: int = int(os.getenv("RAG_GRAPH_K", "8"))
    fused_k: int = int(os.getenv("RAG_FUSED_K", "5"))
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))
    max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
    max_new_tokens: int = int(os.getenv("RAG_MAX_NEW_TOKENS", "512"))
    load_in_4bit: bool = _bool_env("RAG_LOAD_IN_4BIT", False)
    device: str = os.getenv("RAG_DEVICE", "cpu")
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "")
    enable_reranker: bool = _bool_env("RAG_ENABLE_RERANKER", False)

    def ensure_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
