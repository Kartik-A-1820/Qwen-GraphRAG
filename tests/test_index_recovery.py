from dataclasses import replace

from langchain_core.documents import Document
import pytest

from backend.config import Settings
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.graph import RuleBasedGraphExtractor
from backend.retrieval.hybrid import HybridRetriever


class MemoryStore:
    def __init__(self):
        self.rows = {}
        self.fail_once = False

    def get(self, include=None):
        return {"ids": list(self.rows), "documents": [d.page_content for d in self.rows.values()],
                "metadatas": [dict(d.metadata) for d in self.rows.values()]}

    def add_texts(self, texts, metadatas, ids):
        for key, text, metadata in zip(ids, texts, metadatas):
            self.rows[key] = Document(page_content=text, metadata=dict(metadata))
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("injected partial vector write")

    def similarity_search(self, query, k):
        return list(self.rows.values())[:k]


def config(tmp_path):
    return Settings(graph_backend="json", graph_path=tmp_path / "graph.json")


def test_incremental_upload_and_restart_reuse_extraction(tmp_path, monkeypatch):
    calls = []
    original = RuleBasedGraphExtractor.extract

    def extract(self, text):
        calls.append(text)
        return original(self, text)

    monkeypatch.setattr(RuleBasedGraphExtractor, "extract", extract)
    store, settings = MemoryStore(), config(tmp_path)
    retriever = HybridRetriever(store, settings)
    retriever.add_documents([Document(page_content="Qwen supports retrieval.")], "file-a")
    retriever.add_documents([Document(page_content="Chroma stores embeddings.")], "file-b")
    assert len(calls) == 2
    assert len(retriever.graph.doc_metadata) == 2
    restarted = HybridRetriever(store, settings)
    assert len(calls) == 2
    assert restarted.add_documents([Document(page_content="ignored duplicate")], "file-a") == (False, 0)
    assert len(store.rows) == 2


def test_partial_vector_write_repairs_without_duplicate_ids(tmp_path):
    store, settings = MemoryStore(), config(tmp_path)
    retriever = HybridRetriever(store, settings)
    store.fail_once = True
    with pytest.raises(RuntimeError, match="partial vector"):
        retriever.add_documents([Document(page_content="alpha"), Document(page_content="beta")], "file-a")
    assert retriever.catalog.dirty()
    restarted = HybridRetriever(store, settings)
    assert len(store.rows) == 2
    assert restarted.catalog.ready("file-a")
    assert not restarted.catalog.dirty()
    assert len(restarted.retrieve("alpha").documents) == 2


def test_graph_write_failure_is_repaired_before_next_query(tmp_path, monkeypatch):
    store, settings = MemoryStore(), config(tmp_path)
    retriever = HybridRetriever(store, settings)
    original = retriever.graph.add_documents
    monkeypatch.setattr(retriever.graph, "add_documents", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("graph failure")))
    with pytest.raises(RuntimeError, match="graph failure"):
        retriever.add_documents([Document(page_content="Qwen uses Chroma")], "file-a")
    monkeypatch.setattr(retriever.graph, "add_documents", original)
    assert retriever.retrieve("Qwen").documents
    assert not retriever.catalog.dirty()


def test_incremental_bm25_matches_complete_rebuild():
    incremental, full = BM25Index(), BM25Index()
    incremental.add({"a": "invoice refund", "b": "payment policy"})
    incremental.add({"a": "updated refund policy", "c": "Qwen retrieval"})
    full.build({"a": "updated refund policy", "b": "payment policy", "c": "Qwen retrieval"})
    assert incremental.search("refund policy") == full.search("refund policy")


def test_cached_profile_switch_rebuilds_graph_projection(tmp_path):
    store, settings = MemoryStore(), config(tmp_path)
    a = HybridRetriever(store, settings)
    a.add_documents([Document(page_content="Qwen retrieval")], "a")
    b = HybridRetriever(store, replace(settings, graph_extraction_max_chunk_chars=100))
    assert b.catalog.profile() == b._profile()
    c = HybridRetriever(store, settings)
    assert c.catalog.profile() == c._profile()


def test_embedding_drift_requires_explicit_new_index(tmp_path):
    store, settings = MemoryStore(), config(tmp_path)
    HybridRetriever(store, settings)
    with pytest.raises(ValueError, match="configuration changed"):
        HybridRetriever(store, replace(settings, embedding_revision="different-revision"))
