from langchain_core.documents import Document
import pytest

from backend.config import Settings
from backend.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from backend.evaluation.monitoring import MetricsMonitor
from backend.evaluation.run_retrieval_eval import evaluate_rows
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.fusion import RankedItem, reciprocal_rank_fusion
from backend.retrieval.graph import (
    GraphDocument,
    GraphEntity,
    GraphRelation,
    KnowledgeGraphIndex,
    LocalLLMGraphExtractor,
    extract_entities,
)
from backend.retrieval.rerankers import NoOpReranker


class FakeVectorStore:
    def __init__(self):
        self.ids = []
        self.documents = []
        self.metadatas = []

    def get(self, include=None):
        return {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }

    def add_texts(self, texts, metadatas, ids):
        self.ids.extend(ids)
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

    def similarity_search(self, query, k):
        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(self.documents, self.metadatas)
        ][:k]


def test_bm25_returns_lexical_match_first():
    index = BM25Index()
    index.build({
        "a": "alpha beta gamma",
        "b": "invoice payment refund policy",
        "c": "graph neural retrieval",
    })

    hits = index.search("refund invoice", k=2)

    assert hits[0].doc_id == "b"
    assert hits[0].score > 0


def test_rrf_combines_dense_and_sparse_rankings():
    fused = reciprocal_rank_fusion([
        [RankedItem("dense-only", 1.0), RankedItem("shared", 0.5)],
        [RankedItem("shared", 2.0), RankedItem("sparse-only", 1.0)],
    ])

    assert fused[0].doc_id == "shared"
    assert {item.doc_id for item in fused} == {"dense-only", "shared", "sparse-only"}


def test_noop_reranker_preserves_order_and_limit():
    docs = [Document(page_content=str(i)) for i in range(3)]

    assert NoOpReranker().rerank("query", docs, limit=2) == docs[:2]


def test_basic_eval_metrics():
    relevant = {"a", "c"}
    retrieved = ["b", "a", "c"]

    assert precision_at_k(relevant, retrieved, 2) == 0.5
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert hit_rate_at_k(relevant, retrieved, 2) == 1.0
    assert ndcg_at_k(relevant, retrieved, 3) > 0.0
    assert mean_reciprocal_rank(relevant, retrieved) == 0.5


def test_eval_runner_aggregates_rows():
    rows = [
        {"relevant_ids": ["a"], "retrieved_ids": ["b", "a"]},
        {"relevant_ids": ["c"], "retrieved_ids": ["c", "d"]},
    ]

    result = evaluate_rows(rows, k=2)

    assert result["queries"] == 2
    assert result["recall@2"] == 1.0
    assert result["mrr"] == 0.75


def test_hybrid_retriever_adds_stable_chunk_ids_and_retrieves(tmp_path):
    settings = Settings(dense_k=2, sparse_k=2, graph_k=2, fused_k=2, graph_path=tmp_path / "graph.json")
    vectorstore = FakeVectorStore()
    retriever = HybridRetriever(vectorstore, settings, reranker=NoOpReranker())
    docs = [
        Document(page_content="refund invoice policy", metadata={"source": "a.txt", "chunk_index": 0}),
        Document(page_content="graph retrieval notes", metadata={"source": "b.txt", "chunk_index": 0}),
    ]

    added, count = retriever.add_documents(docs, file_hash="hash-1")
    result = retriever.retrieve("refund invoice")

    assert added is True
    assert count == 2
    assert all(metadata.get("chunk_id") for metadata in vectorstore.metadatas)
    assert result.documents[0].metadata["source"] == "a.txt"
    assert result.sparse_count >= 1
    assert result.graph_count >= 1


def test_graph_index_extracts_entities_and_retrieves(tmp_path):
    graph = KnowledgeGraphIndex(tmp_path / "graph.json")
    docs = {
        "doc-a": Document(
            page_content="Qwen GraphRAG uses BM25, Chroma, and knowledge graph retrieval.",
            metadata={"source": "a.txt", "chunk_index": 0},
        ),
        "doc-b": Document(
            page_content="Seafood freshness classification uses CNN computer vision.",
            metadata={"source": "b.txt", "chunk_index": 0},
        ),
    }

    graph.build(docs)
    snapshot = graph.snapshot(limit=5)
    hits = graph.search("How does GraphRAG use Chroma and BM25?", k=2)

    assert "bm25" in extract_entities(docs["doc-a"].page_content)
    assert graph.stats()["entities"] > 0
    assert graph.stats()["relations"] > 0
    assert snapshot["nodes"]
    assert snapshot["edges"]
    assert hits[0].doc_id == "doc-a"


class FakeGraphExtractor:
    def extract(self, text: str) -> GraphDocument:
        return GraphDocument(
            entities=[
                GraphEntity(id="qwen", name="qwen", type="MODEL", confidence=0.9),
                GraphEntity(id="graphrag", name="graphrag", type="TECH", confidence=0.9),
            ],
            relations=[
                GraphRelation(
                    source_id="qwen",
                    target_id="graphrag",
                    relation="POWERS",
                    confidence=0.85,
                    evidence="Qwen powers GraphRAG extraction.",
                )
            ],
        )


def test_graph_index_supports_typed_entities_and_relations(tmp_path):
    graph = KnowledgeGraphIndex(tmp_path / "graph.json")
    graph.extractor = FakeGraphExtractor()

    graph.build({
        "doc-a": Document(
            page_content="Qwen powers GraphRAG extraction.",
            metadata={"source": "a.txt", "chunk_index": 0},
        )
    })
    snapshot = graph.snapshot(limit=5)

    assert snapshot["nodes"][0]["type"] in {"MODEL", "TECH"}
    assert snapshot["edges"][0]["relation"] == "POWERS"
    assert graph.stats()["relation_types"] == 1
    assert graph.search("Qwen GraphRAG", k=1)[0].doc_id == "doc-a"


def test_kuzu_graph_backend_persists_typed_graph(tmp_path):
    pytest.importorskip("kuzu")
    settings = Settings(
        graph_path=tmp_path / "graph.json",
        graph_db_path=tmp_path / "kuzu_graph",
        graph_backend="kuzu",
        graph_extraction_mode="rules",
    )
    graph = KnowledgeGraphIndex(settings=settings)
    graph.extractor = FakeGraphExtractor()

    graph.build({
        "doc-a": Document(
            page_content="Qwen powers GraphRAG extraction.",
            metadata={"source": "a.txt", "chunk_index": 0},
        )
    })

    assert graph.stats()["backend"] == "kuzu"
    assert (tmp_path / "kuzu_graph").exists()
    assert graph.snapshot()["edges"][0]["relation"] == "POWERS"


def test_llm_graph_extractor_accepts_common_spo_json_shape():
    extractor = LocalLLMGraphExtractor(Settings())
    response = """
    {
      "knowledge_graph": {
        "entities": ["Qwen", "GraphRAG", "Kuzu"],
        "relations": [
          {"subject": "Qwen", "predicate": "improves", "object": "GraphRAG"},
          {"subject": "Kuzu", "predicate": "stores", "object": "GraphRAG"}
        ]
      }
    }
    """

    graph_doc = extractor._parse_response(response)

    assert {entity.name for entity in graph_doc.entities} == {"qwen", "graphrag", "kuzu"}
    assert {relation.relation for relation in graph_doc.relations} == {"IMPROVES", "STORES"}


def test_monitor_records_runtime_metrics():
    monitor = MetricsMonitor()

    monitor.record_ingest("a.txt", chunks=2, latency_ms=10)
    monitor.record_query(
        "What is GraphRAG?",
        latency_ms=25,
        dense_count=3,
        sparse_count=2,
        graph_count=1,
        fused_count=4,
        returned_count=2,
    )

    snapshot = monitor.snapshot(graph_stats={"entities": 3})

    assert snapshot["ingest"]["count"] == 1
    assert snapshot["query"]["avg_graph_count"] == 1
    assert snapshot["graph"]["entities"] == 3
