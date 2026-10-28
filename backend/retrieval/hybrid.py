from dataclasses import asdict, dataclass
import hashlib
import json
from threading import RLock

from langchain_core.documents import Document

from backend.config import Settings
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.catalog import IngestCatalog, chunk_id
from backend.retrieval.fusion import RankedItem, reciprocal_rank_fusion
from backend.retrieval.graph import GraphDocument, GraphEntity, GraphRelation, KnowledgeGraphIndex
from backend.retrieval.rerankers import BaseReranker, CrossEncoderReranker, NoOpReranker


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[Document]
    dense_count: int
    sparse_count: int
    graph_count: int
    fused_count: int
    graph_stats: dict


class HybridRetriever:
    """One-process catalog owner; readers/writers share the same lock."""

    def __init__(self, vectorstore, settings: Settings, reranker: BaseReranker | None = None) -> None:
        self.vectorstore = vectorstore
        self.settings = settings
        self.lock = RLock()
        self.reranker = reranker or self._build_reranker(settings)
        self.bm25 = BM25Index()
        self.graph = KnowledgeGraphIndex(settings=settings)
        self.catalog = IngestCatalog(settings.catalog_path or settings.graph_path.with_suffix(".catalog.sqlite3"))
        self.catalog.bind_index_profile(json.dumps({
            "embedding": settings.embedding_model, "revision": settings.embedding_revision,
            "chunk_size": settings.chunk_size, "overlap": settings.chunk_overlap,
            "collection": settings.collection_name,
        }, sort_keys=True))
        self.documents_by_id = {}
        self._broken = True
        if self.catalog.empty():
            raw = vectorstore.get(include=["documents", "metadatas"])
            groups = {}
            for key, text, metadata in zip(raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])):
                if not text:
                    continue
                metadata = dict(metadata or {})
                metadata["chunk_id"] = key
                file_hash = metadata.get("hash") or hashlib.sha256(("legacy:" + key).encode()).hexdigest()
                metadata["hash"] = file_hash
                groups.setdefault(file_hash, {})[key] = Document(page_content=text, metadata=metadata)
            for file_hash, documents in groups.items():
                self.catalog.prepare(file_hash, documents)
        self.rebuild_sparse_index()

    def _build_reranker(self, settings):
        if settings.enable_reranker and settings.reranker_model:
            return CrossEncoderReranker(settings.reranker_model, settings.reranker_revision)
        return NoOpReranker()

    def _profile(self):
        return json.dumps({
            "schema": 2, "mode": self.settings.graph_extraction_mode,
            "model": self.settings.graph_extractor_model,
            "revision": self.settings.graph_extractor_revision or (
                self.settings.model_revision if self.settings.graph_extractor_model == self.settings.model_id else None
            ),
            "chars": self.settings.graph_extraction_max_chunk_chars,
            "tokens": self.settings.graph_extraction_max_new_tokens,
            "extractor": type(self.graph.extractor).__qualname__,
        }, sort_keys=True)

    def _graphs(self, documents):
        graphs = {}
        profile = self._profile()
        for key, document in documents.items():
            digest = hashlib.sha256((profile + "\n" + document.page_content).encode()).hexdigest()
            cached = self.catalog.cached_graph(digest)
            if cached is None:
                cached = asdict(self.graph.extractor.extract(document.page_content))
                self.catalog.cache_graph(digest, cached)
            graphs[key] = GraphDocument(
                entities=[GraphEntity(**entity) for entity in cached["entities"]],
                relations=[GraphRelation(**relation) for relation in cached["relations"]],
            )
        return graphs

    def _upsert(self, documents):
        if documents:
            self.vectorstore.add_texts(
                texts=[doc.page_content for doc in documents.values()],
                metadatas=[doc.metadata for doc in documents.values()], ids=list(documents),
            )

    def rebuild_sparse_index(self):
        """Restore projections from the ledger, reusing cached graph extraction."""
        with self.lock:
            self._broken = True
            documents = self.catalog.documents()
            graphs = self._graphs(documents)
            dirty = self.catalog.dirty() or self.catalog.profile() != self._profile()
            if dirty:
                self._upsert(self.catalog.documents(pending_only=True))
            self.bm25.build({key: doc.page_content for key, doc in documents.items()})
            if dirty or set(self.graph.doc_metadata) != set(documents):
                self.graph.build(documents, extracted=graphs)
            self.documents_by_id = documents
            self.catalog.finish(self._profile())
            self._broken = False

    def _ensure_recovered(self):
        if self._broken or self.catalog.dirty():
            self.rebuild_sparse_index()

    def add_documents(self, documents: list[Document], file_hash: str) -> tuple[bool, int]:
        with self.lock:
            self._ensure_recovered()
            if self.catalog.ready(file_hash):
                return False, 0
            prepared = {}
            for index, doc in enumerate(documents):
                key = chunk_id(file_hash, index, doc.page_content)
                prepared[key] = Document(page_content=doc.page_content, metadata={
                    **doc.metadata, "hash": file_hash, "chunk_id": key,
                })
            if not prepared:
                raise ValueError("No chunks to ingest.")
            self.catalog.prepare(file_hash, prepared)
            try:
                graphs = self._graphs(prepared)
                self._upsert(prepared)
                self.bm25.add({key: doc.page_content for key, doc in prepared.items()})
                self.graph.add_documents(prepared, extracted=graphs)
                self.documents_by_id.update(prepared)
                self.catalog.finish(self._profile())
            except BaseException:
                self._broken = True
                raise
            return True, len(prepared)

    def retrieve(self, query: str, mode: str = "reranked") -> RetrievalResult:
        with self.lock:
            self._ensure_recovered()
            if mode not in {"dense", "bm25", "hybrid", "graph", "reranked"}:
                raise ValueError("Unknown retrieval mode.")
            dense_docs = self.vectorstore.similarity_search(query, k=self.settings.dense_k) if mode != "bm25" else []
            dense_ranked = [RankedItem(self._doc_key(doc), 1.0 / rank) for rank, doc in enumerate(dense_docs, 1)]
            sparse_hits = self.bm25.search(query, k=self.settings.sparse_k) if mode != "dense" else []
            sparse_ranked = [RankedItem(hit.doc_id, hit.score) for hit in sparse_hits]
            graph_hits = self.graph.search(query, k=self.settings.graph_k) if mode in {"graph", "reranked"} else []
            rankings = [dense_ranked, sparse_ranked, [RankedItem(hit.doc_id, hit.score) for hit in graph_hits]]
            fused = reciprocal_rank_fusion(rankings, k=self.settings.rrf_k)
            candidates = [self.documents_by_id[item.doc_id] for item in fused if item.doc_id in self.documents_by_id]
            reranker = self.reranker if mode == "reranked" else NoOpReranker()
            reranked = reranker.rerank(query, candidates, self.settings.fused_k)
            return RetrievalResult(reranked, len(dense_docs), len(sparse_hits), len(graph_hits), len(candidates), self.graph.stats())

    def _doc_key(self, document):
        metadata = document.metadata or {}
        return metadata.get("chunk_id") or f"{metadata.get('source', 'unknown')}:{metadata.get('page', 'na')}:{metadata.get('chunk_index', 'na')}"
