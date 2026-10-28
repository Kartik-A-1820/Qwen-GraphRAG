import json
import math
import re
import shutil
import gc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from backend.config import Settings
from backend.retrieval.fusion import RankedItem


ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9+\-]*(?:\s+[A-Z][A-Za-z0-9+\-]*){0,3}|[A-Za-z]+(?:RAG|LLM|BM25|API|GPU|CPU|CUDA|FAISS|Chroma|GraphRAG)[A-Za-z0-9+\-]*)\b"
)
DOMAIN_TERM_PATTERN = re.compile(r"\b[a-z][a-z0-9\-]{4,}\b")
TECH_TERMS = {
    "api", "bm25", "chroma", "cuda", "dense retrieval", "embedding", "faiss",
    "gemma", "graph", "graphrag", "hybrid rag", "knowledge graph", "llm",
    "local model", "qwen", "reranker", "retrieval", "rrf", "sparse retrieval",
    "vector store",
}
STOP_ENTITIES = {
    "about", "after", "again", "along", "also", "because", "before", "between",
    "could", "from", "into", "other", "should", "their", "there", "these",
    "thing", "those", "through", "using", "where", "which", "while", "would",
    "the", "this", "that", "it", "and", "or", "a", "an",
}


@dataclass(frozen=True)
class GraphEntity:
    id: str
    name: str
    type: str = "TERM"
    confidence: float = 0.6


@dataclass(frozen=True)
class GraphRelation:
    source_id: str
    target_id: str
    relation: str
    confidence: float = 0.6
    evidence: str = ""


@dataclass(frozen=True)
class GraphDocument:
    entities: list[GraphEntity]
    relations: list[GraphRelation]


@dataclass(frozen=True)
class GraphHit:
    doc_id: str
    score: float
    matched_entities: list[str]


class GraphExtractor(Protocol):
    def extract(self, text: str) -> GraphDocument:
        ...


def normalize_entity(entity: str) -> str:
    cleaned = re.sub(r"\s+", " ", entity.strip().lower())
    return cleaned.strip(".,:;()[]{}")


def entity_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_entity(name)).strip("_")


def extract_entities(text: str, max_entities: int = 24) -> list[str]:
    entities = []
    seen = set()
    lowered = text.lower()

    for term in TECH_TERMS:
        if term in lowered and term not in seen:
            seen.add(term)
            entities.append(term)

    for match in ENTITY_PATTERN.findall(text):
        entity = normalize_entity(match)
        if len(entity) < 3 or entity in STOP_ENTITIES or entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)

    for match in DOMAIN_TERM_PATTERN.findall(lowered):
        entity = normalize_entity(match)
        if entity in STOP_ENTITIES or entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)

    return entities[:max_entities]


class RuleBasedGraphExtractor:
    def extract(self, text: str) -> GraphDocument:
        entities = [
            GraphEntity(id=entity_id(name), name=name, type=_guess_entity_type(name), confidence=0.55)
            for name in extract_entities(text)
        ]
        relations = []
        for index, left in enumerate(entities):
            for right in entities[index + 1:]:
                relations.append(
                    GraphRelation(
                        source_id=left.id,
                        target_id=right.id,
                        relation="CO_OCCURS_WITH",
                        confidence=0.4,
                        evidence=text[:220],
                    )
                )
        return GraphDocument(entities=entities, relations=relations)


class LocalLLMGraphExtractor:
    def __init__(self, settings: Settings, fallback: GraphExtractor | None = None) -> None:
        self.settings = settings
        self.fallback = fallback or RuleBasedGraphExtractor()
        self.generator = None
        self.llm_extractions = 0
        self.fallback_extractions = 0

    def extract(self, text: str) -> GraphDocument:
        if not text.strip():
            return GraphDocument(entities=[], relations=[])
        try:
            response = self._generator().generate(
                self._system_prompt(),
                text[: self.settings.graph_extraction_max_chunk_chars],
                max_new_tokens=self.settings.graph_extraction_max_new_tokens,
            )
            graph_doc = self._parse_response(response)
            self.llm_extractions += 1
            return graph_doc
        except Exception:
            self.fallback_extractions += 1
            return self.fallback.extract(text)

    def _generator(self):
        if self.generator is None:
            from dataclasses import replace

            from backend.llm import LocalCausalLMGenerator

            extractor_settings = replace(
                self.settings,
                model_id=self.settings.graph_extractor_model,
                max_new_tokens=self.settings.graph_extraction_max_new_tokens,
            )
            self.generator = LocalCausalLMGenerator(extractor_settings)
        return self.generator

    def _system_prompt(self) -> str:
        return (
            "Extract a compact knowledge graph from the provided document chunk. "
            "Return only valid JSON with this shape: "
            '{"entities":[{"name":"...","type":"PERSON|ORG|TECH|CONCEPT|PLACE|METRIC|OTHER","confidence":0.0}],'
            '"relations":[{"source":"entity name","target":"entity name","type":"USES|PART_OF|CAUSES|IMPROVES|COMPARES_WITH|MENTIONS|RELATED_TO","confidence":0.0,"evidence":"short quote"}]}. '
            "Use 3 to 12 high-signal entities and only relationships directly supported by the text."
        )

    def _parse_response(self, response: str) -> GraphDocument:
        start = response.find("{")
        end = response.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Extractor did not return JSON.")
        payload = json.loads(response[start:end + 1])
        if "knowledge_graph" in payload:
            payload = payload["knowledge_graph"]
        entities_by_name: dict[str, GraphEntity] = {}
        for item in payload.get("entities", []):
            if isinstance(item, str):
                item = {"name": item, "type": "OTHER", "confidence": 0.65}
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            graph_entity = GraphEntity(
                id=entity_id(name),
                name=normalize_entity(name),
                type=str(item.get("type", "OTHER")).upper()[:32],
                confidence=float(item.get("confidence", 0.7)),
            )
            entities_by_name[graph_entity.name] = graph_entity

        relations = []
        for item in payload.get("relations", []):
            source = normalize_entity(str(item.get("source") or item.get("subject") or ""))
            target = normalize_entity(str(item.get("target") or item.get("object") or ""))
            if source not in entities_by_name or target not in entities_by_name or source == target:
                continue
            relations.append(
                GraphRelation(
                    source_id=entities_by_name[source].id,
                    target_id=entities_by_name[target].id,
                    relation=str(item.get("type") or item.get("predicate") or "RELATED_TO").upper()[:48],
                    confidence=float(item.get("confidence", 0.7)),
                    evidence=str(item.get("evidence", ""))[:280],
                )
            )
        if not entities_by_name:
            raise ValueError("Extractor returned no entities.")
        return GraphDocument(entities=list(entities_by_name.values()), relations=relations)


def _guess_entity_type(name: str) -> str:
    if name in TECH_TERMS or any(token in name for token in ("rag", "llm", "retrieval", "model", "cuda")):
        return "TECH"
    return "TERM"


class KuzuGraphStore:
    def __init__(self, path: Path, buffer_pool_mb: int = 128) -> None:
        import kuzu

        if buffer_pool_mb <= 0:
            raise ValueError("Kuzu buffer pool must be positive.")
        self.path = path
        self.buffer_pool_size = buffer_pool_mb * 1024 * 1024
        self.db = kuzu.Database(str(path), buffer_pool_size=self.buffer_pool_size)
        self.conn = kuzu.Connection(self.db)
        self._create_schema()

    def reset(self) -> None:
        self.close()
        if self.path.exists():
            if self.path.is_dir():
                shutil.rmtree(self.path)
            else:
                self.path.unlink()
        import kuzu

        gc.collect()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(str(self.path), buffer_pool_size=self.buffer_pool_size)
        self.conn = kuzu.Connection(self.db)
        self._create_schema()

    def close(self) -> None:
        self.conn = None
        self.db = None

    def _create_schema(self) -> None:
        statements = [
            "CREATE NODE TABLE Entity(id STRING PRIMARY KEY, name STRING, type STRING, mentions INT64)",
            "CREATE NODE TABLE Chunk(id STRING PRIMARY KEY, source STRING, page STRING, chunk_index INT64, text STRING)",
            "CREATE REL TABLE MENTIONED_IN(FROM Entity TO Chunk, confidence DOUBLE)",
            "CREATE REL TABLE RELATED_TO(FROM Entity TO Entity, relation STRING, weight DOUBLE, evidence_chunk_id STRING, confidence DOUBLE)",
        ]
        for statement in statements:
            try:
                self.conn.execute(statement)
            except Exception:
                pass

    def add_chunk(self, doc_id: str, document: Document, graph_doc: GraphDocument) -> None:
        metadata = document.metadata or {}
        self.conn.execute(
            """
            MERGE (c:Chunk {id: $id})
            ON CREATE SET c.source = $source, c.page = $page, c.chunk_index = $chunk_index, c.text = $text
            ON MATCH SET c.source = $source, c.page = $page, c.chunk_index = $chunk_index, c.text = $text
            """,
            {
                "id": doc_id,
                "source": str(metadata.get("source", "")),
                "page": str(metadata.get("page", "")),
                "chunk_index": int(metadata.get("chunk_index") or 0),
                "text": document.page_content[:4000],
            },
        )
        for entity in graph_doc.entities:
            self.conn.execute(
                """
                MERGE (e:Entity {id: $id})
                ON CREATE SET e.name = $name, e.type = $type, e.mentions = 1
                ON MATCH SET e.mentions = e.mentions + 1
                """,
                {"id": entity.id, "name": entity.name, "type": entity.type},
            )
            self.conn.execute(
                """
                MATCH (e:Entity {id: $entity_id})
                MATCH (c:Chunk {id: $chunk_id})
                CREATE (e)-[:MENTIONED_IN {confidence: $confidence}]->(c)
                """,
                {"entity_id": entity.id, "chunk_id": doc_id, "confidence": float(entity.confidence)},
            )
        for relation in graph_doc.relations:
            self.conn.execute(
                """
                MATCH (s:Entity {id: $source_id})
                MATCH (t:Entity {id: $target_id})
                CREATE (s)-[:RELATED_TO {
                    relation: $relation,
                    weight: 1.0,
                    evidence_chunk_id: $chunk_id,
                    confidence: $confidence
                }]->(t)
                """,
                {
                    "source_id": relation.source_id,
                    "target_id": relation.target_id,
                    "relation": relation.relation,
                    "chunk_id": doc_id,
                    "confidence": float(relation.confidence),
                },
            )


class KnowledgeGraphIndex:
    def __init__(self, path: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings
        self.path = path or (settings.graph_path if settings else None)
        self.backend = (settings.graph_backend if settings else "json").lower()
        self.entity_to_docs: dict[str, set[str]] = defaultdict(set)
        self.doc_to_entities: dict[str, set[str]] = defaultdict(set)
        self.entity_names: dict[str, str] = {}
        self.entity_types: dict[str, str] = {}
        self.edges: Counter[tuple[str, str, str]] = Counter()
        self.doc_metadata: dict[str, dict] = {}
        self.store = self._build_store()
        self.extractor = self._build_extractor()
        self.load()

    def _build_store(self):
        if not self.settings or self.backend != "kuzu":
            return None
        try:
            return KuzuGraphStore(self.settings.graph_db_path, self.settings.graph_buffer_pool_mb)
        except Exception:
            return None

    def _build_extractor(self) -> GraphExtractor:
        if self.settings and self.settings.graph_extraction_mode.lower() in {"llm", "local-llm", "gemma", "qwen"}:
            return LocalLLMGraphExtractor(self.settings)
        return RuleBasedGraphExtractor()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.entity_to_docs = defaultdict(set, {
            entity: set(doc_ids) for entity, doc_ids in payload.get("entity_to_docs", {}).items()
        })
        self.doc_to_entities = defaultdict(set, {
            doc_id: set(entities) for doc_id, entities in payload.get("doc_to_entities", {}).items()
        })
        self.entity_names = payload.get("entity_names", {})
        self.entity_types = payload.get("entity_types", {})
        self.edges = Counter({
            tuple(edge.split("||", 2)): weight for edge, weight in payload.get("edges", {}).items()
        })
        self.doc_metadata = payload.get("doc_metadata", {})

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": "kuzu" if self.store else "json",
            "entity_to_docs": {
                entity: sorted(doc_ids) for entity, doc_ids in sorted(self.entity_to_docs.items())
            },
            "doc_to_entities": {
                doc_id: sorted(entities) for doc_id, entities in sorted(self.doc_to_entities.items())
            },
            "entity_names": self.entity_names,
            "entity_types": self.entity_types,
            "edges": {
                "||".join(edge): weight for edge, weight in sorted(self.edges.items())
            },
            "doc_metadata": self.doc_metadata,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def build(self, documents: dict[str, Document]) -> None:
        self.entity_to_docs = defaultdict(set)
        self.doc_to_entities = defaultdict(set)
        self.entity_names = {}
        self.entity_types = {}
        self.edges = Counter()
        self.doc_metadata = {}
        if self.store:
            self.store.reset()

        for doc_id, document in documents.items():
            graph_doc = self.extractor.extract(document.page_content)
            entities = graph_doc.entities
            document.metadata["graph_entities"] = [entity.name for entity in entities]
            self.doc_metadata[doc_id] = {
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "chunk_index": document.metadata.get("chunk_index"),
            }
            for entity in entities:
                self.entity_to_docs[entity.id].add(doc_id)
                self.doc_to_entities[doc_id].add(entity.id)
                self.entity_names[entity.id] = entity.name
                self.entity_types[entity.id] = entity.type
            for relation in graph_doc.relations:
                edge = (relation.source_id, relation.target_id, relation.relation)
                self.edges[edge] += 1
            if self.store:
                self.store.add_chunk(doc_id, document, graph_doc)
        self.save()

    def add_documents(self, documents: dict[str, Document]) -> None:
        merged = {
            doc_id: Document(page_content="", metadata={})
            for doc_id in self.doc_to_entities
        }
        merged.update(documents)
        self.build(merged)

    def search(self, query: str, k: int = 5) -> list[GraphHit]:
        query_entities = [entity_id(name) for name in extract_entities(query, max_entities=12)]
        query_entities = [item for item in query_entities if item]
        if not query_entities:
            return []

        scores: defaultdict[str, float] = defaultdict(float)
        matched: defaultdict[str, set[str]] = defaultdict(set)
        for entity in query_entities:
            direct_docs = self.entity_to_docs.get(entity, set())
            for doc_id in direct_docs:
                scores[doc_id] += 2.0
                matched[doc_id].add(self.entity_names.get(entity, entity))

            for source_id, target_id, relation in self.edges:
                if entity not in {source_id, target_id}:
                    continue
                neighbor = target_id if source_id == entity else source_id
                weight = self.edges[(source_id, target_id, relation)]
                for doc_id in self.entity_to_docs.get(neighbor, set()):
                    scores[doc_id] += 0.35 * math.log1p(weight)
                    matched[doc_id].add(self.entity_names.get(neighbor, neighbor))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [
            GraphHit(doc_id=doc_id, score=score, matched_entities=sorted(matched[doc_id]))
            for doc_id, score in ranked
        ]

    def ranked_items(self, query: str, k: int) -> list[RankedItem]:
        return [RankedItem(doc_id=hit.doc_id, score=hit.score) for hit in self.search(query, k)]

    def stats(self) -> dict:
        return {
            "backend": "kuzu" if self.store else "json",
            "extraction_mode": self.settings.graph_extraction_mode if self.settings else "rules",
            "llm_extractions": getattr(self.extractor, "llm_extractions", 0),
            "fallback_extractions": getattr(self.extractor, "fallback_extractions", 0),
            "entities": len(self.entity_to_docs),
            "relations": sum(self.edges.values()),
            "relation_types": len({relation for _source, _target, relation in self.edges}),
            "documents": len(self.doc_to_entities),
        }

    def snapshot(self, limit: int = 40) -> dict:
        entity_weights = Counter({
            entity: len(doc_ids) for entity, doc_ids in self.entity_to_docs.items()
        })
        top_entities = entity_weights.most_common(limit)
        allowed = {entity for entity, _weight in top_entities}
        edges = [
            {
                "source": self.entity_names.get(source_id, source_id),
                "target": self.entity_names.get(target_id, target_id),
                "relation": relation,
                "weight": weight,
            }
            for (source_id, target_id, relation), weight in self.edges.most_common(limit * 2)
            if source_id in allowed and target_id in allowed
        ]
        return {
            "stats": self.stats(),
            "nodes": [
                {
                    "id": entity,
                    "label": self.entity_names.get(entity, entity),
                    "type": self.entity_types.get(entity, "TERM"),
                    "weight": weight,
                }
                for entity, weight in top_entities
            ],
            "edges": edges[:limit],
        }
