import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class BM25Hit:
    doc_id: str
    score: float


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lengths: dict[str, int] = {}
        self.term_freqs: dict[str, Counter[str]] = {}
        self.doc_freqs: defaultdict[str, int] = defaultdict(int)
        self.avg_doc_length = 0.0
        self.total_doc_length = 0

    def build(self, documents: dict[str, str]) -> None:
        self.doc_ids = []
        self.doc_lengths = {}
        self.term_freqs = {}
        self.doc_freqs = defaultdict(int)

        self.total_doc_length = 0
        self.add(documents)

    def add(self, documents: dict[str, str]) -> None:
        for doc_id, text in documents.items():
            if doc_id in self.term_freqs:
                self.total_doc_length -= self.doc_lengths[doc_id]
                for term in self.term_freqs[doc_id]:
                    self.doc_freqs[term] -= 1
            else:
                self.doc_ids.append(doc_id)
            terms = tokenize(text)
            counts = Counter(terms)
            self.term_freqs[doc_id] = counts
            self.doc_lengths[doc_id] = len(terms)
            self.total_doc_length += len(terms)
            for term in counts:
                self.doc_freqs[term] += 1

        self.avg_doc_length = self.total_doc_length / len(self.doc_ids) if self.doc_ids else 0.0

    def search(self, query: str, k: int = 5) -> list[BM25Hit]:
        if not self.doc_ids:
            return []

        query_terms = tokenize(query)
        scores: dict[str, float] = defaultdict(float)
        corpus_size = len(self.doc_ids)

        for term in query_terms:
            doc_freq = self.doc_freqs.get(term, 0)
            if doc_freq == 0:
                continue
            idf = math.log(1 + (corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))
            for doc_id in self.doc_ids:
                term_freq = self.term_freqs[doc_id].get(term, 0)
                if term_freq == 0:
                    continue
                doc_length = self.doc_lengths[doc_id]
                norm = term_freq + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
                )
                scores[doc_id] += idf * (term_freq * (self.k1 + 1)) / norm

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [BM25Hit(doc_id=doc_id, score=score) for doc_id, score in ranked[:k]]
