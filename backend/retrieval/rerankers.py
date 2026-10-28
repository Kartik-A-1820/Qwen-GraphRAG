from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[Document], limit: int) -> list[Document]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    def rerank(self, query: str, documents: list[Document], limit: int) -> list[Document]:
        return documents[:limit]


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str, revision: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512, revision=revision, device="cpu")

    def rerank(self, query: str, documents: list[Document], limit: int) -> list[Document]:
        if not documents:
            return []
        scores = self.model.predict([(query, document.page_content) for document in documents])
        ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)
        return [document for document, _score in ranked[:limit]]
