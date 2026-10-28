import os
import time
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

from backend.config import settings
from backend.evaluation.monitoring import MetricsMonitor
from backend.ingestion import load_and_split, save_upload
from backend.llm import LocalCausalLMGenerator
from backend.retrieval.hybrid import HybridRetriever


os.environ["PYTORCH_SDP_ATTENTION"] = "0"
settings.ensure_dirs()

app = FastAPI(
    title="Qwen-GraphRAG API",
    description="Local document Q&A with Qwen, dense/BM25/graph retrieval, rank fusion, and optional reranking.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


embedding_function = HuggingFaceEmbeddings(model_name=settings.embedding_model)
vectorstore = Chroma(
    collection_name=settings.collection_name,
    embedding_function=embedding_function,
    persist_directory=str(settings.chroma_dir),
)
retriever = HybridRetriever(vectorstore=vectorstore, settings=settings)
generator = LocalCausalLMGenerator(settings=settings)
monitor = MetricsMonitor()


def build_reference_text(documents) -> str:
    remaining_chars = settings.max_context_chars
    blocks = []
    for index, doc in enumerate(documents, start=1):
        content = doc.page_content[:remaining_chars]
        if not content:
            break
        remaining_chars -= len(content)
        blocks.append(
            "Source: {source}, Page: {page}, Chunk: {chunk}\nContent: {content}".format(
                source=doc.metadata.get("source", f"Document {index}"),
                page=doc.metadata.get("page", "N/A"),
                chunk=doc.metadata.get("chunk_index", "N/A"),
                content=content,
            )
        )
    return "\n\n".join(blocks)


def build_references(documents) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", f"Doc {index}"),
            "page": doc.metadata.get("page", "N/A"),
            "chunk": doc.metadata.get("chunk_index", "N/A"),
            "content": doc.page_content,
        }
        for index, doc in enumerate(documents, start=1)
    ]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "runtime": "local-only",
        "model_id": settings.model_id,
        "embedding_model": settings.embedding_model,
        "load_in_4bit": settings.load_in_4bit,
        "reranker_enabled": settings.enable_reranker,
        "graph_backend": settings.graph_backend,
        "graph_extraction_mode": settings.graph_extraction_mode,
        "graph_extractor_model": settings.graph_extractor_model,
        "graph": retriever.graph.stats(),
    }


@app.get("/metrics")
async def metrics():
    return monitor.snapshot(graph_stats=retriever.graph.stats())


@app.post("/metrics/reset")
async def reset_metrics():
    monitor.reset()
    return {"status": "reset"}


@app.get("/graph")
async def graph(limit: int = 40):
    return retriever.graph.snapshot(limit=max(5, min(limit, 120)))


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    started = time.perf_counter()
    file_path = await save_upload(file, settings)
    try:
        file_hash, chunks = load_and_split(file_path, file.filename or file_path.name, settings)
        added, count = retriever.add_documents(chunks, file_hash)
        if not added:
            monitor.record_ingest(file.filename or file_path.name, 0, (time.perf_counter() - started) * 1000, True)
            return {"message": "Document already ingested.", "status": "already_ingested"}
        monitor.record_ingest(file.filename or file_path.name, count, (time.perf_counter() - started) * 1000)
        return {
            "message": "File ingested successfully.",
            "num_documents": count,
            "retrieval": "dense+bm25+graph+rrf",
            "graph": retriever.graph.stats(),
        }
    except Exception as exc:
        monitor.record_error("/ingest", repr(exc))
        raise
    finally:
        file_path.unlink(missing_ok=True)


@app.post("/query/local")
async def query_local(data: QueryRequest):
    started = time.perf_counter()
    try:
        result = retriever.retrieve(data.query)
        if not result.documents:
            return {"answer": "Not found relevant answer.", "references": []}

        reference_text = build_reference_text(result.documents)
        system_instruction = (
            "You are a local, grounded document assistant. Use ONLY the provided references "
            "to answer the user's question. If the answer is not present or cannot be determined "
            "from the references, reply with: 'Not found in the provided reference.'\n\n"
            f"References:\n{reference_text}"
        )

        answer_text = generator.generate(system_instruction, data.query)
        monitor.record_query(
            data.query,
            (time.perf_counter() - started) * 1000,
            result.dense_count,
            result.sparse_count,
            result.graph_count,
            result.fused_count,
            len(result.documents),
        )
        return {
            "answer": answer_text,
            "references": build_references(result.documents),
            "retrieval_debug": {
                "dense_count": result.dense_count,
                "sparse_count": result.sparse_count,
                "graph_count": result.graph_count,
                "fused_count": result.fused_count,
                "graph": result.graph_stats,
            },
        }
    except Exception as exc:
        monitor.record_error("/query/local", repr(exc))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred during processing.")


@app.post("/summarize_pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    file_path = await save_upload(file, settings)
    try:
        _file_hash, chunks = load_and_split(file_path, file.filename or file_path.name, settings)
        summaries = []
        for index in range(0, len(chunks), 3):
            batch_text = "\n\n".join(chunk.page_content for chunk in chunks[index:index + 3])
            system_instruction = (
                "You are a local text summarization engine. Provide only a concise summary "
                "of the provided text. Do not include greetings or meta commentary."
            )
            summaries.append(
                generator.generate(system_instruction, batch_text, max_new_tokens=settings.max_new_tokens)
            )

        if not summaries:
            return {"summary": "Failed to read document content."}

        if len(summaries) == 1:
            return {"summary": summaries[0]}

        final_instruction = (
            "Combine these partial summaries into a single concise final summary. "
            "Return only the final summary text."
        )
        return {
            "summary": generator.generate(
                final_instruction,
                "\n\n".join(summaries),
                max_new_tokens=settings.max_new_tokens,
            )
        }
    finally:
        file_path.unlink(missing_ok=True)
