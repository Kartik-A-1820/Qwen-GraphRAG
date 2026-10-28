"""FastAPI application factory; importing it never downloads model weights."""
import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.responses import JSONResponse

from backend.config import settings as default_settings
from backend.ingestion import safe_filename, save_upload
from backend.work_queue import QueueBusy, WorkQueue


logger = logging.getLogger(__name__)


class BodyLimit:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length."}, 400)(scope, receive, send)
        if length < 0:
            return await JSONResponse({"detail": "Invalid Content-Length."}, 400)(scope, receive, send)
        if length > self.limit:
            return await JSONResponse({"detail": "Request body exceeds the configured limit."}, 413)(scope, receive, send)
        received = 0
        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise HTTPException(413, "Request body exceeds the configured limit.")
            return message
        await self.app(scope, limited_receive, send)


def default_factory(settings):
    from backend.service import RAGService
    return RAGService(settings)


def create_app(settings=None, runtime_factory=None):
    settings = settings or default_settings
    runtime_factory = runtime_factory or default_factory

    @asynccontextmanager
    async def lifespan(app):
        runtime = await asyncio.to_thread(runtime_factory, settings)
        app.state.runtime = runtime
        app.state.work = WorkQueue(settings.work_queue_size)
        try:
            yield
        finally:
            await app.state.work.close()
            await asyncio.to_thread(runtime.close)

    app = FastAPI(title="Qwen-GraphRAG API",
                  description="Local Qwen document Q&A with hybrid retrieval and bounded background work.",
                  lifespan=lifespan)
    app.add_middleware(BodyLimit, limit=settings.max_upload_bytes + 65536)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    class QueryRequest(BaseModel):
        query: str = Field(min_length=1, max_length=settings.max_query_chars)

        @field_validator("query")
        @classmethod
        def nonblank(cls, value):
            value = value.strip()
            if not value:
                raise ValueError("Query must contain non-whitespace characters.")
            return value

    async def execute(operation, cleanup=None):
        try:
            future = app.state.work.submit(operation)
        except QueueBusy:
            if cleanup:
                cleanup()
            raise HTTPException(503, "Worker queue is full; retry later.", headers={"Retry-After": "1"}) from None
        try:
            return await asyncio.shield(future)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception:
            logger.exception("Application operation failed")
            raise HTTPException(500, "Application operation failed; inspect server logs.") from None

    async def upload(file, operation):
        try:
            path = await save_upload(file, settings)
        finally:
            await file.close()
        name = safe_filename(file.filename or "upload")
        def job():
            try:
                return operation(path, name)
            finally:
                path.unlink(missing_ok=True)
        return await execute(job, cleanup=lambda: path.unlink(missing_ok=True))

    @app.get("/health")
    async def health():
        return {**app.state.runtime.health(), "worker_active": app.state.work.active,
                "queued_jobs": app.state.work.queue.qsize()}

    @app.get("/metrics")
    async def metrics():
        return app.state.runtime.metrics()

    @app.post("/metrics/reset")
    async def reset_metrics():
        return await execute(app.state.runtime.reset_metrics)

    @app.get("/graph")
    async def graph(limit: int = 40):
        return await execute(lambda: app.state.runtime.graph(max(5, min(limit, 120))))

    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...)):
        return await upload(file, app.state.runtime.ingest)

    @app.post("/query/local")
    async def query(data: QueryRequest):
        return await execute(lambda: app.state.runtime.query(data.query))

    @app.post("/summarize_pdf")
    async def summarize(file: UploadFile = File(...)):
        return await upload(file, app.state.runtime.summarize)

    return app


app = create_app()
