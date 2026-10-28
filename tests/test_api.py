import asyncio
from io import BytesIO
from threading import Event

from fastapi import UploadFile
from fastapi.testclient import TestClient
import httpx
import pytest

from backend.config import Settings
from backend.ingestion import save_upload
from backend.main import create_app
from backend.service import RAGService
from test_index_recovery import MemoryStore


class Generator:
    def generate(self, instruction, query):
        assert "[S1]" in instruction
        return "The policy allows thirty days. [S1]"


def config(tmp_path, **kwargs):
    return Settings(graph_backend="json", graph_path=tmp_path / "graph.json",
                    upload_dir=tmp_path / "uploads", chroma_dir=tmp_path / "chroma",
                    offload_dir=tmp_path / "offload", **kwargs)


def test_ingestion_answer_references_limits_and_cleanup(tmp_path):
    settings = config(tmp_path, max_upload_bytes=256, max_query_chars=30)
    app = create_app(settings, lambda cfg: RAGService(cfg, MemoryStore(), Generator()))
    with TestClient(app) as client:
        assert client.post("/query/local", json={"query": "  "}).status_code == 422
        assert client.post("/query/local", json={"query": "x" * 31}).status_code == 422
        assert client.post("/ingest", files={"file": ("bad.exe", b"bad")}).status_code == 400
        assert client.post("/ingest", files={"file": ("large.txt", b"x" * 257)}).status_code == 413
        assert client.post("/ingest", files={"file": ("empty.txt", b"")}).status_code == 400
        upload = client.post("/ingest", files={"file": ("policy.txt", b"Returns are accepted for thirty days.")})
        assert upload.status_code == 200, upload.text
        answer = client.post("/query/local", json={"query": "What is the return policy?"})
        assert answer.status_code == 200, answer.text
        assert answer.json()["references"][0]["id"] == "S1"
        assert "thirty days" in answer.json()["references"][0]["content"]
        assert client.get("/health").status_code == 200
    assert not list(settings.upload_dir.iterdir())


def test_same_filename_uploads_have_distinct_paths(tmp_path):
    settings = config(tmp_path)
    async def upload_both():
        return await asyncio.gather(save_upload(UploadFile(BytesIO(b"one"), filename="same.txt"), settings),
                                    save_upload(UploadFile(BytesIO(b"two"), filename="same.txt"), settings))
    first, second = asyncio.run(upload_both())
    assert first != second
    assert {first.read_bytes(), second.read_bytes()} == {b"one", b"two"}


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_invalid_citations_get_one_repair_attempt_then_are_withheld(tmp_path, repair_succeeds):
    class UnreliableGenerator:
        calls = 0
        def generate(self, instruction, query):
            self.calls += 1
            return "Thirty days. [S1]" if repair_succeeds and self.calls == 2 else "Thirty days. [S99]"
    generator = UnreliableGenerator()
    app = create_app(config(tmp_path), lambda cfg: RAGService(cfg, MemoryStore(), generator))
    with TestClient(app) as client:
        assert client.post("/ingest", files={"file": ("a.txt", b"Thirty days.")}).status_code == 200
        result = client.post("/query/local", json={"query": "How long?"}).json()
        assert result["generation_attempts"] == 2
        assert generator.calls == 2
        assert result["citation_status"] == ("valid" if repair_succeeds else "invalid")
        assert "S99" not in result["answer"]


def test_declared_and_chunked_body_limits(tmp_path):
    settings = config(tmp_path, max_upload_bytes=16)
    app = create_app(settings, lambda cfg: RAGService(cfg, MemoryStore(), Generator()))
    with TestClient(app) as client:
        result = client.post("/query/local", content=b"x" * 65553)
        assert result.status_code == 413
    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                async def body():
                    yield b'{"query":"'
                    yield b"x" * 65553
                    yield b'"}'
                result = await client.post("/query/local", content=body(), headers={"Content-Type": "application/json"})
                assert result.status_code == 413, result.text
    asyncio.run(run())


def test_cancelled_client_does_not_free_active_worker_or_block_health(tmp_path):
    started, release = Event(), Event()
    class SlowService:
        def health(self): return {"status": "ok"}
        def metrics(self): return {}
        def close(self): pass
        def query(self, query):
            started.set()
            assert release.wait(5)
            return {"answer": query, "references": []}
    async def run():
        app = create_app(config(tmp_path, work_queue_size=1), lambda cfg: SlowService())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                active = asyncio.create_task(client.post("/query/local", json={"query": "first"}))
                assert await asyncio.to_thread(started.wait, 2)
                active.cancel()
                with pytest.raises(asyncio.CancelledError): await active
                pending = asyncio.create_task(client.post("/query/local", json={"query": "second"}))
                try:
                    for _ in range(100):
                        if app.state.work.queue.qsize() == 1: break
                        await asyncio.sleep(0)
                    assert app.state.work.queue.qsize() == 1
                    health = await asyncio.wait_for(client.get("/health"), 1)
                    assert health.json()["worker_active"]
                    rejected = await client.post("/query/local", json={"query": "third"})
                    assert rejected.status_code == 503
                    assert rejected.headers["retry-after"] == "1"
                finally:
                    release.set()
                assert (await pending).status_code == 200
    asyncio.run(run())
