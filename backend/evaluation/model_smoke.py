"""Exercise actual ingestion, Chroma retrieval, pinned Qwen generation, and restart."""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import resource
import sys
import tempfile
import time

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app


ROOT = Path(__file__).resolve().parents[2]


def run(work):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(42)
    models = json.loads((ROOT / "evaluation/models.json").read_text())
    settings = Settings(
        model_id=models["generation"]["id"], model_revision=models["generation"]["revision"],
        embedding_model=models["embedding"]["id"], embedding_revision=models["embedding"]["revision"],
        device="cpu", load_in_4bit=False, max_new_tokens=64, graph_backend="json",
        graph_extraction_mode="rules", chroma_dir=work / "chroma", upload_dir=work / "uploads",
        offload_dir=work / "offload", graph_path=work / "graph.json", catalog_path=work / "catalog.sqlite3",
    )
    fixture = b"The Redwood release number is 7319. Its maintainer team is Lumen. This is a fictional integration fixture."
    app = create_app(settings)
    started = time.perf_counter()
    with TestClient(app) as client:
        upload = client.post("/ingest", files={"file": ("redwood.txt", fixture)})
        assert upload.status_code == 200, upload.text
        assert upload.json()["status"] == "ingested", "Use an empty work directory for this smoke run."
        response = client.post("/query/local", json={"query": "What is the Redwood release number?"})
        assert response.status_code == 200, response.text
        result = response.json()
        assert "7319" in result["answer"], result
        labels = set(re.findall(r"\[(S\d+)\]", result["answer"]))
        assert labels and labels <= {ref["id"] for ref in result["references"]}, result
        assert result["references"][0]["source"] == "redwood.txt"
        assert "7319" in result["references"][0]["content"]
        unknown = client.post("/query/local", json={"query": "What is Redwood's office street address?"})
        assert unknown.status_code == 200, unknown.text
        assert "not found" in unknown.json()["answer"].lower(), unknown.text
        resolved = getattr(app.state.runtime.generator.model.config, "_commit_hash", None)
        assert resolved == models["generation"]["revision"], resolved
    gc.collect()
    with TestClient(create_app(settings)) as restarted:
        duplicate = restarted.post("/ingest", files={"file": ("redwood.txt", fixture)})
        assert duplicate.status_code == 200, duplicate.text
        assert duplicate.json()["status"] == "already_ingested"
        assert restarted.get("/health").json()["graph"]["documents"] == 1
    assert not list(settings.upload_dir.iterdir())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "scope": "One fictional document and two questions; functional smoke, not a model-quality benchmark.",
        "models": models, "resolved_generation_revision": resolved,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "sentence-transformers", "chromadb"]},
        "checks": {"real_ingest": True, "real_retrieval": True, "real_qwen_generation": True,
                   "fact_present": True, "supported_citation_label": True,
                   "single_unknown_abstention": True, "restart_dedup": True, "upload_cleanup": True},
        "answer": result, "unknown_answer": unknown.json(),
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_mb": peak / (1024*1024 if sys.platform == "darwin" else 1024),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/model_smoke.json")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="qwen-model-smoke-") as directory:
        report = run(Path(directory))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
