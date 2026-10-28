"""Reproduce five retrieval arms using pinned real embedding/reranking models."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

from langchain_core.documents import Document

from backend.config import Settings
from backend.evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from backend.evaluation.memory import MemorySampler
from backend.retrieval.hybrid import HybridRetriever


ROOT = Path(__file__).resolve().parents[2]
ARMS = ["dense", "bm25", "hybrid", "graph", "reranked"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dataset():
    corpus = read_jsonl(ROOT / "evaluation/corpus.jsonl")
    queries = read_jsonl(ROOT / "evaluation/queries.jsonl")
    ids = {row["id"] for row in corpus}
    if len(ids) != len(corpus) or len({row["id"] for row in queries}) != len(queries):
        raise ValueError("Duplicate corpus/query IDs.")
    for query in queries:
        if not set(query["relevant_ids"]) <= ids:
            raise ValueError("Unknown relevance ID.")
        if (query["category"] == "unanswerable") != (not query["relevant_ids"]):
            raise ValueError("Unanswerable labels and relevance sets disagree.")
    return corpus, queries


def configuration(work, arm, k):
    models = json.loads((ROOT / "evaluation/models.json").read_text())
    return Settings(
        chroma_dir=work / "chroma", graph_path=work / "graph.json",
        graph_db_path=work / "kuzu", graph_backend="json", graph_extraction_mode="rules",
        catalog_path=work / "catalog.sqlite3", fused_k=k,
        embedding_model=models["embedding"]["id"], embedding_revision=models["embedding"]["revision"],
        enable_reranker=arm == "reranked", reranker_model=models["reranker"]["id"],
        reranker_revision=models["reranker"]["revision"],
    )


def retriever_for(work, arm, k):
    print(f"Preparing {arm} retrieval", file=sys.stderr, flush=True)
    from langchain_chroma import Chroma
    cfg = configuration(work, arm, k)
    embedding = None
    if arm != "bm25":
        print("Loading pinned CPU embedding model", file=sys.stderr, flush=True)
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding = HuggingFaceEmbeddings(model_name=cfg.embedding_model,
            model_kwargs={"device": "cpu", "revision": cfg.embedding_revision})
    store = Chroma(collection_name=cfg.collection_name, persist_directory=str(cfg.chroma_dir),
                   embedding_function=embedding)
    print("Opening durable retrieval indexes", file=sys.stderr, flush=True)
    return HybridRetriever(store, cfg)


def prepare(work, k):
    corpus, _ = dataset()
    retriever = retriever_for(work, "dense", k)
    for row in corpus:
        text = row["title"] + "\n" + row["text"]
        digest = hashlib.sha256((row["id"] + text).encode()).hexdigest()
        retriever.add_documents([Document(page_content=text, metadata={
            "source": row["source"], "evaluation_id": row["id"], "chunk_index": 0,
        })], digest)
    expected = {row["id"] for row in corpus}
    actual = {doc.metadata["evaluation_id"] for doc in retriever.documents_by_id.values()}
    if actual != expected:
        raise ValueError("Work directory contains a different corpus; choose a fresh directory.")


def aggregate(rows, k):
    answerable = [row for row in rows if row["relevant_ids"]]
    unknown = [row for row in rows if not row["relevant_ids"]]
    def mean(metric):
        return statistics.mean(metric(set(row["relevant_ids"]), row["retrieved_ids"]) for row in answerable)
    latencies = sorted(value for row in rows for value in row["latencies_ms"])
    return {
        "answerable_queries": len(answerable), "unanswerable_queries": len(unknown),
        f"recall@{k}": mean(lambda rel, got: recall_at_k(rel, got, k)),
        f"ndcg@{k}": mean(lambda rel, got: ndcg_at_k(rel, got, k)),
        "mrr": mean(mean_reciprocal_rank),
        "unanswerable_candidate_return_rate": statistics.mean(bool(row["retrieved_ids"]) for row in unknown) if unknown else None,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": latencies[min(len(latencies)-1, int(.95 * len(latencies)))],
    }


def run_arm(work, arm, k, repeats):
    start = time.perf_counter()
    retriever = retriever_for(work, arm, k)
    setup = time.perf_counter() - start
    _, queries = dataset()
    retriever.retrieve("A setup warmup query", mode=arm)
    rows = {q["id"]: {**q, "retrieved_ids": None, "latencies_ms": []} for q in queries}
    for repeat in range(repeats):
        order = list(queries)
        random.Random(42 + repeat).shuffle(order)
        for query in order:
            started = time.perf_counter()
            result = retriever.retrieve(query["query"], mode=arm)
            elapsed = (time.perf_counter()-started)*1000
            ids = [doc.metadata["evaluation_id"] for doc in result.documents]
            row = rows[query["id"]]
            if row["retrieved_ids"] is not None and row["retrieved_ids"] != ids:
                raise ValueError("Ranking changed between repeats; inspect nondeterminism.")
            row["retrieved_ids"] = ids
            row["latencies_ms"].append(elapsed)
    values = list(rows.values())
    groups = {category: aggregate([r for r in values if r["category"] == category], k)
              for category in {r["category"] for r in values} if category != "unanswerable"}
    return {"arm": arm, "metrics": aggregate(values, k), "groups": groups, "setup_seconds": setup,
            "rows": values}


def provenance():
    versions = {}
    for name in ["torch", "transformers", "sentence-transformers", "chromadb", "langchain-core", "langchain-chroma"]:
        versions[name] = importlib.metadata.version(name)
    import os
    source_paths = sorted((ROOT / "backend").rglob("*.py"))
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "thread_environment": {key: os.getenv(key) for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
            "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            "models": json.loads((ROOT / "evaluation/models.json").read_text()),
            "dataset_sha256": {p: hashlib.sha256((ROOT / "evaluation" / p).read_bytes()).hexdigest()
                               for p in ["corpus.jsonl", "queries.jsonl"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".local/evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/summary.json")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.k <= 0 or args.repeats <= 0:
        parser.error("k and repeats must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.arm:
        with MemorySampler() as memory:
            result = run_arm(args.work_dir, args.arm, args.k, args.repeats)
        result["sampled_peak_rss_mb"] = memory.peak_mb
        print("BENCHMARK_JSON:" + json.dumps(result))
        return
    prepare(args.work_dir, args.k)
    arms = []
    for arm in ARMS:
        print(f"Benchmarking {arm}", file=sys.stderr, flush=True)
        completed = subprocess.run([sys.executable, "-m", "backend.evaluation.benchmark",
            "--work-dir", str(args.work_dir), "--arm", arm, "--k", str(args.k), "--repeats", str(args.repeats)],
            cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(arm + " failed:\n" + completed.stderr[-6000:])
        records = [line.removeprefix("BENCHMARK_JSON:") for line in completed.stdout.splitlines() if line.startswith("BENCHMARK_JSON:")]
        if len(records) != 1:
            raise RuntimeError("Missing benchmark result for " + arm)
        arms.append(json.loads(records[0]))
    report = {"scope": "Authored fictional handbook; retrieval evaluation, not real-corpus or answer-quality validation.",
              "k": args.k, "repeats": args.repeats, **provenance(), "arms": arms,
              "measurement": "Each arm uses a fresh subprocess. RSS is sampled every 10 ms, including setup/model residency; brief peaks can be missed. Latency excludes setup and one warmup. Fixed arm order is not a paired speedup experiment."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_name("raw_results.jsonl")
    raw_path.write_text("\n".join(json.dumps({"arm": arm["arm"], **row}) for arm in arms for row in arm.pop("rows")) + "\n")
    report["raw_results"] = raw_path.name
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
