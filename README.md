# Qwen-GraphRAG

**Local document question answering with dense, lexical, and graph retrieval.**

This project combines Chroma embeddings, BM25, a Kuzu knowledge graph, reciprocal-rank fusion, and optional reranking behind a FastAPI API and Streamlit dashboard. Use it to explore how different retrieval signals affect document answers while keeping model inference and indexes on your own machine.

## Engineering scope

- Multi-format ingestion: PDF, TXT, DOCX, and CSV; hashing and stable chunk IDs support duplicate handling and result fusion.
- Typed entity/relation extraction with a local language model, plus a rule-based fallback.
- Dense, sparse, and graph candidates fused before optional reranking and answer generation.
- Retrieval-quality utilities and runtime latency/error monitoring, with source references returned alongside answers.

The implementation is a local experimentation platform. Retrieval metrics on the included two-query fixture check metric plumbing; they do not establish real-corpus retrieval quality or answer faithfulness. Graph extraction, reranking, and model choice need evaluation on your documents.

## Choose a starting point

| Goal | Starting point | Downloads/hardware |
| --- | --- | --- |
| Inspect retrieval and test logic | [tests/test_retrieval.py](tests/test_retrieval.py), [backend/retrieval](backend/retrieval/) | Most tests use fixtures/fakes; native Kuzu tests have their own dependency |
| Run document Q&A | Manual setup below | Embedding and causal-LM weights; CPU or a compatible GPU |
| Explore graph extraction | Set `RAG_GRAPH_EXTRACTION_MODE=llm` | Local model inference during ingestion; slower than rule extraction |

## Local setup

Use Python 3.11. Run commands from the repository root:

```bash
git clone https://github.com/Kartik-A-1820/Qwen-GraphRAG.git
cd Qwen-GraphRAG
python -m venv .venv
```

Activate the environment:

| Shell | Command |
| --- | --- |
| Linux/macOS | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |

```bash
python -m pip install -r requirements.txt
```

**Dependency note:** the current requirements still reference a Gemma-specific Transformers Git tag and leave several packages unpinned. This is a known reproducibility constraint, not a portable lockfile. Installation can depend on Python/platform/package availability. If it fails, include the failing package, Python version, OS, and full error in an issue; do not assume a model or GPU problem. The manual launch path avoids the launcher's automatic CUDA-wheel installation.

Copy `config/local.example.env` to `.env` and edit it before importing the backend. For a CPU starting configuration, change:

```dotenv
RAG_DEVICE=cpu
RAG_LOAD_IN_4BIT=false
RAG_GRAPH_EXTRACTION_MODE=rules
RAG_ENABLE_RERANKER=false
```

Then run the API and UI in separate activated terminals:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
python -m streamlit run frontend/app.py
```

Open http://localhost:8501. API documentation is at http://127.0.0.1:8000/docs.

**First successful run:** upload a small TXT/PDF with known facts, wait for ingestion to finish, ask a question answered by that file, and inspect returned source references and retrieval counts. The API initializes embeddings at startup; the first run can download weights before the service becomes ready. Offline use requires those artifacts to be cached or supplied locally.

## Configuration and hardware

The [example environment](config/local.example.env) lists all supported settings. Key choices:

| Setting | Purpose |
| --- | --- |
| `RAG_MODEL_ID` | Hugging Face causal-LM identifier or a local model directory; example uses Qwen2.5-1.5B-Instruct |
| `RAG_EMBEDDING_MODEL` | Embedding model identifier or local directory |
| `RAG_GRAPH_BACKEND` | Kuzu graph store; implementation can fall back to JSON |
| `RAG_GRAPH_EXTRACTION_MODE` | `rules` for lightweight extraction or `llm` for typed model extraction |
| `RAG_DEVICE`, `RAG_LOAD_IN_4BIT` | CPU/GPU placement and quantized loading |
| `RAG_CHUNK_SIZE`, `RAG_FUSED_K`, `RAG_MAX_CONTEXT_CHARS` | Context/latency/memory tradeoffs |
| `RAG_ENABLE_RERANKER` | Optional second-stage candidate reranking |

Use model directories appropriate to your machine rather than copying an author's drive path. Run from the repository root when using relative paths from the example environment. Change model/extraction settings before restarting the service; reingest legacy indexes that lack stable chunk IDs.

The original development profile used a 4 GB GPU. That is a reference environment, not a requirement or a guarantee: CPU inference is possible, while usable context length and quantization depend on the selected model and platform. The bundled `requirements-cuda-cu130.txt` is a specific CUDA profile, not a universal prerequisite. Keep reranking and graph LLM extraction disabled until the basic path works.

`python run_project.py` is an optional Python-3.11 launcher that provisions its own `venv` and defaults to a CUDA wheel. It is intended for that configured environment; prefer the explicit commands above for other setups.

Docker configuration is included, but the image installs the same requirements and inherits their dependency constraints. Review `docker-compose.yml` before running it; it uses the example environment file directly.

## Evaluate and inspect

After installing project dependencies:

```bash
python -m pytest -q tests/test_retrieval.py
python -m backend.evaluation.run_retrieval_eval tests/fixtures/retrieval_results.jsonl --k 5
```

The offline evaluator consumes saved relevant/retrieved chunk IDs. It computes Recall@K, Precision@K, HitRate@K, nDCG@K, and MRR; it does not itself build a labeled real-document benchmark.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Runtime and graph configuration |
| `GET /metrics` | Ingestion/query latency, counts, errors |
| `GET /graph` | Graph snapshot for inspection |
| `POST /ingest` | Document ingestion |
| `POST /query/local` | Retrieve, generate, and return references |
| `POST /summarize_pdf` | Document summarization |

## Contribute

Documentation, setup reports, labeled retrieval fixtures, and focused bug fixes are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for a repeatable contribution workflow.

Useful starting areas are a reproducible dependency lock, dense/BM25/graph ablations on a shared labeled corpus, graph-extraction error analysis, and tests for ingestion edge cases. Report model ID/revision, hardware, configuration, and corpus scope when sharing results.

This repository currently has no standalone LICENSE file. Model and dataset terms remain separate; clarify intended reuse terms with the maintainer before redistribution.
