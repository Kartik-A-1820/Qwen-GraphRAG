# Qwen-GraphRAG

**Local document Q&A with Qwen and hybrid dense, lexical, and graph retrieval.**

Qwen-GraphRAG combines Chroma vector search, BM25, a Kuzu knowledge graph, reciprocal-rank fusion, and optional reranking behind a FastAPI API and Streamlit dashboard. The default generation model is **`Qwen/Qwen2.5-1.5B-Instruct`**. Both answer generation and optional LLM graph extraction are configurable.

Use it to inspect how retrieval signals affect answers grounded in your own documents. This is a local experimentation platform; no real-corpus answer-quality or model-superiority benchmark is claimed.

## Implementation

- PDF, TXT, DOCX, and CSV ingestion with file hashing and stable chunk IDs.
- Dense, lexical, and graph candidates fused before optional reranking.
- Rule-based entity/relation extraction by default; optional local-model extraction.
- Source references, retrieval counts, graph inspection, and latency/error monitoring.
- Lazy generation-model loading, optional CUDA 4-bit loading, and a configurable Kuzu buffer pool.

The Qwen tokenizer's chat template formats generation input. The decoder returns only newly generated tokens. Alternative text models require a compatible tokenizer/chat template and their own validation.

## Choose a starting point

| Goal | Start here | Downloads |
| --- | --- | --- |
| Inspect retrieval and scoring | Offline checks below | Small test dependencies; no model weights |
| Run document Q&A | Local application setup | Embedding and Qwen weights on first use |
| Compare retrieval approaches | Retrieval code and evaluation utilities | A labeled corpus you provide |
| Explore model-based graph extraction | Set `RAG_GRAPH_EXTRACTION_MODE=llm` | Local inference during ingestion |

## Offline checks

Use Python 3.11 or 3.12 in a separate environment:

```bash
git clone https://github.com/Kartik-A-1820/Qwen-GraphRAG.git
cd Qwen-GraphRAG
python -m venv .venv
```

Activate the environment before installing packages:

| Shell | Command |
| --- | --- |
| Linux/macOS | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |

From the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
python -m backend.evaluation.run_retrieval_eval tests/fixtures/retrieval_results.jsonl --k 5
```

The native Kuzu test skips if Kuzu is absent. To include it, install `kuzu>=0.11,<0.12` and rerun. CI checks the lightweight suite on both Python versions and native graph persistence separately.

Tests cover fusion, ranking, graph extraction/persistence, metric calculations, Qwen-style text-message formatting, continuation-only decoding, and launcher cleanup. Generator tests use a fake model/tokenizer; they do not establish real-model quality or CUDA compatibility.

## Run the application

Install a PyTorch build appropriate to your platform first. For CPU use:

```bash
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Copy `config/local.example.env` to `.env` at the repository root. The example starts with CPU generation, rule-based graph extraction, 4-bit loading disabled, and reranking disabled.

```bash
python run_project.py
```

Open `http://127.0.0.1:8501`. The launcher uses the active Python environment, checks `/health` before starting the UI, binds both services to loopback, and stops them when either exits or you press Ctrl+C. It does not install packages or replace your PyTorch build.

Custom ports and a longer startup allowance:

```bash
python run_project.py --api-port 8100 --ui-port 8600 --startup-timeout 600
```

The launcher passes the API address to the UI. For manual startup, run these in separate activated terminals:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
python -m streamlit run frontend/app.py --server.address 127.0.0.1
```

The API initializes embeddings/indexes at startup and loads the generation model on its first request. Initial weight downloads can take time. Offline operation requires those artifacts to be cached or supplied as local directories.

For a first check, upload a short TXT/PDF with known facts, ask a question answered by that file, and inspect the returned references and retrieval counts. Also test a question the document cannot answer. A fluent answer alone does not establish grounding.

## Model and hardware configuration

[config/local.example.env](config/local.example.env) lists the available settings.

| Setting | Purpose |
| --- | --- |
| `RAG_MODEL_ID` | Causal text-model ID or local directory; defaults to Qwen2.5-1.5B-Instruct |
| `RAG_EMBEDDING_MODEL` | Embedding model ID or local directory |
| `RAG_DEVICE`, `RAG_LOAD_IN_4BIT` | Placement and optional quantization; CPU/unquantized by default |
| `RAG_GRAPH_BACKEND` | Kuzu store, with JSON fallback if native setup fails |
| `RAG_GRAPH_BUFFER_POOL_MB` | Kuzu buffer pool; 128 MB by default, separate from total process memory |
| `RAG_GRAPH_EXTRACTION_MODE` | `rules` or `llm` |
| `RAG_GRAPH_EXTRACTOR_MODEL` | Model used when LLM graph extraction is enabled |
| `RAG_CHUNK_SIZE`, `RAG_FUSED_K`, `RAG_MAX_CONTEXT_CHARS` | Retrieval/context tradeoffs |
| `RAG_ENABLE_RERANKER`, `RAG_RERANKER_MODEL` | Optional second-stage reranking |
| `RAG_API_URL` | UI backend address for manual/custom deployment |

For a compatible NVIDIA CUDA environment, install the appropriate PyTorch build and then the application and optional quantization requirements:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-4bit.txt
```

Set `RAG_DEVICE=auto` and `RAG_LOAD_IN_4BIT=true` in `.env` before restarting. The runner rejects 4-bit loading on CPU. Compatibility depends on the selected GPU, driver, PyTorch, and bitsandbytes versions. See the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [PyTorch installation selector](https://pytorch.org/get-started/locally/).

The original development target included a 4 GB GPU. It is not a memory-fit guarantee: model size, context, reranking, and graph extraction all affect resource use. LLM graph extraction creates a separate generator and can increase model residency; start with rule extraction on constrained machines.

Requirements use a bounded Transformers 4.x/LangChain 0.3 compatibility family, not the former model-specific Git build. They are version ranges, not a complete lockfile. Record resolved versions when reporting an experiment. `requirements-cuda-cu130.txt` remains an explicit hardware-specific override; the launcher never applies it automatically.

## Evaluate and inspect

The included two-query fixture validates metric plumbing. The evaluator consumes saved relevant/retrieved chunk IDs and reports Recall@K, Precision@K, HitRate@K, nDCG@K, and MRR. It does not itself construct a labeled benchmark.

For a meaningful comparison, keep the corpus, chunking, queries, relevance judgments, model revisions, and settings fixed. Compare dense-only, BM25-only, graph-assisted fusion, and optional reranking; report retrieval quality separately from latency and answer faithfulness.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Configured models, graph mode, and runtime status |
| `GET /metrics` | Ingestion/query latency, counts, errors |
| `GET /graph` | Graph snapshot |
| `POST /ingest` | Document ingestion |
| `POST /query/local` | Retrieval, generation, references |
| `POST /summarize_pdf` | Document summarization |

## Containers and local state

The Dockerfile installs CPU PyTorch. `docker compose up --build` runs the API with a loopback port binding and persists `data/`; the Streamlit UI is launched separately. Inspect `docker-compose.yml` before changing ports or mounts.

Generated documents, indexes, cached weights, and `.env` belong outside version control. The API has no authentication or public-service resource controls. Use it in a trusted local environment; see [SECURITY.md](SECURITY.md).

## Contribute

[CONTRIBUTING.md](CONTRIBUTING.md) describes setup, tests, and experiment reporting. Useful starting areas include labeled retrieval fixtures, dense/BM25/graph ablations, ingestion edge cases, dependency locking, and graph-extraction error analysis.

No standalone LICENSE file is currently included. Clarify repository reuse terms with the maintainer; model and dataset terms are separate.
