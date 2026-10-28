# Qwen-GraphRAG

**Local document Q&A with Qwen and hybrid dense, lexical, and graph retrieval.**

Qwen-GraphRAG combines Chroma vector search, BM25, a Kuzu knowledge graph, reciprocal-rank fusion, and optional reranking behind a FastAPI API and Streamlit dashboard. The default generation model is **`Qwen/Qwen2.5-1.5B-Instruct`**. Both answer generation and optional LLM graph extraction are configurable.

Use it to inspect how retrieval signals affect answers grounded in your own documents. This is a local experimentation platform; the included evaluation measures retrieval on a small fictional corpus, not real-world answer quality.

## Implementation

- PDF, TXT, DOCX, and CSV ingestion with file hashing and stable chunk IDs.
- Dense, lexical, and graph candidates fused before optional reranking.
- Rule-based entity/relation extraction by default; optional local-model extraction.
- Exact context excerpts with source labels, citation validation, graph inspection, and runtime monitoring.
- A durable SQLite ingestion catalog, incremental indexes, cached graph extraction, and interrupted-write recovery.
- Upload/query limits and one background worker with a bounded waiting queue.
- Lazy generation-model loading, optional CUDA 4-bit loading, and a configurable Kuzu buffer pool.

The Qwen tokenizer's chat template formats generation input. The decoder returns only newly generated tokens. Alternative text models require a compatible tokenizer/chat template and their own validation.

## Choose a starting point

| Goal | Start here | Downloads |
| --- | --- | --- |
| Inspect retrieval and scoring | Offline checks below | Small test dependencies; no model weights |
| Run document Q&A | Local application setup | Embedding and Qwen weights on first use |
| Compare retrieval approaches | [Five-arm evaluation](evaluation/README.md) | Pinned embedding/reranking weights |
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

Tests cover ranking, exact context/reference alignment, incremental indexing, interrupted-write recovery, parser limits, queue saturation, cancellation, and generation formatting. Unit tests use fake models. A separate integration run uses actual pinned Qwen weights through the API; see [evaluation evidence](evaluation/README.md). CUDA compatibility is not tested.

## Run the application

Install a PyTorch build appropriate to your platform first. For CPU use:

```bash
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For the tested **Linux x86_64 / Python 3.12 CPU environment**, use the complete hash-locked dependencies instead of the range-based install above:

```bash
python -m pip install uv==0.13.4
uv pip sync --python .venv/bin/python --torch-backend cpu --require-hashes requirements-cpu-py312.lock
```

The lock targets Linux; use platform-appropriate requirements on other systems. The `--torch-backend cpu` flag is required for the locked `+cpu` wheel.

Copy `config/local.example.env` to `.env` at the repository root. The example starts with CPU generation, rule-based graph extraction, 4-bit loading disabled, and reranking disabled.

```bash
python run_project.py
```

Open `http://127.0.0.1:8501`. Run one API worker per data directory; a file lock rejects a second owner. The launcher uses the active Python environment, checks `/health` before starting the UI, binds both services to loopback, and stops them when either exits or you press Ctrl+C. It does not install packages or replace your PyTorch build.

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
| `RAG_MODEL_REVISION`, `RAG_EMBEDDING_REVISION`, `RAG_RERANKER_REVISION`, `RAG_GRAPH_EXTRACTOR_REVISION` | Optional immutable model revisions; the example pins the evaluated generation/embedding/reranking models |
| `RAG_CATALOG_PATH` | Canonical chunk ledger and graph-extraction cache |
| `RAG_DEVICE`, `RAG_LOAD_IN_4BIT` | Placement and optional quantization; CPU/unquantized by default |
| `RAG_GRAPH_BACKEND` | Kuzu store, with JSON fallback if native setup fails |
| `RAG_GRAPH_BUFFER_POOL_MB` | Kuzu buffer pool; 128 MB by default, separate from total process memory |
| `RAG_GRAPH_EXTRACTION_MODE` | `rules` or `llm` |
| `RAG_GRAPH_EXTRACTOR_MODEL` | Model used when LLM graph extraction is enabled |
| `RAG_CHUNK_SIZE`, `RAG_FUSED_K`, `RAG_MAX_CONTEXT_CHARS` | Retrieval/context tradeoffs |
| `RAG_ENABLE_RERANKER`, `RAG_RERANKER_MODEL` | Optional second-stage reranking |
| `RAG_MAX_UPLOAD_BYTES`, `RAG_MAX_DOCUMENT_CHARS`, `RAG_MAX_DOCUMENT_PAGES` | Upload, extracted-text, and PDF page limits |
| `RAG_MAX_QUERY_CHARS`, `RAG_WORK_QUEUE_SIZE` | Query length and pending-job capacity |
| `RAG_API_URL` | UI backend address for manual/custom deployment |

For a compatible NVIDIA CUDA environment, install the appropriate PyTorch build and then the application and optional quantization requirements:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-4bit.txt
```

Set `RAG_DEVICE=auto` and `RAG_LOAD_IN_4BIT=true` in `.env` before restarting. The runner rejects 4-bit loading on CPU. Compatibility depends on the selected GPU, driver, PyTorch, and bitsandbytes versions. See the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [PyTorch installation selector](https://pytorch.org/get-started/locally/).

The original development target included a 4 GB GPU. It is not a memory-fit guarantee: model size, context, reranking, and graph extraction all affect resource use. LLM graph extraction creates a separate generator and can increase model residency; start with rule extraction on constrained machines.

Requirements use a bounded Transformers 4.x/LangChain 0.3 compatibility family, not the former model-specific Git build. The range files support other environments; `requirements-cpu-py312.lock` records the tested CPU environment with exact versions and hashes. Record resolved versions when reporting a different environment. `requirements-cuda-cu130.txt` remains an explicit hardware-specific override; the launcher never applies it automatically.

## Evaluate and inspect

[Evaluation setup and recorded results](evaluation/README.md) compare dense, BM25, dense+BM25 fusion, graph-assisted fusion, and cross-encoder reranking on 24 fictional documents and 40 labeled queries. Raw rankings, timings, model revisions, dependency versions, and dataset/source hashes are committed alongside the summary. The original two-query fixture remains a metric-plumbing test.

References contain exactly the excerpts included in the generation context, with headers counted against the context budget. Answers must use valid `[S1]` labels; the API makes one repair attempt and withholds answers that still have missing/unknown labels. **A valid label does not prove the claim follows from its source.** Inspect excerpts when assessing answer faithfulness.

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

The SQLite catalog is the ingestion ledger. New documents receive deterministic chunk IDs; vectors are upserted, BM25 and graph projections are extended, and graph extraction is cached by text and extraction configuration. A failed write leaves the ledger pending; restart or the next query rebuilds projections before serving retrieval. Existing Chroma records are adopted once when the catalog is empty. Back up the entire `data/` directory together; this is interrupted-write recovery, not a substitute for backups.

Changing embedding identity/revision or chunking on an existing catalog is rejected. Use a new data directory and re-ingest original files for those changes. Changing graph extraction settings rebuilds the graph while reusing matching cached extractions. Default filenames deduplicate by file content, so identical uploads under another name retain the first source name.

Defaults allow 10 MiB per upload, 1,000,000 extracted characters, 500 PDF pages, and 4,000 query characters. One worker executes parsing, indexing, and inference; two jobs may wait. A full queue returns HTTP 503 with `Retry-After`. Health and metrics remain available during worker execution. Client cancellation does not free the worker early, and shutdown drains accepted work. Threads cannot forcibly interrupt a stuck parser/model; there is no hard execution deadline.

Temporary upload copies are deleted after work completes or fails. Extracted text remains in the catalog/indexes. Generated documents, indexes, cached weights, and `.env` belong outside version control. Use a trusted local environment; see [SECURITY.md](SECURITY.md).

## Contribute

[CONTRIBUTING.md](CONTRIBUTING.md) describes setup, tests, and experiment reporting. Useful starting areas include independently labeled real-corpus evaluations, answer-faithfulness checks, parser edge cases, data migrations, and graph-extraction error analysis.

No standalone LICENSE file is currently included. Clarify repository reuse terms with the maintainer; model and dataset terms are separate.
