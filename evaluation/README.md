# Retrieval evaluation and model smoke test

This directory contains a reproducible comparison and its recorded outputs. It is deliberately small enough to inspect: 24 authored, fictional handbook documents and 40 queries. It does not establish performance on a real organization's documents or evaluate answer faithfulness at scale.

## Data and judgments

`corpus.jsonl` contains document IDs, titles, source names, and text. `queries.jsonl` contains the question, category, relevant document IDs, and an expected answer or manual grading rubric.

| Category | Queries | Purpose |
| --- | ---: | --- |
| Factual | 12 | Direct facts and numeric thresholds |
| Paraphrase | 12 | Wording different from the source |
| Multi-hop | 8 | Evidence distributed across documents |
| Ambiguous | 4 | Similar entities or policies requiring discrimination |
| Unanswerable | 4 | No supporting evidence in the corpus |

The documents and relevance judgments were authored together for this project; they have not received independent annotation. Multi-hop labels indicate the evidence required, not that the generator successfully reasons across it. Inspect labels before drawing conclusions or adding data. These fixtures contain no real customer or medical records and use no external dataset.

## Matched comparison

All arms use the same documents, labels, model revisions, candidate limits, and final cutoff of three documents. Each fixture document is one chunk, so scores are document retrieval scores. Arms are:

1. **Dense:** MiniLM embeddings with Chroma similarity search.
2. **BM25:** lexical retrieval only.
3. **Hybrid:** dense + BM25 with reciprocal-rank fusion.
4. **Graph:** hybrid + rule-extracted graph candidates with reciprocal-rank fusion.
5. **Reranked:** graph fusion followed by a MiniLM cross-encoder.

Candidate limits are eight per retrieval signal; fusion uses RRF constant 60. Reranking scores the fused candidate set before choosing three results. Graph extraction uses rules, not Qwen. Generation is excluded from retrieval timings.

Recall@3, nDCG@3, and MRR average the 36 queries with relevance labels. Binary relevance treats every labeled document as relevant. Unknown queries are reported separately: candidate-return rate measures whether retrieval returned anything, **not whether an answer should be trusted or whether the generator abstained**. Every arm returned candidates for all four unknown queries; no retrieval rejection threshold is calibrated here.

Each arm runs in a fresh subprocess, performs one unscored warmup, and answers all queries three times in seeded shuffled order. Rankings must remain stable across repeats. Latency includes query embedding, retrieval/fusion, and reranking where applicable; setup/download time is separate. RSS samples every 10 ms include model loading and residency, but may miss brief peaks. Arms run in fixed order, so these timings are descriptive, not a paired speedup experiment.

## Reproduce

Use the Linux x86_64 / Python 3.12 CPU lockfile setup in the root README. Run from the repository root, sequentially:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_DISABLE_XET=1 python -m backend.evaluation.benchmark
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_DISABLE_XET=1 python -m backend.evaluation.model_smoke
```

The benchmark downloads the embedding and reranking weights. The separate smoke downloads Qwen2.5-1.5B-Instruct and uses CPU float32; the recorded run required roughly 7.5 GiB peak RSS, so leave additional headroom. Pinned public revisions are in `models.json`. No hosted inference API or credentials are required.

The benchmark work directory defaults to `.local/evaluation`. Use `--work-dir .local/another-run` for a clean index, and `--output .local/results/summary.json` to avoid replacing recorded results. The smoke always uses a temporary data directory and accepts `--output` too. Other hardware/platforms should use suitable dependencies and record their resolved versions.

## Evidence and interpretation

The recorded [summary](results/summary.json) contains per-arm and per-category metrics, model revisions, package versions, corpus hashes, source hashes, memory, and timing details. [Raw results](results/raw_results.jsonl) retain all 200 query/arm rankings and three timings each. [The result table](RESULTS.md) provides a compact view.

On this corpus, adding graph candidates did not improve recall over dense/hybrid retrieval and reduced nDCG. Reranking increased recall slightly at a latency cost. These results support investigating graph noise and testing independently labeled real data before claiming a graph advantage.

[The actual-model smoke](results/model_smoke.json) exercises API ingestion, real embeddings/Chroma, pinned Qwen generation, a supported citation, one unknown-question abstention, restart deduplication, and temporary-file cleanup. It covers one fictional document and two questions. It is a functional compatibility check; label validity does not establish claim support across arbitrary questions. The first local attempt returned the correct fact without a citation; the implementation now requires source labels, allows one repair, and withholds a second invalid draft.

CI reruns the locked integration on pull requests and publishes fresh JSON artifacts. Recorded wall times are specific to their execution environment. CPU results say nothing about CUDA quantization or model-based graph extraction, neither of which is covered by this run.
