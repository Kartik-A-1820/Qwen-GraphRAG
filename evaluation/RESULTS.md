# Recorded CPU results

One recorded run on Linux x86_64, Python 3.12.14. Two OpenMP/MKL threads; pinned models and exact dependency versions are recorded in [summary.json](results/summary.json). These results describe the fictional fixture corpus only.

| Arm | Recall@3 | nDCG@3 | MRR | p50 ms | p95 ms | Sampled peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.972 | 0.977 | 1.000 | 15.40 | 18.76 | 692.8 |
| bm25 | 0.935 | 0.910 | 0.912 | 0.21 | 0.28 | 122.7 |
| hybrid | 0.972 | 0.968 | 0.986 | 14.27 | 18.58 | 694.8 |
| graph | 0.972 | 0.958 | 0.972 | 16.12 | 21.42 | 693.1 |
| reranked | 0.981 | 0.973 | 0.986 | 192.47 | 288.50 | 795.2 |

Quality averages cover 36 answerable queries. All arms returned candidates for all four unanswerable queries; this is not an answer-abstention score. Each query has three timed repeats after one warmup. Setup is excluded from latency and included in sampled RSS. Arms run in separate processes, in fixed order.

Graph candidates did not increase recall relative to dense/hybrid retrieval and reduced nDCG. Reranking increased recall slightly, with higher latency. Treat these as findings to investigate on independently labeled real data, not evidence of a general advantage.

See [methodology](README.md) and the [200 raw rankings](results/raw_results.jsonl) before comparing changes.
