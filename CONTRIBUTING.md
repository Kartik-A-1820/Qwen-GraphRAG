# Contributing to Qwen-GraphRAG

Start with a small reproducible issue or documentation correction. For a larger retrieval/graph change, describe the expected behavior and evaluation plan in an issue before implementation.

1. Fork or create a branch and use a separate Python 3.11 environment.
2. Follow the manual setup in [README.md](README.md). Record dependency installation failures rather than hiding them with untracked local changes.
3. Reproduce the behavior on a small non-sensitive document or fixture. Include expected versus actual references/ranking and relevant environment settings.
4. Add or update tests when behavior changes, then run `python -m pytest -q tests/test_retrieval.py`.
5. Open a focused pull request describing the problem, change, validation, and remaining limits.

Retrieval changes should compare against the existing baseline on the same corpus and queries. Report quality separately from latency and memory; a successful fixture test is not a real-data benchmark. Graph extraction should record invalid JSON/fallback behavior as well as successful triples.

Do not commit model weights, indexes, private documents, credentials, or local machine paths. Prefer synthetic or explicitly licensed, redistributable fixtures. Keep optional model dependencies out of lightweight helpers where possible.

There is currently no standalone LICENSE file. Do not introduce third-party code or datasets without stating their source and terms; discuss reuse/licensing questions with the maintainer.
