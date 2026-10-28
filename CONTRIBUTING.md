# Contributing to Qwen-GraphRAG

Start with a small reproducible issue or documentation correction. For a larger retrieval or model change, describe the expected behavior and evaluation plan in an issue first.

## Development workflow

1. Fork and create a branch. Use an isolated Python 3.11 or 3.12 environment.
2. Follow the offline setup in [README.md](README.md): `python -m pip install -r requirements-test.txt`.
3. Reproduce the behavior with a small synthetic or public, redistributable fixture.
4. Add focused tests for changed behavior and run `python -m pytest -q`.
5. Include the problem, implementation, validation, and remaining limits in the pull request.

Install `kuzu>=0.11,<0.12` to exercise native graph persistence; without it the native test skips. Fake-model generation tests check formatting and decoding contracts. Actual model, quantization, embedding, and dashboard changes need an appropriately configured integration run in addition to unit tests.

## Retrieval and model experiments

Keep the corpus, queries, relevance labels, chunking, and baseline matched. Record the commit, package versions, model ID/revision, hardware, settings, and corpus scope.

Report retrieval quality separately from latency, memory, and answer faithfulness. A fixture test is not a real-data benchmark. Preserve unsuccessful runs. Graph-extraction reports should include malformed-output and rule-fallback behavior, not only successful triples.

## Reproduce the integration evidence

Use Linux x86_64, Python 3.12, and the CPU lockfile setup in the README. Then run these sequentially:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m backend.evaluation.model_smoke
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m backend.evaluation.benchmark
```

They download public model weights on first use. The Qwen smoke run peaked around 7.5 GiB RSS in the recorded environment; allow additional headroom. CI runs the locked integration job on pull requests and uploads fresh results as artifacts. Committed results are a recorded run, not an assertion that timings are portable.

See [evaluation/README.md](evaluation/README.md) for labels and measurement limits. Keep failing and negative results visible. Do not adjust relevance labels just to favor a retrieval arm. Regenerate the lock after dependency edits:

```bash
uv pip compile requirements.txt requirements-test.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28 --torch-backend cpu --generate-hashes --emit-index-url -o requirements-cpu-py312.lock
```

Keep API mutations behind its worker and preserve the single-owner data-directory contract. Recovery tests should inject a failure between durable catalog preparation and projection completion, then verify retrieval and deduplication after restart.

## Data and credentials

Do not commit weights, indexes, private documents, credentials, or author-specific machine paths. Prefer synthetic or explicitly licensed redistributable fixtures. Keep model/database dependencies out of lightweight helpers when practical.

See [SECURITY.md](SECURITY.md) for local-use boundaries. There is no standalone LICENSE file; identify third-party sources and terms in proposed additions and discuss reuse questions with the maintainer.
