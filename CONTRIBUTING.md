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

## Data and credentials

Do not commit weights, indexes, private documents, credentials, or author-specific machine paths. Prefer synthetic or explicitly licensed redistributable fixtures. Keep model/database dependencies out of lightweight helpers when practical.

See [SECURITY.md](SECURITY.md) for local-use boundaries. There is no standalone LICENSE file; identify third-party sources and terms in proposed additions and discuss reuse questions with the maintainer.
