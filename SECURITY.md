# Security and data boundaries

Qwen-GraphRAG is intended for trusted local experimentation. The API has no authentication or tenant isolation. Upload bytes, extracted characters, PDF pages, query length, and queued work are bounded, but parsers and model execution have no hard process-level memory/time limits. Loopback defaults do not turn it into a hardened multi-user service.

Models, embeddings, the SQLite catalog, graph-extraction cache, indexes, and runtime metrics may contain private information. Temporary uploads are deleted after processing; extracted text persists until the data directory is removed. Back up related data files together. Model artifacts can be downloaded on first use; pre-cache them for offline operation. Review model code and configuration before substituting a different model.

Source passages are untrusted model input. Citation-label validation detects missing or unknown labels; it does not verify claim support or prevent every prompt-injection attack.

Keep `.env`, weights, documents, and generated indexes out of commits. Do not publish private questions or source passages in issue logs.

For a vulnerability, contact [anagawadi.kartik@gmail.com](mailto:anagawadi.kartik@gmail.com) privately with a minimal redacted reproduction. Do not include working credentials or private documents.
