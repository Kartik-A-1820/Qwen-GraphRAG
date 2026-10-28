# Security and data boundaries

Qwen-GraphRAG is intended for trusted local experimentation. The API has no authentication, tenant isolation, or comprehensive upload/resource limits. Loopback defaults do not turn it into a hardened multi-user service.

Models, embeddings, indexes, uploaded text, and runtime metrics may contain private information. Model artifacts can be downloaded on first use; pre-cache them for offline operation. Review model code and configuration before substituting a different model.

Keep `.env`, weights, documents, and generated indexes out of commits. Do not publish private questions or source passages in issue logs.

For a vulnerability, contact [anagawadi.kartik@gmail.com](mailto:anagawadi.kartik@gmail.com) privately with a minimal redacted reproduction. Do not include working credentials or private documents.
