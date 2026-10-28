"""Build one bounded context and return exactly the excerpts it contains."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class GroundedContext:
    text: str
    references: list[dict]


def build_context(documents, max_chars: int) -> GroundedContext:
    blocks, references = [], []
    remaining = max(0, max_chars)
    for doc in documents:
        if not doc.page_content.strip():
            continue
        metadata = doc.metadata or {}
        label = f"S{len(references) + 1}"
        source = " ".join(str(metadata.get("source", "unknown")).split())
        header = f"[{label}] Source: {source}, Page: {metadata.get('page', 'N/A')}, Chunk: {metadata.get('chunk_index', 'N/A')}\n"
        separator = "\n\n" if blocks else ""
        available = remaining - len(separator) - len(header)
        if available <= 0:
            break
        excerpt = doc.page_content[:available]
        block = separator + header + excerpt
        blocks.append(block)
        remaining -= len(block)
        references.append({
            "id": label, "source": source, "page": metadata.get("page", "N/A"),
            "chunk": metadata.get("chunk_index", "N/A"),
            "chunk_id": metadata.get("chunk_id"), "content": excerpt,
        })
    return GroundedContext("".join(blocks), references)


def citation_status(answer, references):
    if answer.strip().casefold().rstrip(".") == "not found in the provided reference":
        return "abstained"
    labels = set(re.findall(r"\[(S\d+)\]", answer))
    if not labels:
        return "missing"
    if labels - {ref["id"] for ref in references}:
        return "invalid"
    return "valid"
