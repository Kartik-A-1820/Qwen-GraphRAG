from langchain_core.documents import Document

from backend.context import build_context, citation_status


def test_budget_counts_headers_and_excludes_unseen_sources():
    docs = [Document(page_content="a" * 100, metadata={"source": "first.txt"}),
            Document(page_content="unseen", metadata={"source": "second.txt"})]
    context = build_context(docs, 85)
    assert len(context.text) == 85
    assert len(context.references) == 1
    assert context.references[0]["content"] in context.text
    assert len(context.references[0]["content"]) < 100
    assert "second.txt" not in context.text


def test_every_reference_identifies_an_exact_context_excerpt():
    docs = [Document(page_content="first fact", metadata={"source": "a.txt", "chunk_id": "a"}),
            Document(page_content="second fact", metadata={"source": "b.txt", "chunk_id": "b"})]
    context = build_context(docs, 500)
    assert [ref["id"] for ref in context.references] == ["S1", "S2"]
    assert [ref["chunk_id"] for ref in context.references] == ["a", "b"]
    for ref in context.references:
        assert f"[{ref['id']}]" in context.text
        assert ref["content"] in context.text
    assert build_context(docs, 0).references == []
    assert build_context(docs, 5).text == ""


def test_citation_labels_are_validated_without_inventing_attribution():
    refs = [{"id": "S1"}]
    assert citation_status("A fact. [S1]", refs) == "valid"
    assert citation_status("A fact. [S2]", refs) == "invalid"
    assert citation_status("A fact.", refs) == "missing"
    assert citation_status("Not found in the provided reference.", refs) == "abstained"
