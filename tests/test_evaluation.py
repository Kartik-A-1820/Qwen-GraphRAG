from backend.evaluation.benchmark import aggregate, dataset


def test_labeled_corpus_has_distinct_queries_and_supported_judgments():
    corpus, queries = dataset()
    assert len(corpus) == 24
    assert len(queries) == 40
    assert {q["category"] for q in queries} == {"factual", "paraphrase", "multi_hop", "ambiguous", "unanswerable"}
    for query in queries:
        assert query["expected"]
        if query["category"] == "multi_hop":
            assert len(query["relevant_ids"]) >= 2


def test_unknown_queries_are_not_counted_as_failed_answerable_recall():
    report = aggregate([
        {"relevant_ids": ["a"], "retrieved_ids": ["a"], "latencies_ms": [1]},
        {"relevant_ids": [], "retrieved_ids": ["b"], "latencies_ms": [3]},
    ], 3)
    assert report["recall@3"] == 1
    assert report["unanswerable_candidate_return_rate"] == 1
    assert report["latency_p50_ms"] == 2
