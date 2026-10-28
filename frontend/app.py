import html
import math
import os
import time

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Qwen-GraphRAG", layout="wide")

API_URL = st.sidebar.text_input("Backend URL", value=os.getenv("RAG_API_URL", "http://localhost:8000"))
st.sidebar.caption("Local-only FastAPI backend")


def api_get(path: str, **params):
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def api_post(path: str, **kwargs):
    try:
        response = requests.post(f"{API_URL}{path}", timeout=300, **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
        return None, detail


def metric_row(items: list[tuple[str, object]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def render_graph(graph: dict) -> None:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        st.info("No graph entities yet. Ingest a document to build the local knowledge graph.")
        return

    width = 920
    height = 540
    cx = width / 2
    cy = height / 2
    radius = min(width, height) * 0.36
    positions = {}

    for index, node in enumerate(nodes):
        angle = 2 * math.pi * index / max(len(nodes), 1)
        positions[node["id"]] = (
            cx + radius * math.cos(angle),
            cy + radius * math.sin(angle),
        )

    edge_svg = []
    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        opacity = min(0.85, 0.25 + edge.get("weight", 1) * 0.08)
        edge_svg.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#708090" stroke-width="1.4" stroke-opacity="{opacity:.2f}" />'
        )

    node_svg = []
    for node in nodes:
        x, y = positions[node["id"]]
        label = html.escape(node["label"])
        weight = node.get("weight", 1)
        size = min(30, 12 + weight * 4)
        node_svg.append(
            f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="{size}" fill="#2563eb" '
            f'fill-opacity="0.88" stroke="#0f172a" stroke-width="1.5" />'
            f'<text x="{x:.1f}" y="{y + size + 14:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#0f172a">{label}</text></g>'
        )

    svg = f"""
    <div style="width:100%; overflow:auto; border:1px solid #d9e2ec; border-radius:8px; background:#f8fafc;">
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">
        <rect width="100%" height="100%" fill="#f8fafc"></rect>
        {''.join(edge_svg)}
        {''.join(node_svg)}
      </svg>
    </div>
    """
    components.html(svg, height=height + 24, scrolling=False)


st.title("Qwen-GraphRAG")
st.caption("Document Q&A with dense, BM25, and graph retrieval, rank fusion, and local Qwen generation")

health, health_error = api_get("/health")
if health_error:
    st.warning(f"Backend is not reachable: {health_error}")
else:
    graph_stats = health.get("graph", {})
    st.sidebar.caption(f"Generation model: {health.get('model_id', 'unknown')}")
    metric_row([
        ("Runtime", health.get("runtime", "unknown")),
        ("4-bit", str(health.get("load_in_4bit", False))),
        ("Graph DB", health.get("graph_backend", graph_stats.get("backend", "unknown"))),
        ("Graph Entities", graph_stats.get("entities", 0)),
        ("Graph Relations", graph_stats.get("relations", 0)),
    ])
    st.sidebar.caption(f"Graph extraction: {health.get('graph_extraction_mode', 'rules')}")

tab_ask, tab_graph, tab_monitor, tab_summary = st.tabs([
    "Ask",
    "GraphRAG",
    "Monitor",
    "Summarize",
])

with tab_ask:
    left, right = st.columns([0.35, 0.65])
    with left:
        st.subheader("Ingest")
        uploaded_file = st.file_uploader(
            "Upload PDF, TXT, DOCX, or CSV",
            type=["pdf", "txt", "doc", "docx", "csv"],
        )
        if uploaded_file and st.button("Index Document", use_container_width=True):
            with st.spinner("Parsing, chunking, embedding, and updating graph..."):
                result, error = api_post(
                    "/ingest",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                )
            if error:
                st.error(f"Ingest failed: {error}")
            else:
                st.success(result.get("message", "Document indexed."))
                metric_row([
                    ("Chunks", result.get("num_documents", 0)),
                    ("Entities", result.get("graph", {}).get("entities", 0)),
                    ("Relations", result.get("graph", {}).get("relations", 0)),
                ])

    with right:
        st.subheader("Query")
        query = st.text_input("Question", placeholder="Ask about your indexed documents")
        if st.button("Run Hybrid GraphRAG", type="primary", use_container_width=True):
            if not query:
                st.warning("Enter a question first.")
            else:
                with st.spinner("Retrieving dense, sparse, and graph evidence..."):
                    result, error = api_post("/query/local", json={"query": query})
                if error:
                    st.error(f"Query failed: {error}")
                else:
                    debug = result.get("retrieval_debug", {})
                    metric_row([
                        ("Dense", debug.get("dense_count", 0)),
                        ("BM25", debug.get("sparse_count", 0)),
                        ("Graph", debug.get("graph_count", 0)),
                        ("Fused", debug.get("fused_count", 0)),
                    ])
                    st.markdown("### Answer")
                    st.markdown(result.get("answer", ""))
                    if result.get("citation_status") in {"missing", "invalid"}:
                        st.warning("The model did not produce valid source labels. Review the excerpts or try a narrower question.")

                    st.markdown("### Sources")
                    st.caption("These are the excerpts provided to the model. Check the answer against their text.")
                    for index, ref in enumerate(result.get("references", []), start=1):
                        with st.expander(
                            f"[{ref.get('id', 'S' + str(index))}] {ref.get('source', 'Document')} | page {ref.get('page', 'N/A')} | chunk {ref.get('chunk', 'N/A')}"
                        ):
                            st.write(ref.get("content", ""))

with tab_graph:
    st.subheader("Knowledge Graph")
    col_a, col_b = st.columns([0.25, 0.75])
    with col_a:
        limit = st.slider("Entities to show", min_value=8, max_value=80, value=32, step=4)
        refresh_graph = st.button("Refresh Graph", use_container_width=True)
    graph, graph_error = api_get("/graph", limit=limit)
    if graph_error:
        st.error(f"Graph unavailable: {graph_error}")
    else:
        stats = graph.get("stats", {})
        metric_row([
            ("Entities", stats.get("entities", 0)),
            ("Relations", stats.get("relations", 0)),
            ("Relation Types", stats.get("relation_types", 0)),
            ("Documents", stats.get("documents", 0)),
        ])
        metric_row([
            ("LLM Extracted", stats.get("llm_extractions", 0)),
            ("Fallback Extracted", stats.get("fallback_extractions", 0)),
        ])
        render_graph(graph)
        if graph.get("nodes"):
            st.markdown("### Top Entities")
            st.dataframe(pd.DataFrame(graph["nodes"]), use_container_width=True, hide_index=True)
        if graph.get("edges"):
            st.markdown("### Strong Relations")
            st.dataframe(pd.DataFrame(graph["edges"]), use_container_width=True, hide_index=True)

with tab_monitor:
    st.subheader("Runtime Monitor")
    col_a, col_b, col_c = st.columns([0.2, 0.2, 0.6])
    auto_refresh = col_a.checkbox("Auto refresh", value=False)
    if col_b.button("Reset Metrics", use_container_width=True):
        _result, error = api_post("/metrics/reset")
        if error:
            st.error(f"Reset failed: {error}")
        else:
            st.success("Metrics reset.")

    metrics, metrics_error = api_get("/metrics")
    if metrics_error:
        st.error(f"Metrics unavailable: {metrics_error}")
    else:
        ingest = metrics.get("ingest", {})
        query_metrics = metrics.get("query", {})
        graph_metrics = metrics.get("graph", {})
        errors = metrics.get("errors", {})

        st.markdown("### Ingestion")
        metric_row([
            ("Ingests", ingest.get("count", 0)),
            ("Chunks", ingest.get("total_chunks", 0)),
            ("Duplicates", ingest.get("duplicates", 0)),
            ("Avg ms", ingest.get("avg_latency_ms", 0)),
        ])

        st.markdown("### Retrieval And Generation")
        metric_row([
            ("Queries", query_metrics.get("count", 0)),
            ("Avg ms", query_metrics.get("avg_latency_ms", 0)),
            ("P95 ms", query_metrics.get("p95_latency_ms", 0)),
            ("Errors", errors.get("count", 0)),
        ])
        metric_row([
            ("Avg Dense", query_metrics.get("avg_dense_count", 0)),
            ("Avg BM25", query_metrics.get("avg_sparse_count", 0)),
            ("Avg Graph", query_metrics.get("avg_graph_count", 0)),
            ("Avg Fused", query_metrics.get("avg_fused_count", 0)),
        ])

        st.markdown("### Graph Index")
        metric_row([
            ("Entities", graph_metrics.get("entities", 0)),
            ("Relations", graph_metrics.get("relations", 0)),
            ("Relation Types", graph_metrics.get("relation_types", 0)),
            ("Documents", graph_metrics.get("documents", 0)),
        ])
        metric_row([
            ("LLM Extracted", graph_metrics.get("llm_extractions", 0)),
            ("Fallback Extracted", graph_metrics.get("fallback_extractions", 0)),
        ])

        recent_queries = metrics.get("recent_queries", [])
        if recent_queries:
            st.markdown("### Recent Queries")
            st.dataframe(pd.DataFrame(recent_queries), use_container_width=True, hide_index=True)

        recent_errors = errors.get("recent", [])
        if recent_errors:
            st.markdown("### Recent Errors")
            st.dataframe(pd.DataFrame(recent_errors), use_container_width=True, hide_index=True)

    if auto_refresh:
        time.sleep(3)
        st.rerun()

with tab_summary:
    st.subheader("Summarize")
    summary_file = st.file_uploader(
        "Upload PDF, TXT, DOCX, or CSV",
        type=["pdf", "txt", "doc", "docx", "csv"],
        key="summary",
    )
    if summary_file and st.button("Generate Summary", type="primary", use_container_width=True):
        with st.spinner("Generating local summary..."):
            result, error = api_post(
                "/summarize_pdf",
                files={"file": (summary_file.name, summary_file.getvalue(), summary_file.type)},
            )
        if error:
            st.error(f"Summary failed: {error}")
        else:
            st.markdown("### Final Summary")
            st.markdown(result.get("summary", "No summary returned."))
