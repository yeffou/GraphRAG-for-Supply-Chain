"""GraphRAG Supply-Chain QA — Interactive Demo.

A polished, step-based Streamlit UI that walks the viewer through the full
retrieval → graph → answer → comparison pipeline.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from src.baseline_rag.dense_query import run_dense_query
from src.baseline_rag.query import run_baseline_query
from src.config import load_config
from src.evaluation.harness import load_questions, normalize_text
from src.generation import (
    LLMClientError,
    build_grounded_answer_prompt,
    call_chat_completion,
    prompt_to_messages,
    require_generation_config,
)
from src.graph_rag import run_graph_query
from src.graph_rag.hybrid_query import run_hybrid_graph_query

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "GraphRAG"
APP_SUBTITLE = "Supply-Chain Intelligence Demo"
ROOT_DIR = Path(__file__).resolve().parent
DEMO_RUNS_DIR = ROOT_DIR / "results" / "demo_runs"

STEP_LABELS = [
    "Query",
    "Retrieval",
    "Graph Build",
    "Traversal",
    "Answers",
    "Compare",
]
STEP_ICONS = ["01", "02", "03", "04", "05", "06"]
STEP_DESCRIPTIONS = [
    "Select or write a question",
    "Run all four retrieval methods",
    "Construct explanation graph",
    "Traverse graph to find evidence",
    "Generate grounded answers",
    "Compare methods side-by-side",
]

METHOD_ORDER = ["tfidf", "dense", "graph", "hybrid_graph"]
METHOD_LABELS = {
    "tfidf": "TF-IDF",
    "dense": "Dense",
    "graph": "Pure GraphRAG",
    "hybrid_graph": "Hybrid GraphRAG",
}
METHOD_COLORS = {
    "tfidf": "#3B82F6",
    "dense": "#10B981",
    "graph": "#F59E0B",
    "hybrid_graph": "#8B5CF6",
}
METHOD_ICONS = {
    "tfidf": "T",
    "dense": "D",
    "graph": "G",
    "hybrid_graph": "H",
}
METHOD_DESCRIPTIONS = {
    "tfidf": "Sparse keyword matching via TF-IDF cosine similarity",
    "dense": "Semantic search with sentence-transformer embeddings",
    "graph": "Knowledge graph entity/relation path scoring",
    "hybrid_graph": "Dense + graph fusion via Reciprocal Rank Fusion",
}
GRAPH_COLORS = {
    "entity": "#3B82F6",
    "chunk": "#10B981",
    "query": "#F59E0B",
    "selected": "#8B5CF6",
    "edge": "#CBD5E1",
    "path": "#F97316",
}


# ===================================================================
# Entry Point
# ===================================================================


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} — {APP_SUBTITLE}",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _ensure_defaults()

    config = load_config()
    benchmarks = _load_benchmarks()
    examples = _load_examples(config.paths.evaluation_data_dir / "questions.jsonl")

    _render_sidebar(examples)

    step = st.session_state["step"]

    if step == 0:
        render_query_step(examples=examples, benchmarks=benchmarks)
        return

    if not st.session_state.get("active_query"):
        st.session_state["step"] = 0
        st.rerun()

    _query_header(benchmarks)

    if step == 1:
        render_retrieval_step(config=config)
    elif step == 2:
        render_graph_build_step(config=config)
    elif step == 3:
        render_traversal_step(config=config)
    elif step == 4:
        render_answer_step(config=config)
    elif step == 5:
        render_comparison_step(benchmarks=benchmarks)

    _nav_buttons()


# ===================================================================
# Massive CSS (the "wow" layer)
# ===================================================================


def _inject_css() -> None:
    st.markdown(
        """
<style>
/* ---------- global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #0F172A 0%, #1E293B 100%);
}
section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
}

/* sidebar brand */
.sidebar-brand {
    text-align: center;
    padding: 20px 0 10px 0;
}
.sidebar-brand h1 {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #818CF8, #A78BFA, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.sidebar-brand p {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748B !important;
    margin: 4px 0 0 0;
}

/* step indicator */
.step-nav {
    padding: 10px 0;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 9px 14px;
    margin: 2px 0;
    border-radius: 10px;
    transition: background 0.2s;
    cursor: default;
}
.step-item.active {
    background: rgba(139, 92, 246, 0.18);
}
.step-item.done {
    opacity: 0.55;
}
.step-num {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-num.active {
    background: linear-gradient(135deg, #818CF8, #8B5CF6);
    color: #FFFFFF !important;
}
.step-num.done {
    background: #334155;
    color: #94A3B8 !important;
}
.step-num.future {
    background: #1E293B;
    border: 1.5px solid #334155;
    color: #475569 !important;
}
.step-text {
    font-size: 0.82rem;
    font-weight: 500;
    line-height: 1.25;
}
.step-text small {
    display: block;
    font-size: 0.7rem;
    font-weight: 400;
    color: #64748B !important;
    margin-top: 1px;
}

/* ---------- main area ---------- */

/* Hero banner for each step */
.hero {
    background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 50%, #F5F3FF 100%);
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px;
    right: -40px;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
    opacity: 0.08;
}
.hero .step-num-lg {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: linear-gradient(135deg, #818CF8, #8B5CF6);
    color: #FFFFFF;
    font-weight: 800;
    font-size: 0.9rem;
    margin-bottom: 10px;
}
.hero h2 {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 6px 0;
    letter-spacing: -0.01em;
}
.hero p {
    color: #64748B;
    font-size: 0.95rem;
    margin: 0;
    max-width: 600px;
}

/* Card */
.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
    transition: box-shadow 0.2s, transform 0.15s;
}
.card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.03);
    transform: translateY(-1px);
}

/* Method card with colored top stripe */
.method-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.method-card .stripe {
    height: 4px;
}
.method-card .body {
    padding: 16px 20px;
}
.method-card h4 {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
}
.method-card .desc {
    color: #64748B;
    font-size: 0.82rem;
    margin: 4px 0 0 0;
}

/* Chunk card */
.chunk-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.chunk-card .rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-right: 10px;
    flex-shrink: 0;
}
.chunk-card .meta {
    color: #94A3B8;
    font-size: 0.75rem;
    margin-top: 8px;
}
.chunk-card .text-preview {
    color: #334155;
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 8px 0 0 0;
}

/* Chip */
.chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 6px;
    background: #EEF2FF;
    color: #4338CA;
}

/* Active query bar */
.active-q {
    background: linear-gradient(135deg, #0F172A, #1E293B);
    border-radius: 14px;
    padding: 16px 24px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}
.active-q .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8B5CF6;
    font-weight: 600;
}
.active-q .query {
    color: #F1F5F9;
    font-size: 1rem;
    font-weight: 500;
    flex: 1;
}
.active-q .chip-dark {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    background: rgba(139,92,246,0.15);
    color: #A78BFA;
    margin-left: 6px;
}

/* Legend chips */
.legend {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 12px 0;
}
.legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    color: #475569;
}
.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

/* Comparison winner */
.winner-card {
    background: linear-gradient(135deg, #F0FDF4, #ECFDF5);
    border: 2px solid #22C55E;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 14px;
    position: relative;
}
.winner-card::after {
    content: 'BEST';
    position: absolute;
    top: 12px;
    right: 16px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #16A34A;
    background: #DCFCE7;
    padding: 3px 10px;
    border-radius: 999px;
}

/* Metric pill */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.metric-pill {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 16px 22px;
    flex: 1;
    min-width: 120px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.metric-pill .value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.02em;
}
.metric-pill .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-top: 2px;
}

/* Nav buttons */
.nav-row {
    display: flex;
    justify-content: space-between;
    margin-top: 32px;
    padding-top: 20px;
    border-top: 1px solid #E2E8F0;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ===================================================================
# Session State
# ===================================================================


def _ensure_defaults() -> None:
    defs: dict = {
        "step": 0,
        "selected_example_id": None,
        "custom_query_text": "",
        "input_top_k": 5,
        "animation_enabled": True,
        "animation_delay": 0.25,
        "active_query": "",
        "active_top_k": 5,
        "active_signature": "",
        "retrieval_runs": None,
        "retrieval_signature": None,
        "graph_payload": None,
        "graph_signature": None,
        "answers": None,
        "answers_signature": None,
        "saved_demo_signature": None,
        "animated_flags": {},
        "_pctr": 0,  # plotly counter for animation frames
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _pkey(prefix: str) -> str:
    """Unique key for plotly charts in animation frames only."""
    st.session_state["_pctr"] += 1
    return f"p_{prefix}_{st.session_state['_pctr']}"


# ===================================================================
# Sidebar
# ===================================================================


def _render_sidebar(examples: list) -> None:
    with st.sidebar:
        # Brand
        st.markdown(
            f"<div class='sidebar-brand'>"
            f"<h1>{APP_TITLE}</h1>"
            f"<p>{APP_SUBTITLE}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Vertical step indicator
        current = st.session_state["step"]
        items_html = ""
        for i, (label, desc) in enumerate(zip(STEP_LABELS, STEP_DESCRIPTIONS)):
            if i < current:
                cls, ncls = "done", "done"
            elif i == current:
                cls, ncls = "active", "active"
            else:
                cls, ncls = "", "future"
            items_html += (
                f"<div class='step-item {cls}'>"
                f"<div class='step-num {ncls}'>{STEP_ICONS[i]}</div>"
                f"<div class='step-text'>{label}<small>{desc}</small></div>"
                f"</div>"
            )
        st.markdown(
            f"<div class='step-nav'>{items_html}</div>", unsafe_allow_html=True
        )

        st.markdown("---")

        # Controls
        st.markdown(
            "<p style='font-size:0.72rem; text-transform:uppercase; "
            "letter-spacing:0.1em; color:#64748B !important; font-weight:600;'>"
            "Settings</p>",
            unsafe_allow_html=True,
        )
        st.slider("Retrieved chunks", 3, 15, key="input_top_k")
        st.toggle("Animation", key="animation_enabled")
        if st.session_state["animation_enabled"]:
            st.slider("Speed (s)", 0.0, 1.0, step=0.05, key="animation_delay")


# ===================================================================
# Shared UI components
# ===================================================================


def _hero(step_num: int, title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='hero'>"
        f"<div class='step-num-lg'>{step_num:02d}</div>"
        f"<h2>{title}</h2>"
        f"<p>{subtitle}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _query_header(benchmarks: dict) -> None:
    q = st.session_state["active_query"]
    k = st.session_state["active_top_k"]
    bench = benchmarks.get(normalize_text(q))
    chips = f"<span class='chip-dark'>k={k}</span>"
    if bench:
        chips += f"<span class='chip-dark'>{bench['question_id']}</span>"
    st.markdown(
        f"<div class='active-q'>"
        f"<span class='label'>Active Query</span>"
        f"<span class='query'>{q}</span>"
        f"<span>{chips}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _nav_buttons() -> None:
    cur = st.session_state["step"]
    c1, _, c2 = st.columns([1, 2, 1])
    with c1:
        if st.button(
            "Previous",
            disabled=cur <= 0,
            use_container_width=True,
            key="nav_prev",
        ):
            st.session_state["step"] = max(0, cur - 1)
            st.rerun()
    with c2:
        if st.button(
            "Next Step",
            disabled=cur >= len(STEP_LABELS) - 1,
            use_container_width=True,
            type="primary",
            key="nav_next",
        ):
            st.session_state["step"] = min(len(STEP_LABELS) - 1, cur + 1)
            st.rerun()


def _method_card_open(method: str) -> str:
    c = METHOD_COLORS[method]
    return (
        f"<div class='method-card'>"
        f"<div class='stripe' style='background:linear-gradient(90deg,{c},{c}88);'></div>"
        f"<div class='body'>"
        f"<h4 style='color:{c};'>{METHOD_LABELS[method]}</h4>"
        f"<p class='desc'>{METHOD_DESCRIPTIONS[method]}</p>"
        f"</div></div>"
    )


def _chunk_html(chunk, color: str) -> str:
    return (
        f"<div class='chunk-card'>"
        f"<div style='display:flex; align-items:center;'>"
        f"<span class='rank-badge' style='background:{color};'>#{chunk.rank}</span>"
        f"<strong><code style='font-size:0.82rem;'>{chunk.chunk_id}</code></strong>"
        f"<span style='margin-left:auto;' class='chip'>p.{chunk.page_number}</span>"
        f"</div>"
        f"<div class='text-preview'>{_shorten(chunk.text, 220)}</div>"
        f"<div class='meta'>{chunk.doc_id}</div>"
        f"</div>"
    )


def _legend() -> None:
    items = [
        ("Entity", GRAPH_COLORS["entity"]),
        ("Chunk", GRAPH_COLORS["chunk"]),
        ("Query", GRAPH_COLORS["query"]),
        ("Selected", GRAPH_COLORS["selected"]),
        ("Path", GRAPH_COLORS["path"]),
    ]
    html = "".join(
        f"<span class='legend-item'>"
        f"<span class='legend-dot' style='background:{c};'></span>{lbl}</span>"
        for lbl, c in items
    )
    st.markdown(f"<div class='legend'>{html}</div>", unsafe_allow_html=True)


# ===================================================================
# Step 0 — Query Selection
# ===================================================================


def render_query_step(*, examples: list, benchmarks: dict) -> None:
    _hero(0, "Query Selection", "Choose a benchmark question or type your own to begin the pipeline.")

    opts = {_qlabel(q): q for q in examples}
    labels = list(opts)
    if st.session_state["selected_example_id"] is None and labels:
        st.session_state["selected_example_id"] = labels[0]

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        sel = st.selectbox(
            "Benchmark question",
            labels,
            index=(
                labels.index(st.session_state["selected_example_id"])
                if st.session_state["selected_example_id"] in labels
                else 0
            ),
            key="qs_select",
        )
        st.session_state["selected_example_id"] = sel
        eq = opts[sel]

        custom = st.text_input(
            "Or type a custom question",
            value=st.session_state["custom_query_text"],
            placeholder="e.g. What policies improve supply chain resilience?",
            key="qs_custom",
        )
        st.session_state["custom_query_text"] = custom
        final = custom.strip() or eq.question

        st.markdown("---")
        bench = benchmarks.get(normalize_text(final))
        st.markdown(
            f"<div class='card' style='border-left:4px solid #8B5CF6;'>"
            f"<p style='font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; "
            f"color:#8B5CF6; font-weight:600; margin:0 0 6px 0;'>Ready to run</p>"
            f"<p style='font-size:1.05rem; font-weight:600; color:#0F172A; margin:0;'>{final}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if bench:
            st.caption(f"Benchmark: `{bench['question_id']}` ({bench['category']})")
        else:
            st.caption("Custom query — no benchmark scores available.")

    with right:
        st.markdown(
            "<div class='card'>"
            "<p style='font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; "
            "color:#94A3B8; font-weight:600; margin:0 0 12px 0;'>Configuration</p>"
            f"<p><strong>Chunks:</strong> {st.session_state['input_top_k']}</p>"
            f"<p><strong>Animation:</strong> {'on' if st.session_state['animation_enabled'] else 'off'}</p>"
            f"<p><strong>Speed:</strong> {st.session_state['animation_delay']:.2f}s</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Question metadata", expanded=True, key="qs_meta"):
            st.markdown(f"**ID:** `{eq.question_id}`")
            st.markdown(f"**Category:** `{eq.category}`")
            st.markdown(f"**Notes:** {eq.notes or 'n/a'}")

    # Nav
    _, c = st.columns([3, 1])
    with c:
        if st.button(
            "Start Demo",
            type="primary",
            use_container_width=True,
            key="qs_start",
        ):
            if not final.strip():
                st.error("Please enter a question first.")
            else:
                _activate(final, st.session_state["input_top_k"])
                st.session_state["step"] = 1
                st.rerun()


# ===================================================================
# Step 1 — Retrieval
# ===================================================================


def render_retrieval_step(*, config) -> None:
    _hero(1, "Retrieval", "All four methods retrieve evidence for the same query. Explore what each finds.")

    runs = _ensure_runs(config)
    sig = st.session_state["active_signature"]
    animate = _should_animate(f"ret::{sig}")

    tabs = st.tabs([METHOD_LABELS[m] for m in METHOD_ORDER])
    for tab, method in zip(tabs, METHOD_ORDER, strict=False):
        with tab:
            st.markdown(_method_card_open(method), unsafe_allow_html=True)
            run = runs[method]
            color = METHOD_COLORS[method]
            chunks = run.retrieved_chunks
            limit = min(3, len(chunks))

            if animate:
                ph = st.empty()
                for n in range(1, limit + 1):
                    with ph.container():
                        for c in chunks[:n]:
                            st.markdown(_chunk_html(c, color), unsafe_allow_html=True)
                    _sleep()
                _mark_animated(f"ret::{sig}")
            else:
                for c in chunks[:limit]:
                    st.markdown(_chunk_html(c, color), unsafe_allow_html=True)

            if len(chunks) > 3:
                with st.expander(f"All {len(chunks)} chunks", key=f"ret_all_{method}"):
                    for c in chunks:
                        st.markdown(_chunk_html(c, color), unsafe_allow_html=True)


# ===================================================================
# Step 2 — Graph Build
# ===================================================================


def render_graph_build_step(*, config) -> None:
    _hero(2, "Graph Construction", "Entities and relations are extracted from GraphRAG evidence to build an explanation graph.")

    payload = _ensure_graph(config)
    G = payload["graph"]
    meta = payload["meta"]

    # Metrics
    st.markdown(
        f"<div class='metric-row'>"
        f"<div class='metric-pill'><div class='value'>{G.number_of_nodes()}</div><div class='label'>Nodes</div></div>"
        f"<div class='metric-pill'><div class='value'>{G.number_of_edges()}</div><div class='label'>Edges</div></div>"
        f"<div class='metric-pill'><div class='value'>{len(meta['chunk_nodes'])}</div><div class='label'>Chunks</div></div>"
        f"<div class='metric-pill'><div class='value'>{len(meta['query_nodes'])}</div><div class='label'>Query Entities</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    _legend()

    ph = st.empty()
    sig = st.session_state["active_signature"]
    animate = _should_animate(f"gb::{sig}")

    nodes_all = list(G.nodes())
    edges_all = list(G.edges())
    n_batches = _batched(nodes_all, 6)
    e_batches = _batched(edges_all, 6)
    shown_n: set[str] = set()
    shown_e: set[tuple[str, str]] = set()

    if animate:
        for batch in n_batches:
            shown_n.update(batch)
            ph.plotly_chart(
                _graph_fig(G, meta["positions"], shown_n, shown_e, "Building nodes..."),
                use_container_width=True,
                config={"displayModeBar": False},
                key=_pkey("gbn"),
            )
            _sleep()
        for batch in e_batches:
            shown_e.update(batch)
            ph.plotly_chart(
                _graph_fig(G, meta["positions"], set(G.nodes()), shown_e, "Adding edges..."),
                use_container_width=True,
                config={"displayModeBar": False},
                key=_pkey("gbe"),
            )
            _sleep()
        _mark_animated(f"gb::{sig}")
    else:
        ph.plotly_chart(
            _graph_fig(G, meta["positions"], set(G.nodes()), set(G.edges()), "Complete graph"),
            use_container_width=True,
            config={"displayModeBar": False},
            key=_pkey("gbf"),
        )


# ===================================================================
# Step 3 — Traversal
# ===================================================================


def render_traversal_step(*, config) -> None:
    _hero(3, "Graph Traversal", "Query entities are matched to the graph, then paths are traversed to score evidence.")

    payload = _ensure_graph(config)
    G = payload["graph"]
    meta = payload["meta"]
    graph_run = payload["graph_run"]

    st.markdown(
        "<p style='font-weight:600; margin-bottom:6px;'>Matched query entities</p>"
        + " ".join(f"<span class='chip'>{l}</span>" for l in meta["query_labels"]),
        unsafe_allow_html=True,
    )

    ph = st.empty()
    sig = st.session_state["active_signature"]
    animate = _should_animate(f"gt::{sig}")
    all_n = set(G.nodes())
    all_e = set(G.edges())
    hl = set(meta["query_nodes"])
    sel = {meta["selected_chunk_node"]} if meta["selected_chunk_node"] else set()

    if animate:
        ph.plotly_chart(
            _graph_fig(G, meta["positions"], all_n, all_e, "Query mapped", highlighted_nodes=hl),
            use_container_width=True,
            config={"displayModeBar": False},
            key=_pkey("gtq"),
        )
        _sleep()
        cum: set[tuple[str, str]] = set()
        for edge in meta["selected_edges"]:
            if edge in all_e:
                cum.add(edge)
            ph.plotly_chart(
                _graph_fig(
                    G, meta["positions"], all_n, all_e,
                    "Traversing...",
                    highlighted_nodes=hl,
                    selected_nodes=sel,
                    highlighted_edges=cum,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
                key=_pkey("gtp"),
            )
            _sleep()
        _mark_animated(f"gt::{sig}")
    else:
        ph.plotly_chart(
            _graph_fig(
                G, meta["positions"], all_n, all_e,
                "Traversal complete",
                highlighted_nodes=hl,
                selected_nodes=sel,
                highlighted_edges=set(meta["selected_edges"]),
            ),
            use_container_width=True,
            config={"displayModeBar": False},
            key=_pkey("gtf"),
        )

    _legend()

    if graph_run.retrieved_chunks:
        top = graph_run.retrieved_chunks[0]
        st.markdown("### Top Evidence")
        with st.expander(f"{top.chunk_id} | page {top.page_number}", expanded=True, key="gt_ev"):
            st.caption(f"{top.doc_id} | {top.source_url}")
            st.write(top.text)


# ===================================================================
# Step 4 — Answers
# ===================================================================


def render_answer_step(*, config) -> None:
    _hero(4, "Answer Generation", "Each method's evidence is fed to the LLM to produce a grounded, cited answer.")

    runs = _ensure_runs(config)
    answers, ok = _ensure_answers(config)
    if not ok:
        st.warning("LLM not configured — set OPENROUTER_API_KEY and model in .env")
        return

    sig = st.session_state["active_signature"]
    animate = _should_animate(f"ans::{sig}")

    tabs = st.tabs([METHOD_LABELS[m] for m in METHOD_ORDER])
    for tab, method in zip(tabs, METHOD_ORDER, strict=False):
        with tab:
            run = runs[method]
            rec = answers.get(method, {})
            color = METHOD_COLORS[method]
            st.markdown(_method_card_open(method), unsafe_allow_html=True)

            # Supporting chunks
            chips = " ".join(f"<span class='chip'>{c.chunk_id}</span>" for c in run.retrieved_chunks[:3])
            st.markdown(f"<p style='margin-bottom:8px;'><strong>Sources:</strong> {chips}</p>", unsafe_allow_html=True)

            text = rec.get("answer_text", "Unavailable")
            ph = st.empty()
            if animate:
                _stream(ph, text)
            else:
                with ph.container():
                    st.markdown(text)

            with st.expander("Evidence details", key=f"ans_ev_{method}"):
                for ch in run.retrieved_chunks[:5]:
                    st.markdown(f"**#{ch.rank} `{ch.chunk_id}`** — {_shorten(ch.text, 200)}")

    if animate:
        _mark_animated(f"ans::{sig}")


# ===================================================================
# Step 5 — Comparison
# ===================================================================


def render_comparison_step(*, benchmarks: dict) -> None:
    _hero(5, "Comparison", "All methods side by side. Benchmark scores shown when the query matches the evaluation set.")

    config = load_config()
    runs = _ensure_runs(config)
    answers, ok = _ensure_answers(config)
    bench = benchmarks.get(normalize_text(st.session_state["active_query"]))
    rows = _comparison_rows(runs, answers, bench)

    scored = [r for r in rows if r["score_value"] is not None]
    best = max(scored, key=lambda r: r["score_value"])["key"] if scored else None

    # Visual comparison cards
    for row in rows:
        color = METHOD_COLORS[row["key"]]
        is_best = row["key"] == best
        card_cls = "winner-card" if is_best else "card"

        st.markdown(
            f"<div class='{card_cls}'>"
            f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:8px;'>"
            f"<span style='width:8px; height:8px; border-radius:50%; background:{color}; display:inline-block;'></span>"
            f"<strong style='color:{color}; font-size:1.05rem;'>{row['method']}</strong>"
            f"<span class='chip' style='margin-left:auto;'>Score: {row['score_text']}</span>"
            f"</div>"
            f"<p style='color:#334155; font-size:0.92rem; margin:0;'>{_shorten(row['answer_text'], 180)}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if bench and best:
        bscore = next(r["score_value"] for r in rows if r["key"] == best)
        st.success(f"Best: **{METHOD_LABELS[best]}** — {_fmt(bscore)}")
    elif ok:
        st.info("Custom query — no benchmark scores.")

    # Full answer tabs
    st.markdown("### Full Answers")
    atabs = st.tabs([r["method"] for r in rows])
    for tab, row in zip(atabs, rows, strict=False):
        with tab:
            st.markdown(row["answer_text"])
            if row["metrics"]:
                with st.expander("Metrics", key=f"cmp_{row['key']}"):
                    st.json(row["metrics"])

    _save_run(runs, answers, rows, bench)


# ===================================================================
# Backend (unchanged logic)
# ===================================================================


def _activate(query: str, top_k: int) -> None:
    sig = f"{normalize_text(query)}::{top_k}"
    old = st.session_state.get("active_signature")
    st.session_state.update(
        active_query=query,
        active_top_k=top_k,
        active_signature=sig,
    )
    if sig != old:
        for k in [
            "retrieval_runs", "retrieval_signature",
            "graph_payload", "graph_signature",
            "answers", "answers_signature",
            "saved_demo_signature",
        ]:
            st.session_state[k] = None
        st.session_state["animated_flags"] = {}


@st.cache_data(show_spinner=False)
def _load_examples(path: Path | str) -> list:
    return load_questions(path)


@st.cache_data(show_spinner=False)
def _load_benchmarks() -> dict[str, dict]:
    files = sorted(ROOT_DIR.glob("results/answer_evaluation/answer_eval_*/per_question_results.jsonl"))
    if not files:
        return {}
    lookup: dict[str, dict] = {}
    for line in files[-1].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lookup[normalize_text(r["question"])] = r
    return lookup


def _qlabel(q) -> str:
    return f"[{q.question_id}] {q.question}"


def _specs(config) -> list[dict]:
    idx = config.paths.indexes_dir
    return [
        {"label": "tfidf", "runner": run_baseline_query, "kwargs": {"index_dir": idx / "baseline_tfidf"}},
        {"label": "dense", "runner": run_dense_query, "kwargs": {"index_dir": idx / "baseline_dense"}},
        {"label": "graph", "runner": run_graph_query, "kwargs": {"index_dir": idx / "baseline_graph"}},
        {"label": "hybrid_graph", "runner": run_hybrid_graph_query, "kwargs": {"dense_index_dir": idx / "baseline_dense", "graph_index_dir": idx / "baseline_graph"}},
    ]


def _ensure_runs(config) -> dict:
    sig = st.session_state["active_signature"]
    if st.session_state.get("retrieval_signature") == sig and st.session_state.get("retrieval_runs"):
        return st.session_state["retrieval_runs"]
    q = st.session_state["active_query"]
    k = st.session_state["active_top_k"]
    runs = {}
    with st.spinner("Running retrieval..."):
        for s in _specs(config):
            runs[s["label"]] = s["runner"](
                query_text=q, output_dir=config.paths.runs_dir, top_k=k,
                llm_config=config.llm, generate_answer=False,
                project_root=config.paths.root_dir, **s["kwargs"],
            )
    st.session_state["retrieval_runs"] = runs
    st.session_state["retrieval_signature"] = sig
    return runs


def _ensure_graph(config) -> dict:
    sig = st.session_state["active_signature"]
    if st.session_state.get("graph_signature") == sig and st.session_state.get("graph_payload"):
        return st.session_state["graph_payload"]
    runs = _ensure_runs(config)
    G, meta = _build_graph(runs["graph"])
    payload = {"graph": G, "meta": meta, "graph_run": runs["graph"]}
    st.session_state["graph_payload"] = payload
    st.session_state["graph_signature"] = sig
    return payload


def _ensure_answers(config) -> tuple[dict, bool]:
    sig = st.session_state["active_signature"]
    if st.session_state.get("answers_signature") == sig and st.session_state.get("answers") is not None:
        return st.session_state["answers"], True
    try:
        model = require_generation_config(config.llm)
    except LLMClientError:
        st.session_state["answers"] = {}
        st.session_state["answers_signature"] = sig
        return {}, False
    runs = _ensure_runs(config)
    answers = {}
    with st.spinner("Generating answers..."):
        for m in METHOD_ORDER:
            prompt = build_grounded_answer_prompt(
                question=st.session_state["active_query"],
                retrieved_chunks=runs[m].retrieved_chunks,
                method_label=m,
            )
            try:
                comp = call_chat_completion(
                    llm_config=config.llm, model=model,
                    messages=prompt_to_messages(prompt),
                    temperature=0.0, max_tokens=700,
                )
                answers[m] = {
                    "answer_text": comp.content, "model": model,
                    "latency_seconds": comp.latency_seconds, "usage": comp.usage,
                    "prompt_template_version": prompt.template_version,
                }
            except LLMClientError as e:
                answers[m] = {
                    "answer_text": f"Failed: {e}", "error": str(e),
                    "model": model, "prompt_template_version": prompt.template_version,
                }
    st.session_state["answers"] = answers
    st.session_state["answers_signature"] = sig
    return answers, True


# ===================================================================
# Graph helpers
# ===================================================================


def _build_graph(graph_run) -> tuple[nx.DiGraph, dict]:
    G = nx.DiGraph()
    chunk_nodes, selected_edges = [], []
    for ch in graph_run.retrieved_chunks:
        cn = f"chunk::{ch.chunk_id}"
        G.add_node(cn, label=ch.chunk_id, node_type="chunk")
        chunk_nodes.append(cn)
        for e in ch.contributing_entities[:6]:
            G.add_node(e.entity_id, label=e.canonical_name, node_type="entity")
            G.add_edge(e.entity_id, cn, relation="supports")
        for r in ch.contributing_relations[:8]:
            G.add_node(r.source_entity_id, label=r.source_name, node_type="entity")
            G.add_node(r.target_entity_id, label=r.target_name, node_type="entity")
            G.add_edge(r.source_entity_id, r.target_entity_id, relation=r.relation_type.lower())

    query_nodes, query_labels = [], []
    for qe in graph_run.query_entities:
        query_labels.append(qe.canonical_name)
        if not G.has_node(qe.entity_id):
            G.add_node(qe.entity_id, label=qe.canonical_name, node_type="entity")
        query_nodes.append(qe.entity_id)

    pos = nx.spring_layout(G, seed=7, k=1.25, iterations=250)
    sel_chunk = None
    if graph_run.retrieved_chunks:
        top = graph_run.retrieved_chunks[0]
        sel_chunk = f"chunk::{top.chunk_id}"
        for r in top.contributing_relations[:6]:
            selected_edges.append((r.source_entity_id, r.target_entity_id))
            for eid in (r.source_entity_id, r.target_entity_id):
                if G.has_node(eid) and sel_chunk in G:
                    selected_edges.append((eid, sel_chunk))

    return G, {
        "positions": pos, "query_nodes": query_nodes, "query_labels": query_labels,
        "selected_edges": selected_edges, "selected_chunk_node": sel_chunk, "chunk_nodes": chunk_nodes,
    }


def _graph_fig(
    G: nx.DiGraph,
    pos: dict,
    vis_n: set[str],
    vis_e: set[tuple[str, str]],
    title: str = "",
    *,
    highlighted_nodes: set[str] | None = None,
    selected_nodes: set[str] | None = None,
    highlighted_edges: set[tuple[str, str]] | None = None,
) -> go.Figure:
    hl_n = highlighted_nodes or set()
    sel_n = selected_nodes or set()
    hl_e = highlighted_edges or set()
    fig = go.Figure()

    for s, t in G.edges():
        if s not in vis_n or t not in vis_n or (s, t) not in vis_e:
            continue
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        is_hl = (s, t) in hl_e
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line={"color": GRAPH_COLORS["path"] if is_hl else GRAPH_COLORS["edge"], "width": 4 if is_hl else 1.5},
            hoverinfo="skip", showlegend=False,
        ))

    nx_, ny_, labels, colors, sizes = [], [], [], [], []
    for n in vis_n:
        x, y = pos[n]
        nx_.append(x)
        ny_.append(y)
        labels.append(G.nodes[n].get("label", n))
        nt = G.nodes[n].get("node_type", "entity")
        if n in sel_n:
            colors.append(GRAPH_COLORS["selected"]); sizes.append(38)
        elif n in hl_n:
            colors.append(GRAPH_COLORS["query"]); sizes.append(36)
        elif nt == "chunk":
            colors.append(GRAPH_COLORS["chunk"]); sizes.append(28)
        else:
            colors.append(GRAPH_COLORS["entity"]); sizes.append(22)

    fig.add_trace(go.Scatter(
        x=nx_, y=ny_, mode="markers+text", text=labels, textposition="top center",
        textfont={"size": 9, "color": "#475569"},
        marker={"size": sizes, "color": colors, "line": {"width": 2, "color": "#FFFFFF"}},
        hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(
        title={"text": title, "font": {"size": 13, "color": "#94A3B8", "family": "Inter"}},
        margin={"l": 10, "r": 10, "t": 44, "b": 10},
        paper_bgcolor="#FAFBFF", plot_bgcolor="#FAFBFF",
        xaxis={"visible": False}, yaxis={"visible": False},
        height=750,
    )
    return fig


# ===================================================================
# Comparison helpers
# ===================================================================


def _comparison_rows(runs, answers, bench) -> list[dict]:
    rows = []
    for m in METHOD_ORDER:
        score, metrics = None, {}
        if bench:
            for mr in bench["method_results"]:
                if mr["method_label"] == m:
                    score = mr["overall_score"]
                    metrics = {k: mr[k] for k in ("correctness", "completeness", "reasoning", "groundedness")}
                    break
        rows.append({
            "method": METHOD_LABELS[m], "key": m,
            "score_value": score, "score_text": _fmt(score),
            "answer_text": answers.get(m, {}).get("answer_text", "Unavailable"),
            "metrics": metrics,
            "top_chunk": runs[m].retrieved_chunks[0].chunk_id if runs[m].retrieved_chunks else "n/a",
        })
    return rows


def _save_run(runs, answers, rows, bench) -> None:
    sig = st.session_state["active_signature"]
    if st.session_state.get("saved_demo_signature") == sig:
        return
    DEMO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(st.session_state["active_query"])[:50] or "q"
    d = DEMO_RUNS_DIR / f"demo_{ts}_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query": st.session_state["active_query"],
        "top_k": st.session_state["active_top_k"],
        "retrieval_run_ids": {m: runs[m].run_id for m in METHOD_ORDER},
        "answers": answers, "comparison_rows": rows, "benchmark_entry": bench,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    st.session_state["saved_demo_signature"] = sig


# ===================================================================
# Utilities
# ===================================================================


def _should_animate(flag: str) -> bool:
    return st.session_state["animation_enabled"] and not st.session_state["animated_flags"].get(flag, False)


def _mark_animated(flag: str) -> None:
    flags = dict(st.session_state["animated_flags"])
    flags[flag] = True
    st.session_state["animated_flags"] = flags


def _sleep(mult: float = 1.0) -> None:
    if st.session_state["animation_enabled"]:
        d = st.session_state["animation_delay"] * mult
        if d > 0:
            time.sleep(d)


def _stream(ph, text: str) -> None:
    parts = text.split(". ")
    if len(parts) <= 1:
        words = text.split()
        step = max(3, len(words) // 12) if words else 1
        for end in range(step, len(words) + step, step):
            with ph.container():
                st.markdown(" ".join(words[:min(end, len(words))]))
            _sleep(0.5)
        return
    vis = []
    for s in parts:
        vis.append(s if s.endswith(".") else f"{s}.")
        with ph.container():
            st.markdown(" ".join(vis))
        _sleep(0.6)


def _shorten(t: str, n: int) -> str:
    c = " ".join(t.split())
    return c if len(c) <= n else c[:n - 3].rstrip() + "..."


def _slugify(t: str) -> str:
    c = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(t))
    while "__" in c:
        c = c.replace("__", "_")
    return c.strip("_")


def _fmt(s: float | None) -> str:
    return "n/a" if s is None else f"{s:.3f}".rstrip("0").rstrip(".")


def _batched(items: list, count: int) -> list[list]:
    if not items:
        return []
    count = max(1, min(count, len(items)))
    sz = math.ceil(len(items) / count)
    return [items[i:i + sz] for i in range(0, len(items), sz)]


# ===================================================================

if __name__ == "__main__":
    main()
