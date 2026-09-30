import json

import pandas as pd
import streamlit as st

import fts_queries as fq
from search_core import (
    INCLUDE_FIELDS,
    RRF_K,
    TICKERS,
    YEARS,
    compare_stats,
    highlight,
    keyword_coverage,
    match_record,
    metadata_filter,
    missing_env,
    rank_badge,
    rank_rows,
    rrf_fuse,
    run_compare,
    run_search_hybrid,
    run_search_semantic,
    run_search_text,
    search,
)

_missing_env = missing_env()


def render_card(r: dict, terms: list[str], rank: int | None = None, badge: str | None = None):
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            prefix = f"**#{rank}** · " if rank else ""
            st.markdown(f"{prefix}**{r['ticker']}** · {r['year']} · {r['filing']} · chunk {r['chunk']}")
            if badge:
                st.caption(badge)
        with col2:
            st.metric("score", f"{r['score']:.4f}", label_visibility="collapsed")

        snippet, full = highlight(r["text"], terms)
        st.markdown(f"<small>{snippet}</small>", unsafe_allow_html=True)
        if len(r["text"]) > 300:
            with st.expander("Full text"):
                st.markdown(f"<small>{full}</small>", unsafe_allow_html=True)


def render_results(matches, terms: list[str] | None = None):
    if not matches:
        st.info("No results found.")
        return

    st.caption(f"{len(matches)} result(s)")
    for m in matches:
        render_card(match_record(m), terms or [])


# ── Query builder state ───────────────────────────────────────────────────────
SCORING_LABELS = {"query_string": "Lucene (query_string)", "text": "BM25 keywords (text)"}
LUCENE_MODE_LABELS = {"clauses": "Clause builder", "raw": "Raw Lucene"}
BLANK_EXAMPLE = "— Start from scratch —"
EXAMPLES_BY_NAME = {ex["name"]: ex for ex in fq.EXAMPLES}
CLAUSE_COLUMNS = ["occur", "kind", "value", "slop", "boost"]
TEXT_FILTER_COLUMNS = ["op", "value", "negate"]


def load_builder_state(state: dict):
    ss = st.session_state
    ss.b_scoring = state["scoring"]
    ss.b_bm25 = state["bm25"]
    ss.b_lucene_mode = state["lucene_mode"]
    ss.b_raw = state["raw"]
    ss.b_clauses_data = [dict(c) for c in state["clauses"]]
    ss.b_tf_data = [dict(f) for f in state["text_filters"]]
    ss.b_tf_join = state["text_filter_join"]
    ss.b_tickers = list(state["tickers"])
    ss.b_excl = list(state["exclude_tickers"])
    ss.b_year_on = state["year_range"] is not None
    ss.b_year = tuple(state["year_range"] or (YEARS[0], YEARS[-1]))
    ss.b_chunk_on = state["chunk_range"] is not None
    ss.b_chunk = tuple(state["chunk_range"] or (0, 1500))
    ss.b_clauses_current = ss.b_clauses_data
    ss.b_tf_current = ss.b_tf_data
    ss.b_ver = ss.get("b_ver", 0) + 1


def on_example_change():
    ex = EXAMPLES_BY_NAME.get(st.session_state.b_example)
    load_builder_state(ex["state"] if ex else fq.BLANK)


def on_lucene_mode_change():
    ss = st.session_state
    if ss.b_lucene_mode == "raw" and not ss.b_raw.strip():
        try:
            ss.b_raw = fq.clauses_to_lucene(ss.get("b_clauses_current", ss.b_clauses_data))
        except fq.QueryError:
            pass


def editor_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for rec in df.to_dict("records"):
        rec = {k: None if pd.isna(v) else v for k, v in rec.items()}
        if str(rec.get("value") or "").strip():
            rows.append(rec)
    return rows


def apply_mode_example(mode: str):
    ss = st.session_state
    ex = next((e for e in fq.MODE_EXAMPLES[mode] if e["name"] == ss[f"ex_{mode}"]), None)
    if not ex:
        return
    for key, value in ex["values"].items():
        ss[key] = value
    if ex.get("builder_example"):
        load_builder_state(EXAMPLES_BY_NAME[ex["builder_example"]]["state"])
        ss.b_example = ex["builder_example"]
    ss.pop("c_results", None)


def example_picker(mode: str):
    examples = fq.MODE_EXAMPLES[mode]
    st.selectbox(
        "Example queries",
        [BLANK_EXAMPLE] + [e["name"] for e in examples],
        key=f"ex_{mode}",
        on_change=apply_mode_example,
        args=(mode,),
    )
    ex = next((e for e in examples if e["name"] == st.session_state[f"ex_{mode}"]), None)
    if ex:
        st.info(ex["description"])


BUILDER_KEYS = [
    "b_example", "b_scoring", "b_bm25", "b_lucene_mode", "b_raw", "b_tf_join",
    "b_tickers", "b_excl", "b_year_on", "b_year", "b_chunk_on", "b_chunk",
]

if "b_ver" not in st.session_state:
    load_builder_state(fq.BLANK)
    st.session_state.b_example = BLANK_EXAMPLE
# Re-assigning keeps state for widgets that are hidden on this run; Streamlit
# otherwise drops it, which would wipe the query when toggling scoring modes.
for _k in BUILDER_KEYS:
    st.session_state[_k] = st.session_state[_k]


def current_builder_state() -> dict:
    ss = st.session_state
    return {
        "scoring": ss.b_scoring,
        "bm25": ss.b_bm25,
        "lucene_mode": ss.b_lucene_mode,
        "clauses": ss.b_clauses_current,
        "raw": ss.b_raw,
        "text_filters": ss.b_tf_current,
        "text_filter_join": ss.b_tf_join,
        "tickers": ss.b_tickers,
        "exclude_tickers": ss.b_excl,
        "year_range": ss.b_year if ss.b_year_on else None,
        "chunk_range": ss.b_chunk if ss.b_chunk_on else None,
    }


def render_builder(top_k: int):
    ss = st.session_state
    ver = ss.b_ver

    st.selectbox(
        "Example queries",
        [BLANK_EXAMPLE] + list(EXAMPLES_BY_NAME),
        key="b_example",
        on_change=on_example_change,
        help="Load a pre-built complex query into the builder, then tweak it.",
    )
    ex = EXAMPLES_BY_NAME.get(ss.b_example)
    if ex:
        st.info(f"**Demonstrates:** {ex['shows']}\n\n{ex['description']}")

    with st.expander("Syntax cheat sheet"):
        st.markdown(
            """
| Goal | Lucene (`query_string` scoring) | Filter operator (hard constraint, no scoring) |
|---|---|---|
| Any of these terms | `text:(cloud revenue)` | `{"text": {"$match_any": "cloud revenue"}}` |
| All terms, any order | `text:(+cloud +revenue)` | `{"text": {"$match_all": "cloud revenue"}}` |
| Exact phrase | `text:("supply chain")` | `{"text": {"$match_phrase": "supply chain"}}` |
| Exclude | `text:(cloud -covid)` / `NOT text:(covid)` | `{"$not": {"text": {"$match_any": "covid"}}}` |
| Boolean nesting | `text:((a OR b) AND c)` | `{"$or": [...]}`, `{"$and": [...]}` |
| Proximity (within N words) | `text:("interest rates increase"~5)` | — scoring only |
| Boost a term | `text:(tariffs^3 trade)` | — scoring only |
| Phrase prefix | `text:("artificial intel"*)` (≥ 2 words) | — scoring only |

- One scoring type per request: BM25 `text` **or** `query_string`, never both.
- Filters run first and shrink the candidate set; scoring ranks what's left.
- The `text` field uses English stemming, so `increase` also matches `increases` / `increased`.
- A Lucene query made only of exclusions (`-covid`) is rejected — include at least one positive clause.
- Phrase-prefix matches all get the same constant score.
"""
        )

    st.subheader("1 · Scoring (ranks results)")
    st.radio(
        "Scoring type",
        list(SCORING_LABELS),
        format_func=SCORING_LABELS.get,
        key="b_scoring",
        horizontal=True,
        label_visibility="collapsed",
    )

    clauses = ss.b_clauses_current
    if ss.b_scoring == "text":
        st.text_input(
            "BM25 keywords",
            key="b_bm25",
            placeholder="e.g. battery cells charging range",
            help="Token-OR BM25: documents matching more / rarer terms score higher.",
        )
    else:
        st.radio(
            "Lucene input",
            list(LUCENE_MODE_LABELS),
            format_func=LUCENE_MODE_LABELS.get,
            key="b_lucene_mode",
            on_change=on_lucene_mode_change,
            horizontal=True,
        )
        if ss.b_lucene_mode == "clauses":
            st.caption(
                "Each row is one clause. **MUST** → `+`, **MUST NOT** → `-`, **SHOULD** → optional "
                "(adds score). Slop applies to phrases; boost multiplies a clause's weight."
            )
            edited = st.data_editor(
                pd.DataFrame(clauses, columns=CLAUSE_COLUMNS),
                key=f"b_clauses_{ver}",
                num_rows="dynamic",
                width="stretch",
                column_config={
                    "occur": st.column_config.SelectboxColumn("Occur", options=fq.OCCURS, default="SHOULD", required=True),
                    "kind": st.column_config.SelectboxColumn("Kind", options=fq.KINDS, default="term(s)", required=True),
                    "value": st.column_config.TextColumn("Value", width="large"),
                    "slop": st.column_config.NumberColumn("Slop (~N)", min_value=0, max_value=50, step=1, default=0),
                    "boost": st.column_config.NumberColumn("Boost (^N)", min_value=0.1, max_value=20.0, step=0.5, default=1.0),
                },
            )
            clauses = editor_rows(edited)
            ss.b_clauses_current = clauses
        else:
            st.text_area(
                "Lucene query",
                key="b_raw",
                height=90,
                placeholder='text:(("supply chain" OR semiconductor) AND shortage) NOT text:(covid)',
                help="Qualify terms with the field name: text:(...). AND / OR / NOT, +required, -excluded, "
                '"phrase"~N, term^N, "two words"*.',
            )

    st.subheader("2 · Filters (hard constraints)")
    st.markdown("**Text-match filters** on the `text` field")
    edited_tf = st.data_editor(
        pd.DataFrame(ss.b_tf_data, columns=TEXT_FILTER_COLUMNS),
        key=f"b_tf_{ver}",
        num_rows="dynamic",
        width="stretch",
        column_config={
            "op": st.column_config.SelectboxColumn("Operator", options=fq.MATCH_OPS, default="$match_phrase", required=True),
            "value": st.column_config.TextColumn("Value", width="large"),
            "negate": st.column_config.CheckboxColumn("NOT", default=False, help="Wrap in $not (exclude matches)"),
        },
    )
    text_filters = editor_rows(edited_tf)
    ss.b_tf_current = text_filters
    if len(text_filters) > 1:
        st.radio(
            "Combine text-match filters with",
            ["all", "any"],
            format_func={"all": "AND ($and) — every condition", "any": "OR ($or) — at least one"}.get,
            key="b_tf_join",
            horizontal=True,
        )

    st.markdown("**Metadata filters**")
    m1, m2 = st.columns(2)
    with m1:
        st.multiselect("Ticker is one of ($in)", TICKERS, key="b_tickers", placeholder="Any")
        st.checkbox("Year range ($gte / $lte)", key="b_year_on")
        if ss.b_year_on:
            st.slider("Years", YEARS[0], YEARS[-1], key="b_year", label_visibility="collapsed")
    with m2:
        st.multiselect("Ticker is not ($nin)", TICKERS, key="b_excl", placeholder="None")
        st.checkbox("Chunk index range", key="b_chunk_on", help="Position of the chunk within the filing.")
        if ss.b_chunk_on:
            st.slider("Chunks", 0, 1500, key="b_chunk", label_visibility="collapsed")

    state = current_builder_state()

    st.subheader("3 · Request")
    try:
        req = fq.build_request(state, top_k, INCLUDE_FIELDS)
    except fq.QueryError as e:
        st.warning(str(e))
        return

    tab_json, tab_py = st.tabs(["JSON", "Python"])
    with tab_json:
        st.code(json.dumps(req, indent=2), language="json")
    with tab_py:
        st.code(fq.to_python(req), language="python")

    if st.button("Search", type="primary", key="b_search"):
        with st.spinner("Searching..."):
            try:
                resp = search(req)
            except Exception as e:
                st.error(f"Search failed: {e}")
                return
        render_results(resp.matches, fq.highlight_terms(state))


# ── Compare: dense vs full-text ───────────────────────────────────────────────
FTS_SOURCES = {"same": "Same query as BM25 keywords", "builder": "Current query-builder query"}


def render_compare_results(res: dict):
    dense, fts, terms = res["dense"], res["fts"], res["terms"]
    stats = compare_stats(res)
    dense_rank, fts_rank, shared, union = stats["dense_rank"], stats["fts_rank"], stats["shared"], stats["union"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overlap", f"{len(shared)} / {stats['k']}",
              f"Jaccard {len(shared) / len(union):.0%}" if union else None, delta_color="off")
    c2.metric("Only dense", len(dense_rank.keys() - shared))
    c3.metric("Only full-text", len(fts_rank.keys() - shared))
    c4.metric("Dense latency", f"{res['embed_ms'] + res['dense_ms']:.0f} ms",
              f"embed {res['embed_ms']:.0f} + search {res['dense_ms']:.0f}", delta_color="off")
    c5.metric("Full-text latency", f"{res['fts_ms']:.0f} ms")

    cov_d, cov_f = stats["cov_dense"], stats["cov_fts"]
    if cov_d is not None:
        st.caption(
            f"**Keyword coverage** — average share of the full-text terms ({', '.join(terms)}) present in each result: "
            f"dense **{cov_d:.0%}** vs full-text **{cov_f:.0%}**. Low dense coverage means it is finding "
            "paraphrases and related concepts the keywords miss; low full-text rank overlap means lexical "
            "matches the embedding doesn't consider close."
        )

    with st.expander("Full-text request sent"):
        st.code(json.dumps(res["fts_req"], indent=2), language="json")

    tab_side, tab_table, tab_rrf = st.tabs(["Side by side", "Rank comparison", "Fused (RRF)"])

    with tab_side:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Dense (semantic)")
            if not dense:
                st.info("No results.")
            for i, r in enumerate(dense, 1):
                hits, n = keyword_coverage(r["text"], terms)
                cov = f" · keywords {hits}/{n}" if n else ""
                render_card(r, terms, rank=i, badge=rank_badge(r["id"], fts_rank, "full-text") + cov)
        with right:
            st.markdown("#### Full-text (BM25 / Lucene)")
            if not fts:
                st.info("No results.")
            for i, r in enumerate(fts, 1):
                hits, n = keyword_coverage(r["text"], terms)
                cov = f" · keywords {hits}/{n}" if n else ""
                render_card(r, terms, rank=i, badge=rank_badge(r["id"], dense_rank, "dense") + cov)

    with tab_table:
        st.dataframe(pd.DataFrame(rank_rows(res, stats)), hide_index=True, width="stretch")
        st.caption("Δ rank < 0 → dense ranks it higher; > 0 → full-text ranks it higher. Blank rank = not in that top-k.")

    with tab_rrf:
        st.caption(
            f"Reciprocal rank fusion merges both lists client-side: score = Σ 1 / ({RRF_K} + rank). "
            "Documents found by both searches rise to the top — a preview of what a two-query hybrid returns."
        )
        for i, (doc_id, score) in enumerate(rrf_fuse(stats), 1):
            r = {**stats["by_id"][doc_id], "score": score}
            d, f = dense_rank.get(doc_id), fts_rank.get(doc_id)
            render_card(r, terms, rank=i, badge=f"dense #{d if d else '—'} · full-text #{f if f else '—'}")


def render_compare(top_k: int, tickers: list, years: list):
    ss = st.session_state
    example_picker("compare")
    query = st.text_input(
        "Query",
        key="c_query",
        placeholder="e.g. risks from rising interest rates on consumer demand",
        help="Embedded for the dense search. Also used as BM25 keywords unless the full-text side uses the query builder.",
    )
    source = st.radio("Full-text side", list(FTS_SOURCES), format_func=FTS_SOURCES.get, key="c_source", horizontal=True)

    dense_extra_filter = None
    if source == "builder":
        fts_state = current_builder_state()
        try:
            preview = fq.build_request(fts_state, top_k, INCLUDE_FIELDS)
        except fq.QueryError as e:
            st.warning(f"Query builder: {e} Set up a query in the Query builder mode first.")
            return
        st.code(json.dumps({k: preview[k] for k in ("score_by", "filter") if k in preview}, indent=2), language="json")
        if preview.get("filter") and st.checkbox(
            "Apply the builder's filters to the dense search too",
            key="c_share_filters",
            help="Makes the dense side a hybrid (dense ranking + the same hard filters), isolating the effect of the ranking signal.",
        ):
            dense_extra_filter = fq.build_filter(fts_state)
    else:
        fts_state = {**fq.BLANK, "scoring": "text", "bm25": query}

    if tickers or years:
        st.caption("Sidebar ticker / year filters apply to both searches.")

    if st.button("Compare", type="primary", disabled=not query.strip(), key="c_run"):
        with st.spinner("Running dense and full-text searches in parallel..."):
            try:
                ss.c_results = run_compare(query, fts_state, metadata_filter(tickers, years), dense_extra_filter, top_k)
            except Exception as e:
                ss.pop("c_results", None)
                st.error(f"Search failed: {e}")

    if ss.get("c_results"):
        render_compare_results(ss.c_results)


# ── Layout ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SEC Search", layout="wide")
st.title("SEC Document Search")

if _missing_env:
    st.error(
        f"Missing environment variable(s): {', '.join(_missing_env)}. "
        "Set them in `.env` locally, or as Secrets in the Hugging Face Space settings."
    )
    st.stop()

mode = st.radio(
    "Search mode",
    [
        "Full-text",
        "Query builder (full-text)",
        "Semantic (dense)",
        "Hybrid (semantic + text filter)",
        "Compare: dense vs full-text",
    ],
    horizontal=True,
)
is_builder = mode == "Query builder (full-text)"

with st.sidebar:
    st.header("Filters")
    if is_builder:
        st.caption("The query builder has its own filters in the main panel.")
        sel_tickers, sel_years = [], []
    else:
        sel_tickers = st.multiselect("Ticker", TICKERS, placeholder="All companies")
        sel_years = st.multiselect("Year", YEARS, placeholder="All years")
    top_k = st.slider("Results", min_value=3, max_value=25, value=10)

st.divider()

# ── Query builder ─────────────────────────────────────────────────────────────
if is_builder:
    render_builder(top_k)

elif mode == "Compare: dense vs full-text":
    render_compare(top_k, sel_tickers, sel_years)

# ── Full-text ─────────────────────────────────────────────────────────────────
elif mode == "Full-text":
    example_picker("fulltext")
    query = st.text_input("Keyword query", key="ft_query", placeholder="e.g. revenue growth operating income")
    if st.button("Search", type="primary", disabled=not query):
        with st.spinner("Searching..."):
            resp = run_search_text(query, sel_tickers, sel_years, top_k)
        render_results(resp.matches, fq.highlight_terms({"scoring": "text", "bm25": query}))

# ── Semantic ──────────────────────────────────────────────────────────────────
elif mode == "Semantic (dense)":
    example_picker("semantic")
    query = st.text_input("Semantic query", key="sem_query", placeholder="e.g. risks related to supply chain disruption")
    if st.button("Search", type="primary", disabled=not query):
        with st.spinner("Embedding & searching..."):
            resp = run_search_semantic(query, sel_tickers, sel_years, top_k)
        render_results(resp.matches)

# ── Hybrid ────────────────────────────────────────────────────────────────────
elif mode == "Hybrid (semantic + text filter)":
    example_picker("hybrid")
    col_a, col_b = st.columns(2)
    with col_a:
        sem_query = st.text_input(
            "Semantic query (drives ranking)",
            key="hy_sem",
            placeholder="e.g. cloud infrastructure investment",
        )
    with col_b:
        txt_filter = st.text_input(
            "Must-contain keywords (full-text filter)",
            key="hy_txt",
            placeholder="e.g. Azure AWS capital expenditure",
            help="All tokens must appear in the document chunk (case-insensitive).",
        )

    if st.button("Search", type="primary", disabled=not sem_query):
        with st.spinner("Embedding & searching..."):
            resp = run_search_hybrid(sem_query, txt_filter, sel_tickers, sel_years, top_k)
        render_results(resp.matches)
