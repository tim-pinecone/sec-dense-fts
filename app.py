import html
import json
import os

import gradio as gr
import pandas as pd

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

os.environ["GRADIO_SSR_MODE"] = "False"

try:
    import spaces

    @spaces.GPU
    def zero_gpu_noop():
        return True
except Exception:
    def zero_gpu_noop():
        return False


BLANK = "— Start from scratch —"
N_CLAUSES = 6
N_TEXT_FILTERS = 4
ANY_YEAR = "Any"
BUILDER_EXAMPLES = {ex["name"]: ex for ex in fq.EXAMPLES}

CSS = """
.card {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 12px;
       margin-bottom: 8px; background: var(--block-background-fill);}
.card .hdr {display: flex; justify-content: space-between; gap: 8px; font-size: 14px;}
.card .score {font-variant-numeric: tabular-nums; color: var(--body-text-color-subdued);}
.card .badge {font-size: 12px; color: var(--body-text-color-subdued); margin-top: 2px;}
.card .snip {font-size: 13px; line-height: 1.45; margin-top: 6px;}
.card details {font-size: 13px; margin-top: 4px;}
.card mark, .snip mark {background: rgba(250, 204, 21, .45); color: inherit; padding: 0 1px; border-radius: 2px;}
.count {font-size: 13px; color: var(--body-text-color-subdued); margin: 4px 0 8px;}
.err {border: 1px solid #dc2626; border-radius: 8px; padding: 10px 12px; color: #dc2626;}
.metrics {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin: 4px 0 8px;}
.metric {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 8px 10px;}
.metric .label {font-size: 12px; color: var(--body-text-color-subdued);}
.metric .value {font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums;}
.metric .sub {font-size: 12px; color: var(--body-text-color-subdued);}
.cols {display: grid; grid-template-columns: 1fr 1fr; gap: 12px;}
@media (max-width: 800px) {.cols {grid-template-columns: 1fr;}}
"""

CHEAT_SHEET = """
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


def card_html(r: dict, terms: list[str], rank: int | None = None, badge: str | None = None) -> str:
    snippet, full = highlight(r["text"], terms)
    prefix = f"<b>#{rank}</b> · " if rank else ""
    meta = f"{prefix}<b>{html.escape(r['ticker'])}</b> · {r['year']} · {html.escape(r['filing'])} · chunk {r['chunk']}"
    badge_html = f"<div class='badge'>{html.escape(badge)}</div>" if badge else ""
    more = f"<details><summary>Full text</summary>{full}</details>" if len(r["text"]) > 300 else ""
    return (
        f"<div class='card'><div class='hdr'><span>{meta}</span><span class='score'>{r['score']:.4f}</span></div>"
        f"{badge_html}<div class='snip'>{snippet}</div>{more}</div>"
    )


def results_html(records: list[dict], terms: list[str]) -> str:
    if not records:
        return "<div class='count'>No results found.</div>"
    cards = "".join(card_html(r, terms) for r in records)
    return f"<div class='count'>{len(records)} result(s)</div>{cards}"


def error_html(message: str) -> str:
    return f"<div class='err'>{html.escape(message)}</div>"


def metric_html(label: str, value: str, sub: str = "") -> str:
    sub_html = f"<div class='sub'>{html.escape(sub)}</div>" if sub else ""
    return f"<div class='metric'><div class='label'>{label}</div><div class='value'>{value}</div>{sub_html}</div>"


def mode_example(mode: str, name: str, fields: list[str]):
    ex = next((e for e in fq.MODE_EXAMPLES[mode] if e["name"] == name), None)
    if not ex:
        return [gr.update() for _ in fields] + [""]
    return [ex["values"].get(f, "") for f in fields] + [f"ℹ️ {ex['description']}"]


def do_fulltext(query, tickers, years, top_k):
    if not query.strip():
        return error_html("Enter a keyword query.")
    try:
        resp = run_search_text(query, tickers, years, int(top_k))
    except Exception as e:
        return error_html(f"Search failed: {e}")
    terms = fq.highlight_terms({"scoring": "text", "bm25": query})
    return results_html([match_record(m) for m in resp.matches], terms)


def do_semantic(query, tickers, years, top_k):
    if not query.strip():
        return error_html("Enter a semantic query.")
    try:
        resp = run_search_semantic(query, tickers, years, int(top_k))
    except Exception as e:
        return error_html(f"Search failed: {e}")
    return results_html([match_record(m) for m in resp.matches], [])


def do_hybrid(sem_query, txt_filter, tickers, years, top_k):
    if not sem_query.strip():
        return error_html("Enter a semantic query.")
    try:
        resp = run_search_hybrid(sem_query, txt_filter, tickers, years, int(top_k))
    except Exception as e:
        return error_html(f"Search failed: {e}")
    terms = fq.highlight_terms({"scoring": "text", "bm25": txt_filter})
    return results_html([match_record(m) for m in resp.matches], terms)


def builder_state(values: list) -> dict:
    it = iter(values)
    scoring, bm25, lucene_mode, raw = next(it), next(it), next(it), next(it)
    clauses = []
    for _ in range(N_CLAUSES):
        occur, kind, value, slop, boost = (next(it) for _ in range(5))
        if str(value or "").strip():
            clauses.append({"occur": occur, "kind": kind, "value": value, "slop": slop or 0, "boost": boost or 1.0})
    text_filters = []
    for _ in range(N_TEXT_FILTERS):
        op, value, negate = next(it), next(it), next(it)
        if str(value or "").strip():
            text_filters.append({"op": op, "value": value, "negate": bool(negate)})
    join, tickers, excl, year_from, year_to, chunk_min, chunk_max = (next(it) for _ in range(7))

    year_range = None
    if year_from != ANY_YEAR or year_to != ANY_YEAR:
        lo = int(year_from) if year_from != ANY_YEAR else YEARS[0]
        hi = int(year_to) if year_to != ANY_YEAR else YEARS[-1]
        year_range = (min(lo, hi), max(lo, hi))
    chunk_range = None
    if chunk_min is not None or chunk_max is not None:
        chunk_range = (int(chunk_min or 0), int(chunk_max if chunk_max is not None else 100000))

    return {
        "scoring": scoring,
        "bm25": bm25,
        "lucene_mode": lucene_mode,
        "clauses": clauses,
        "raw": raw,
        "text_filters": text_filters,
        "text_filter_join": join,
        "tickers": tickers or [],
        "exclude_tickers": excl or [],
        "year_range": year_range,
        "chunk_range": chunk_range,
    }


def builder_preview(top_k, *values):
    try:
        req = fq.build_request(builder_state(list(values)), int(top_k), INCLUDE_FIELDS)
    except fq.QueryError as e:
        return "", "", f"⚠️ {e}"
    return json.dumps(req, indent=2), fq.to_python(req), ""


def builder_values_from_state(state: dict) -> list:
    values = [state["scoring"], state["bm25"], state["lucene_mode"], state["raw"]]
    clauses = [c for c in state["clauses"] if str(c.get("value") or "").strip()]
    for i in range(N_CLAUSES):
        c = clauses[i] if i < len(clauses) else {}
        values += [c.get("occur", "SHOULD"), c.get("kind", "term(s)"), c.get("value", ""),
                   c.get("slop", 0), c.get("boost", 1.0)]
    for i in range(N_TEXT_FILTERS):
        f = state["text_filters"][i] if i < len(state["text_filters"]) else {}
        values += [f.get("op", "$match_phrase"), f.get("value", ""), f.get("negate", False)]
    yr = state["year_range"]
    ch = state["chunk_range"]
    values += [
        state["text_filter_join"],
        list(state["tickers"]),
        list(state["exclude_tickers"]),
        str(yr[0]) if yr else ANY_YEAR,
        str(yr[1]) if yr else ANY_YEAR,
        ch[0] if ch else None,
        ch[1] if ch else None,
    ]
    return values


def visibility(scoring: str, lucene_mode: str):
    lucene = scoring == "query_string"
    return (
        gr.update(visible=not lucene),
        gr.update(visible=lucene),
        gr.update(visible=lucene and lucene_mode == "clauses"),
        gr.update(visible=lucene and lucene_mode == "raw"),
    )


def load_builder_example(name: str):
    ex = BUILDER_EXAMPLES.get(name)
    state = ex["state"] if ex else fq.BLANK
    info = f"ℹ️ **Demonstrates:** {ex['shows']}\n\n{ex['description']}" if ex else ""
    return builder_values_from_state(state) + list(visibility(state["scoring"], state["lucene_mode"])) + [info]


def on_lucene_mode(lucene_mode, scoring, raw, *clause_values):
    vis = visibility(scoring, lucene_mode)
    if lucene_mode == "raw" and not str(raw or "").strip():
        rows = [clause_values[i:i + 5] for i in range(0, len(clause_values), 5)]
        clauses = [{"occur": o, "kind": k, "value": v, "slop": s, "boost": b} for o, k, v, s, b in rows]
        try:
            raw = fq.clauses_to_lucene(clauses)
        except fq.QueryError:
            pass
    return (*vis, raw)


def do_builder(top_k, *values):
    state = builder_state(list(values))
    try:
        req = fq.build_request(state, int(top_k), INCLUDE_FIELDS)
        resp = search(req)
    except fq.QueryError as e:
        return error_html(str(e))
    except Exception as e:
        return error_html(f"Search failed: {e}")
    return results_html([match_record(m) for m in resp.matches], fq.highlight_terms(state))


def compare_example(name: str):
    ex = next((e for e in fq.MODE_EXAMPLES["compare"] if e["name"] == name), None)
    n_builder = len(load_builder_example(BLANK))
    if not ex:
        return [gr.update(), gr.update(), ""] + [gr.update()] * n_builder
    builder = load_builder_example(ex["builder_example"]) if ex.get("builder_example") else [gr.update()] * n_builder
    return [ex["values"]["c_query"], ex["values"]["c_source"], f"ℹ️ {ex['description']}"] + builder


def compare_source_preview(source, top_k, *values):
    if source != "builder":
        return gr.update(visible=False), gr.update(visible=False)
    try:
        req = fq.build_request(builder_state(list(values)), int(top_k), INCLUDE_FIELDS)
    except fq.QueryError as e:
        return gr.update(visible=True, value=f"// Query builder: {e}"), gr.update(visible=False)
    shown = {k: req[k] for k in ("score_by", "filter") if k in req}
    return gr.update(visible=True, value=json.dumps(shown, indent=2)), gr.update(visible="filter" in req)


def do_compare(query, source, share_filters, tickers, years, top_k, *values):
    empty = ("", "", pd.DataFrame(), "", "")
    if not str(query or "").strip():
        return (error_html("Enter a query."),) + empty[1:]
    dense_extra = None
    if source == "builder":
        fts_state = builder_state(list(values))
        if share_filters:
            dense_extra = fq.build_filter(fts_state)
    else:
        fts_state = {**fq.BLANK, "scoring": "text", "bm25": query}
    try:
        res = run_compare(query, fts_state, metadata_filter(tickers, years), dense_extra, int(top_k))
    except fq.QueryError as e:
        return (error_html(f"Query builder: {e}"),) + empty[1:]
    except Exception as e:
        return (error_html(f"Search failed: {e}"),) + empty[1:]

    stats = compare_stats(res)
    dense, fts, terms = res["dense"], res["fts"], res["terms"]
    dense_rank, fts_rank, shared, union = stats["dense_rank"], stats["fts_rank"], stats["shared"], stats["union"]

    metrics = "".join([
        metric_html("Overlap", f"{len(shared)} / {stats['k']}", f"Jaccard {len(shared) / len(union):.0%}" if union else ""),
        metric_html("Only dense", str(len(dense_rank.keys() - shared))),
        metric_html("Only full-text", str(len(fts_rank.keys() - shared))),
        metric_html("Dense latency", f"{res['embed_ms'] + res['dense_ms']:.0f} ms",
                    f"embed {res['embed_ms']:.0f} + search {res['dense_ms']:.0f}"),
        metric_html("Full-text latency", f"{res['fts_ms']:.0f} ms"),
    ])
    summary = f"<div class='metrics'>{metrics}</div>"
    if stats["cov_dense"] is not None:
        summary += (
            f"<div class='count'><b>Keyword coverage</b> — average share of the full-text terms "
            f"({html.escape(', '.join(terms))}) present in each result: dense <b>{stats['cov_dense']:.0%}</b> vs "
            f"full-text <b>{stats['cov_fts']:.0%}</b>. Low dense coverage means it is finding paraphrases and related "
            "concepts the keywords miss.</div>"
        )

    def side(rows, other_ranks, other_name, title):
        cards = []
        for i, r in enumerate(rows, 1):
            hits, n = keyword_coverage(r["text"], terms)
            cov = f" · keywords {hits}/{n}" if n else ""
            cards.append(card_html(r, terms, rank=i, badge=rank_badge(r["id"], other_ranks, other_name) + cov))
        body = "".join(cards) or "<div class='count'>No results.</div>"
        return f"<div><h4>{title}</h4>{body}</div>"

    side_by_side = (
        "<div class='cols'>"
        + side(dense, fts_rank, "full-text", "Dense (semantic)")
        + side(fts, dense_rank, "dense", "Full-text (BM25 / Lucene)")
        + "</div>"
    )

    fused_cards = []
    for i, (doc_id, score) in enumerate(rrf_fuse(stats), 1):
        r = {**stats["by_id"][doc_id], "score": score}
        d, f = dense_rank.get(doc_id), fts_rank.get(doc_id)
        fused_cards.append(card_html(r, terms, rank=i, badge=f"dense #{d if d else '—'} · full-text #{f if f else '—'}"))
    fused = (
        f"<div class='count'>Reciprocal rank fusion merges both lists client-side: score = Σ 1 / ({RRF_K} + rank). "
        "Documents found by both searches rise to the top — a preview of what a two-query hybrid returns.</div>"
        + "".join(fused_cards)
    )

    return summary, side_by_side, pd.DataFrame(rank_rows(res, stats)), fused, json.dumps(res["fts_req"], indent=2)


with gr.Blocks(title="SEC FTS Hybrid Comparison") as app:
    gr.Markdown(
        "# SEC 10-K Search — Pinecone full-text search vs dense vectors\n"
        "Six companies (AAPL, AMZN, F, GM, MSFT, ORCL) · 10-K filings 2019–2024 · ~24k chunks. "
        "Full-text search uses Pinecone's document-schema BM25 / Lucene; dense search uses OpenAI "
        "`text-embedding-3-small`."
    )
    if missing_env():
        gr.Markdown(
            f"⚠️ **Missing environment variable(s): {', '.join(missing_env())}.** "
            "Set them in `.env` locally, or as Secrets in the Space settings."
        )

    with gr.Sidebar():
        gr.Markdown("### Filters")
        side_tickers = gr.CheckboxGroup(TICKERS, label="Ticker", info="Empty = all companies")
        side_years = gr.CheckboxGroup([str(y) for y in YEARS], label="Year", info="Empty = all years")
        top_k = gr.Slider(3, 25, value=10, step=1, label="Results")
        gr.Markdown("_The Query builder tab has its own filters; these apply to the other tabs._")

    with gr.Tabs():
        with gr.Tab("Full-text"):
            ft_ex = gr.Dropdown([BLANK] + [e["name"] for e in fq.MODE_EXAMPLES["fulltext"]], value=BLANK, label="Example queries")
            ft_info = gr.Markdown()
            ft_query = gr.Textbox(label="Keyword query", placeholder="e.g. revenue growth operating income")
            ft_btn = gr.Button("Search", variant="primary")
            ft_out = gr.HTML()

        with gr.Tab("Query builder (full-text)"):
            b_ex = gr.Dropdown([BLANK] + list(BUILDER_EXAMPLES), value=BLANK, label="Example queries",
                               info="Load a pre-built complex query into the builder, then tweak it.")
            b_info = gr.Markdown()
            with gr.Accordion("Syntax cheat sheet", open=False):
                gr.Markdown(CHEAT_SHEET)

            gr.Markdown("### 1 · Scoring (ranks results)")
            b_scoring = gr.Radio([("Lucene (query_string)", "query_string"), ("BM25 keywords (text)", "text")],
                                 value="query_string", show_label=False)
            b_bm25 = gr.Textbox(label="BM25 keywords", placeholder="e.g. battery cells charging range", visible=False)
            b_lucene_mode = gr.Radio([("Clause builder", "clauses"), ("Raw Lucene", "raw")], value="clauses", label="Lucene input")
            with gr.Group() as b_clause_group:
                gr.Markdown("Each row is one clause. **MUST** → `+`, **MUST NOT** → `-`, **SHOULD** → optional (adds score). "
                            "Slop applies to phrases; boost multiplies a clause's weight. Empty rows are ignored.")
                clause_comps = []
                for i in range(N_CLAUSES):
                    with gr.Row(equal_height=True):
                        clause_comps += [
                            gr.Dropdown(fq.OCCURS, value="SHOULD", label="Occur", scale=1, min_width=110),
                            gr.Dropdown(fq.KINDS, value="term(s)", label="Kind", scale=1, min_width=120),
                            gr.Textbox(label="Value", scale=3, min_width=160),
                            gr.Number(value=0, label="Slop ~N", precision=0, minimum=0, maximum=50, scale=1, min_width=80),
                            gr.Number(value=1.0, label="Boost ^N", minimum=0.1, maximum=20, step=0.5, scale=1, min_width=80),
                        ]
            b_raw = gr.Textbox(label="Lucene query", lines=3, visible=False,
                               placeholder='text:(("supply chain" OR semiconductor) AND shortage) NOT text:(covid)')

            gr.Markdown("### 2 · Filters (hard constraints)")
            gr.Markdown("**Text-match filters** on the `text` field. Empty rows are ignored.")
            tf_comps = []
            for i in range(N_TEXT_FILTERS):
                with gr.Row(equal_height=True):
                    tf_comps += [
                        gr.Dropdown(fq.MATCH_OPS, value="$match_phrase", label="Operator", scale=1, min_width=140),
                        gr.Textbox(label="Value", scale=3, min_width=160),
                        gr.Checkbox(label="NOT", value=False, scale=0, min_width=70),
                    ]
            b_join = gr.Radio([("AND ($and) — every condition", "all"), ("OR ($or) — at least one", "any")],
                              value="all", label="Combine text-match filters with")
            gr.Markdown("**Metadata filters**")
            with gr.Row():
                b_tickers = gr.CheckboxGroup(TICKERS, label="Ticker is one of ($in)")
                b_excl = gr.CheckboxGroup(TICKERS, label="Ticker is not ($nin)")
            with gr.Row():
                year_choices = [ANY_YEAR] + [str(y) for y in YEARS]
                b_year_from = gr.Dropdown(year_choices, value=ANY_YEAR, label="Year from ($gte)")
                b_year_to = gr.Dropdown(year_choices, value=ANY_YEAR, label="Year to ($lte)")
                b_chunk_min = gr.Number(value=None, label="Chunk index ≥", precision=0, minimum=0)
                b_chunk_max = gr.Number(value=None, label="Chunk index ≤", precision=0, minimum=0)

            gr.Markdown("### 3 · Request")
            b_warn = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("JSON"):
                    b_json = gr.Code(language="json", interactive=False, show_label=False)
                with gr.Tab("Python"):
                    b_py = gr.Code(language="python", interactive=False, show_label=False)
            b_btn = gr.Button("Search", variant="primary")
            b_out = gr.HTML()

        with gr.Tab("Semantic (dense)"):
            sem_ex = gr.Dropdown([BLANK] + [e["name"] for e in fq.MODE_EXAMPLES["semantic"]], value=BLANK, label="Example queries")
            sem_info = gr.Markdown()
            sem_query = gr.Textbox(label="Semantic query", placeholder="e.g. risks related to supply chain disruption")
            sem_btn = gr.Button("Search", variant="primary")
            sem_out = gr.HTML()

        with gr.Tab("Hybrid (semantic + text filter)"):
            hy_ex = gr.Dropdown([BLANK] + [e["name"] for e in fq.MODE_EXAMPLES["hybrid"]], value=BLANK, label="Example queries")
            hy_info = gr.Markdown()
            with gr.Row():
                hy_sem = gr.Textbox(label="Semantic query (drives ranking)", placeholder="e.g. cloud infrastructure investment")
                hy_txt = gr.Textbox(label="Must-contain keywords (full-text filter)", placeholder="e.g. Azure AWS capital expenditure",
                                    info="All tokens must appear in the chunk ($match_all).")
            hy_btn = gr.Button("Search", variant="primary")
            hy_out = gr.HTML()

        with gr.Tab("Compare: dense vs full-text"):
            c_ex = gr.Dropdown([BLANK] + [e["name"] for e in fq.MODE_EXAMPLES["compare"]], value=BLANK, label="Example queries")
            c_info = gr.Markdown()
            c_query = gr.Textbox(label="Query", placeholder="e.g. risks from rising interest rates on consumer demand",
                                 info="Embedded for the dense search. Also used as BM25 keywords unless the full-text side uses the query builder.")
            c_source = gr.Radio([("Same query as BM25 keywords", "same"), ("Current query-builder query", "builder")],
                                value="same", label="Full-text side")
            c_builder_preview = gr.Code(language="json", interactive=False, label="Query-builder request (full-text side)", visible=False)
            c_share = gr.Checkbox(label="Apply the builder's filters to the dense search too", value=False, visible=False,
                                  info="Makes the dense side a hybrid, isolating the effect of the ranking signal.")
            gr.Markdown("_Sidebar ticker / year filters apply to both searches._")
            c_btn = gr.Button("Compare", variant="primary")
            c_summary = gr.HTML()
            with gr.Tabs():
                with gr.Tab("Side by side"):
                    c_side = gr.HTML()
                with gr.Tab("Rank comparison"):
                    c_table = gr.Dataframe(interactive=False, wrap=True)
                    gr.Markdown("_Δ rank < 0 → dense ranks it higher; > 0 → full-text ranks it higher. Blank rank = not in that top-k._")
                with gr.Tab("Fused (RRF)"):
                    c_rrf = gr.HTML()
            with gr.Accordion("Full-text request sent", open=False):
                c_req = gr.Code(language="json", interactive=False, show_label=False)

    side_inputs = [side_tickers, side_years, top_k]
    builder_inputs = [b_scoring, b_bm25, b_lucene_mode, b_raw, *clause_comps, *tf_comps,
                      b_join, b_tickers, b_excl, b_year_from, b_year_to, b_chunk_min, b_chunk_max]
    builder_vis = [b_bm25, b_lucene_mode, b_clause_group, b_raw]

    ft_ex.change(lambda n: mode_example("fulltext", n, ["ft_query"]), ft_ex, [ft_query, ft_info])
    sem_ex.change(lambda n: mode_example("semantic", n, ["sem_query"]), sem_ex, [sem_query, sem_info])
    hy_ex.change(lambda n: mode_example("hybrid", n, ["hy_sem", "hy_txt"]), hy_ex, [hy_sem, hy_txt, hy_info])

    gr.on([ft_btn.click, ft_query.submit], do_fulltext, [ft_query, *side_inputs], ft_out)
    gr.on([sem_btn.click, sem_query.submit], do_semantic, [sem_query, *side_inputs], sem_out)
    gr.on([hy_btn.click, hy_sem.submit, hy_txt.submit], do_hybrid, [hy_sem, hy_txt, *side_inputs], hy_out)

    b_ex.change(load_builder_example, b_ex, builder_inputs + builder_vis + [b_info])
    b_scoring.change(visibility, [b_scoring, b_lucene_mode], builder_vis)
    b_lucene_mode.change(on_lucene_mode, [b_lucene_mode, b_scoring, b_raw, *clause_comps], builder_vis + [b_raw])
    gr.on([c.change for c in builder_inputs] + [top_k.change, app.load], builder_preview,
          [top_k, *builder_inputs], [b_json, b_py, b_warn])
    b_btn.click(do_builder, [top_k, *builder_inputs], b_out)

    c_ex.change(compare_example, c_ex, [c_query, c_source, c_info] + builder_inputs + builder_vis + [b_info])
    gr.on([c_source.change] + [c.change for c in builder_inputs], compare_source_preview,
          [c_source, top_k, *builder_inputs], [c_builder_preview, c_share])
    gr.on([c_btn.click, c_query.submit], do_compare, [c_query, c_source, c_share, *side_inputs, *builder_inputs],
          [c_summary, c_side, c_table, c_rrf, c_req])

demo = app

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, css=CSS, theme=gr.themes.Soft())
