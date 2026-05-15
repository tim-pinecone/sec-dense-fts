import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "sec-fts"
NAMESPACE = "__default__"
EMBED_MODEL = "text-embedding-3-small"

TICKERS = ["aapl", "amzn", "f", "gm", "msft", "orcl"]
YEARS = list(range(2019, 2025))


@st.cache_resource
def get_clients():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    idx = pc.preview.index(name=INDEX_NAME)
    return oai, idx


def _get_oai():
    oai, _ = get_clients()
    return oai


def _get_idx():
    _, idx = get_clients()
    return idx


@st.cache_data(show_spinner=False)
def embed(text: str) -> list[float]:
    resp = _get_oai().embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def build_filter(tickers: list[str], years: list[int]) -> dict | None:
    filt = {}
    if tickers:
        filt["ticker"] = {"$in": tickers}
    if years:
        filt["year"] = {"$in": years}
    return filt or None


def run_search_text(query: str, tickers: list, years: list, top_k: int):
    filt = build_filter(tickers, years)
    return _get_idx().documents.search(
        namespace=NAMESPACE,
        top_k=top_k,
        score_by=[{"type": "text", "field": "text", "query": query}],
        **{"filter": filt} if filt else {},
        include_fields=["text", "ticker", "filing_type", "year", "chunk_index"],
    )


def run_search_semantic(query: str, tickers: list, years: list, top_k: int):
    emb = embed(query)
    filt = build_filter(tickers, years)
    return _get_idx().documents.search(
        namespace=NAMESPACE,
        top_k=top_k,
        score_by=[{"type": "dense_vector", "field": "embedding", "values": emb}],
        **{"filter": filt} if filt else {},
        include_fields=["text", "ticker", "filing_type", "year", "chunk_index"],
    )


def run_search_hybrid(
    semantic_query: str,
    text_filter: str,
    tickers: list,
    years: list,
    top_k: int,
):
    emb = embed(semantic_query)
    filt = build_filter(tickers, years) or {}
    if text_filter.strip():
        filt["text"] = {"$match_all": text_filter.strip()}
    return _get_idx().documents.search(
        namespace=NAMESPACE,
        top_k=top_k,
        score_by=[{"type": "dense_vector", "field": "embedding", "values": emb}],
        **{"filter": filt} if filt else {},
        include_fields=["text", "ticker", "filing_type", "year", "chunk_index"],
    )


def render_results(matches):
    if not matches:
        st.info("No results found.")
        return

    st.caption(f"{len(matches)} result(s)")
    for m in matches:
        d = m.to_dict()
        score = getattr(m, "_score", getattr(m, "score", 0.0))
        ticker = str(d.get("ticker", "?")).upper()
        year = int(d.get("year", 0))
        filing = str(d.get("filing_type", "")).upper()
        chunk = d.get("chunk_index", "?")
        text = str(d.get("text", ""))

        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{ticker}** · {year} · {filing} · chunk {chunk}")
            with col2:
                st.metric("score", f"{score:.4f}", label_visibility="collapsed")

            preview = text[:300] + ("..." if len(text) > 300 else "")
            st.markdown(f"<small>{preview}</small>", unsafe_allow_html=True)
            if len(text) > 300:
                with st.expander("Full text"):
                    st.write(text)


# ── Layout ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SEC Search", layout="wide")
st.title("SEC Document Search")

with st.sidebar:
    st.header("Filters")
    sel_tickers = st.multiselect("Ticker", TICKERS, placeholder="All companies")
    sel_years = st.multiselect("Year", YEARS, placeholder="All years")
    top_k = st.slider("Results", min_value=3, max_value=25, value=10)

mode = st.radio(
    "Search mode",
    ["Full-text", "Semantic (dense)", "Hybrid (semantic + text filter)"],
    horizontal=True,
)

st.divider()

# ── Full-text ─────────────────────────────────────────────────────────────────
if mode == "Full-text":
    query = st.text_input("Keyword query", placeholder="e.g. revenue growth operating income")
    if st.button("Search", type="primary", disabled=not query):
        with st.spinner("Searching..."):
            resp = run_search_text(query, sel_tickers, sel_years, top_k)
        render_results(resp.matches)

# ── Semantic ──────────────────────────────────────────────────────────────────
elif mode == "Semantic (dense)":
    query = st.text_input("Semantic query", placeholder="e.g. risks related to supply chain disruption")
    if st.button("Search", type="primary", disabled=not query):
        with st.spinner("Embedding & searching..."):
            resp = run_search_semantic(query, sel_tickers, sel_years, top_k)
        render_results(resp.matches)

# ── Hybrid ────────────────────────────────────────────────────────────────────
else:
    col_a, col_b = st.columns(2)
    with col_a:
        sem_query = st.text_input(
            "Semantic query (drives ranking)",
            placeholder="e.g. cloud infrastructure investment",
        )
    with col_b:
        txt_filter = st.text_input(
            "Must-contain keywords (full-text filter)",
            placeholder="e.g. Azure AWS capital expenditure",
            help="All tokens must appear in the document chunk (case-insensitive).",
        )

    if st.button("Search", type="primary", disabled=not sem_query):
        with st.spinner("Embedding & searching..."):
            resp = run_search_hybrid(sem_query, txt_filter, sel_tickers, sel_years, top_k)
        render_results(resp.matches)
