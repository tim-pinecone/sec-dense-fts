import html
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

import fts_queries as fq

load_dotenv()

INDEX_NAME = "sec-fts"
NAMESPACE = "__default__"
EMBED_MODEL = "text-embedding-3-small"

TICKERS = ["aapl", "amzn", "f", "gm", "msft", "orcl"]
YEARS = list(range(2019, 2025))
INCLUDE_FIELDS = ["text", "ticker", "filing_type", "year", "chunk_index"]
RRF_K = 60


def missing_env() -> list[str]:
    return [k for k in ("PINECONE_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]


@lru_cache(maxsize=1)
def get_index():
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"]).index(name=INDEX_NAME)


@lru_cache(maxsize=1)
def get_openai() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@lru_cache(maxsize=512)
def _embed(text: str) -> tuple[float, ...]:
    resp = get_openai().embeddings.create(model=EMBED_MODEL, input=[text])
    return tuple(resp.data[0].embedding)


def embed(text: str) -> list[float]:
    return list(_embed(text))


def metadata_filter(tickers: list[str], years: list[int]) -> dict | None:
    filt = {}
    if tickers:
        filt["ticker"] = {"$in": list(tickers)}
    if years:
        filt["year"] = {"$in": [int(y) for y in years]}
    return filt or None


def and_filters(*filters: dict | None) -> dict | None:
    parts = [f for f in filters if f]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


def search(req: dict):
    return get_index().documents.search(**req)


def dense_request(query: str, top_k: int, filt: dict | None = None) -> dict:
    req = {
        "namespace": NAMESPACE,
        "top_k": top_k,
        "score_by": [{"type": "dense_vector", "field": "embedding", "values": embed(query)}],
        "include_fields": INCLUDE_FIELDS,
    }
    if filt:
        req["filter"] = filt
    return req


def run_search_text(query: str, tickers: list, years: list, top_k: int):
    req = {
        "namespace": NAMESPACE,
        "top_k": top_k,
        "score_by": [{"type": "text", "field": "text", "query": query}],
        "include_fields": INCLUDE_FIELDS,
    }
    filt = metadata_filter(tickers, years)
    if filt:
        req["filter"] = filt
    return search(req)


def run_search_semantic(query: str, tickers: list, years: list, top_k: int):
    return search(dense_request(query, top_k, metadata_filter(tickers, years)))


def run_search_hybrid(semantic_query: str, text_filter: str, tickers: list, years: list, top_k: int):
    filt = metadata_filter(tickers, years) or {}
    if text_filter.strip():
        filt["text"] = {"$match_all": text_filter.strip()}
    return search(dense_request(semantic_query, top_k, filt or None))


def term_pattern(terms: list[str]) -> re.Pattern | None:
    if not terms:
        return None
    stems = [re.escape(t[:-2] if len(t) > 5 else t) for t in terms]
    return re.compile(r"\b(" + "|".join(stems) + r")\w*", re.IGNORECASE)


def highlight(text: str, terms: list[str]) -> tuple[str, str]:
    escaped = html.escape(text)
    pattern = term_pattern(terms)
    if not pattern:
        return escaped[:300] + ("..." if len(escaped) > 300 else ""), escaped
    marked = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)

    first = pattern.search(escaped)
    start = max(0, first.start() - 150) if first else 0
    snippet_raw = escaped[start : start + 450]
    snippet = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet_raw)
    snippet = ("..." if start else "") + snippet + ("..." if start + 450 < len(escaped) else "")
    return snippet, marked


def keyword_coverage(text: str, terms: list[str]) -> tuple[int, int]:
    hits = sum(1 for t in terms if term_pattern([t]).search(text))
    return hits, len(terms)


def match_record(m) -> dict:
    d = m.to_dict()
    return {
        "id": getattr(m, "_id", d.get("_id")),
        "score": getattr(m, "_score", getattr(m, "score", d.get("_score", 0.0))),
        "ticker": str(d.get("ticker", "?")).upper(),
        "year": int(d.get("year", 0)),
        "filing": str(d.get("filing_type", "")).upper(),
        "chunk": int(d["chunk_index"]) if d.get("chunk_index") is not None else "?",
        "text": str(d.get("text", "")),
    }


def _timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def run_compare(query: str, fts_state: dict, base_filter: dict | None, dense_extra_filter: dict | None, top_k: int) -> dict:
    fts_req = fq.build_request(fts_state, top_k, INCLUDE_FIELDS)
    fts_req["filter"] = and_filters(fts_req.get("filter"), base_filter)
    if not fts_req["filter"]:
        del fts_req["filter"]

    def dense():
        _, embed_ms = _timed(lambda: embed(query))
        resp, search_ms = _timed(lambda: search(dense_request(query, top_k, and_filters(base_filter, dense_extra_filter))))
        return resp, embed_ms, search_ms

    with ThreadPoolExecutor(max_workers=2) as pool:
        dense_future = pool.submit(dense)
        fts_future = pool.submit(_timed, lambda: search(fts_req))
        dense_resp, embed_ms, dense_ms = dense_future.result()
        fts_resp, fts_ms = fts_future.result()

    return {
        "dense": [match_record(m) for m in dense_resp.matches],
        "fts": [match_record(m) for m in fts_resp.matches],
        "embed_ms": embed_ms,
        "dense_ms": dense_ms,
        "fts_ms": fts_ms,
        "fts_req": fts_req,
        "terms": fq.highlight_terms(fts_state),
    }


def compare_stats(res: dict) -> dict:
    dense, fts, terms = res["dense"], res["fts"], res["terms"]
    dense_rank = {r["id"]: i for i, r in enumerate(dense, 1)}
    fts_rank = {r["id"]: i for i, r in enumerate(fts, 1)}
    shared = dense_rank.keys() & fts_rank.keys()
    union = dense_rank.keys() | fts_rank.keys()

    def avg_coverage(rows):
        if not rows or not terms:
            return None
        return sum(keyword_coverage(r["text"], terms)[0] / len(terms) for r in rows) / len(rows)

    return {
        "dense_rank": dense_rank,
        "fts_rank": fts_rank,
        "shared": shared,
        "union": union,
        "by_id": {r["id"]: r for r in dense + fts},
        "cov_dense": avg_coverage(dense),
        "cov_fts": avg_coverage(fts),
        "k": max(len(dense), len(fts)),
    }


def rank_rows(res: dict, stats: dict) -> list[dict]:
    terms = res["terms"]
    rows = []
    for doc_id in stats["union"]:
        r = stats["by_id"][doc_id]
        d, f = stats["dense_rank"].get(doc_id), stats["fts_rank"].get(doc_id)
        hits, n = keyword_coverage(r["text"], terms)
        rows.append({
            "ticker": r["ticker"],
            "year": r["year"],
            "chunk": r["chunk"],
            "dense rank": d,
            "full-text rank": f,
            "Δ rank (dense − FTS)": d - f if d and f else None,
            "found by": "both" if d and f else ("dense" if d else "full-text"),
            "keywords": f"{hits}/{n}" if n else "",
            "snippet": r["text"][:140].replace("\n", " "),
        })
    rows.sort(key=lambda x: min(x["dense rank"] or 999, x["full-text rank"] or 999))
    return rows


def rrf_fuse(stats: dict) -> list[tuple[str, float]]:
    fused = {}
    for ranks in (stats["dense_rank"], stats["fts_rank"]):
        for doc_id, rank in ranks.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (RRF_K + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: stats["k"]]


def rank_badge(doc_id: str, other_ranks: dict, other_name: str) -> str:
    if doc_id in other_ranks:
        return f"🔁 also #{other_ranks[doc_id]} in {other_name}"
    return f"◆ only in {'dense' if other_name == 'full-text' else 'full-text'}"
