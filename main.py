import csv
import glob
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, SchemaBuilder

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_NAME = "sec-fts"
NAMESPACE = "__default__"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 50  # keep small due to dense vector payload size

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── Index setup ───────────────────────────────────────────────────────────────
def create_index():
    if pc.indexes.exists(INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists, skipping creation.")
        return

    schema = (
        SchemaBuilder()
        .add_string_field("text", full_text_search={"language": "en", "stemming": True})
        .add_dense_vector_field("embedding", dimension=EMBED_DIM, metric="cosine")
        .build()
    )
    print(f"Creating index '{INDEX_NAME}' and waiting for Ready...")
    pc.indexes.create(
        name=INDEX_NAME,
        schema=schema,
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    )
    print("Index is ready.")


# ── Data loading ──────────────────────────────────────────────────────────────
def load_docs(data_dir: str = "example_data") -> list[dict]:
    docs = []
    for path in sorted(glob.glob(f"{data_dir}/*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                docs.append({
                    "_id": row["_id"],
                    "text": row["text"],
                    "ticker": row["ticker"],
                    "filing_type": row["filing_type"],
                    "year": int(row["year"]),
                    "chunk_index": int(row["chunk_index"]),
                })
    return docs


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


# ── Ingest ────────────────────────────────────────────────────────────────────
def ingest(idx, docs: list[dict]) -> None:
    total = len(docs)
    print(f"\nIngesting {total} documents in batches of {BATCH_SIZE}...")
    t0 = time.time()

    for start in range(0, total, BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        embeddings = embed_texts([d["text"] for d in batch])
        records = [{**d, "embedding": emb} for d, emb in zip(batch, embeddings)]

        result = idx.documents.batch_upsert(namespace=NAMESPACE, documents=records)
        if result.has_errors:
            for err in result.errors:
                print(f"  ERROR batch @{start}: {err.error_message}")
            raise RuntimeError(f"Batch upsert failed at offset {start}")

        print(f"  batch @{start:5d}: {len(batch):3d} docs  (total: {start + len(batch)}/{total})")

    print(f"\nUpsert complete in {time.time() - t0:.1f}s — polling for searchability...")

    sentinel = docs[0]["text"].split()[0] if docs else "the"
    deadline = time.time() + 300
    while time.time() < deadline:
        probe = idx.documents.search(
            namespace=NAMESPACE,
            top_k=1,
            score_by=[{"type": "text", "field": "text", "query": sentinel}],
            include_fields=[],
        )
        if probe.matches:
            print(f"Searchable after {time.time() - t0:.1f}s total.\n")
            return
        time.sleep(5)

    print("WARNING: poll deadline exceeded — documents may still be indexing.")


# ── Query helpers ─────────────────────────────────────────────────────────────
def _build_filter(ticker: str = None, year: int = None) -> dict | None:
    filt = {}
    if ticker:
        filt["ticker"] = {"$eq": ticker}
    if year:
        filt["year"] = {"$eq": year}
    return filt or None


def search_text(idx, query: str, ticker: str = None, year: int = None, top_k: int = 5):
    """BM25 full-text search with optional ticker/year filters."""
    filt = _build_filter(ticker, year)
    return idx.documents.search(
        namespace=NAMESPACE,
        top_k=top_k,
        score_by=[{"type": "text", "field": "text", "query": query}],
        **{"filter": filt} if filt else {},
        include_fields=["text", "ticker", "filing_type", "year", "chunk_index"],
    )


def search_semantic(idx, query: str, ticker: str = None, year: int = None, top_k: int = 5):
    """Dense vector semantic search with optional ticker/year filters."""
    query_emb = embed_texts([query])[0]
    filt = _build_filter(ticker, year)
    return idx.documents.search(
        namespace=NAMESPACE,
        top_k=top_k,
        score_by=[{"type": "dense_vector", "field": "embedding", "values": query_emb}],
        **{"filter": filt} if filt else {},
        include_fields=["text", "ticker", "filing_type", "year", "chunk_index"],
    )


def search_hybrid(idx, query: str, ticker: str = None, year: int = None, top_k: int = 5):
    """Dense vector ranking with lexical hard filter — guarantees query tokens are present."""
    query_emb = embed_texts([query])[0]
    filt = _build_filter(ticker, year) or {}
    filt["text"] = {"$match_all": query}
    return idx.documents.search(
        namespace=NAMESPACE,
        top_k=top_k,
        score_by=[{"type": "dense_vector", "field": "embedding", "values": query_emb}],
        filter=filt,
        include_fields=["text", "ticker", "filing_type", "year", "chunk_index"],
    )


def print_results(resp, label: str) -> None:
    print(f"\n── {label} ──")
    for m in resp.matches:
        d = m.to_dict()
        score = getattr(m, "_score", getattr(m, "score", None))
        print(f"  [{score:.4f}] {d.get('ticker','?').upper()} {d.get('year')} chunk {d.get('chunk_index')}")
        print(f"    {str(d.get('text', ''))[:140]}...")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_index()
    idx = pc.index(name=INDEX_NAME)

    docs = load_docs("example_data")
    ingest(idx, docs)

    print_results(
        search_text(idx, "revenue growth", ticker="aapl"),
        "FTS — 'revenue growth' (AAPL)",
    )
    print_results(
        search_semantic(idx, "supply chain disruption risk"),
        "Semantic — 'supply chain disruption risk'",
    )
    print_results(
        search_hybrid(idx, "cloud computing", ticker="msft", year=2023),
        "Hybrid — 'cloud computing' (MSFT 2023)",
    )
