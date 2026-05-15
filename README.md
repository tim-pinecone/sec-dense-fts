# SEC Document Search — Pinecone FTS + Semantic

A demo application showing how to build a hybrid search system over SEC 10-K filings using [Pinecone's full-text search (preview)](https://docs.pinecone.io/guides/search/full-text-search) combined with dense vector embeddings.

The app supports three search modes:

| Mode | How it works |
|---|---|
| **Full-text** | BM25 keyword search over the document text |
| **Semantic** | Dense vector similarity via OpenAI embeddings |
| **Hybrid** | Dense vector ranking with a must-contain keyword filter — results are semantically ranked and guaranteed to contain the specified terms |

## Data

`example_data/` contains chunked 10-K filings for six companies across six years:

| Ticker | Company | Years |
|---|---|---|
| AAPL | Apple | 2019–2024 |
| AMZN | Amazon | 2019–2024 |
| F | Ford | 2019–2024 |
| GM | General Motors | 2019–2024 |
| MSFT | Microsoft | 2019–2024 |
| ORCL | Oracle | 2019–2024 |

~24,000 document chunks total. Each chunk has an `_id`, `text`, `ticker`, `filing_type`, `year`, and `chunk_index`.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- A [Pinecone API key](https://app.pinecone.io/) (free tier works)
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```bash
git clone https://github.com/tim-pinecone/sec-dense-fts
cd sec-dense-fts

cp .env.example .env
# Add your PINECONE_API_KEY and OPENAI_API_KEY to .env

uv sync
```

## Ingest

Creates the Pinecone index, embeds all chunks with `text-embedding-3-small`, and upserts them in batches. Safe to re-run — skips index creation if it already exists.

```bash
uv run main.py
```

Ingestion takes a few minutes (OpenAI embedding calls are the bottleneck). The script polls until all documents are searchable before exiting.

## Run the app

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

Use the sidebar to filter by ticker and year, pick a search mode, and enter your query.

## Index schema

```
text          — full-text search field (BM25, English, stemming enabled)
embedding     — dense vector, 1536 dims, cosine similarity
ticker        — filterable metadata (string)
filing_type   — filterable metadata (string)
year          — filterable metadata (integer)
chunk_index   — filterable metadata (integer)
```

The FTS and vector fields are declared in the schema at index creation. Metadata fields (`ticker`, `filing_type`, `year`, `chunk_index`) are automatically indexed — they do not need to be declared.

## How hybrid search works

The hybrid mode uses a single Pinecone query:

- `score_by` — dense vector cosine similarity (semantic ranking)
- `filter` — `$match_all` on the text field (hard lexical requirement)

This means results are ordered by semantic relevance, but only chunks that contain all the specified keywords are returned. It's useful for queries like "what does MSFT say about Azure capital expenditure" — the semantic query captures the intent, and the text filter ensures the specific terms are present.
