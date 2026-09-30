---
title: SEC FTS Hybrid Comparison
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Pinecone full-text search vs dense vectors on 10-Ks
---

# SEC Document Search — Pinecone FTS + Semantic

A demo application showing how to build a hybrid search system over SEC 10-K filings using [Pinecone's full-text search](https://docs.pinecone.io/guides/search/full-text-search) combined with dense vector embeddings.

The app supports three search modes:

| Mode | How it works |
|---|---|
| **Full-text** | BM25 keyword search over the document text |
| **Query builder (full-text)** | Construct complex full-text queries — Lucene boolean / phrase / proximity / boost / prefix scoring plus text-match and metadata filters — starting from 10 worked examples |
| **Semantic** | Dense vector similarity via OpenAI embeddings |
| **Compare: dense vs full-text** | Runs a dense search and a full-text search side by side and contrasts them — overlap, rank shifts, keyword coverage, latency, and an RRF-fused list |
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

Two UIs share the same search logic (`search_core.py`, `fts_queries.py`):

| UI | Command | URL | Notes |
|---|---|---|---|
| Gradio | `uv run python app.py` | [http://localhost:7860](http://localhost:7860) | What the Hugging Face Space runs |
| Streamlit | `uv run streamlit run app_streamlit.py` | [http://localhost:8501](http://localhost:8501) | Local development UI |

Both have the same five tabs/modes. Use the sidebar to filter by ticker and year, pick a search mode, and enter your query — or pick one of the prepared **Example queries** at the top of each mode (three per mode; ten in the query builder). Examples live in `fts_queries.py` (`MODE_EXAMPLES`, `EXAMPLES`).

## Full-text query builder

The **Query builder** mode exposes the full Pinecone FTS query surface. Queries are built in three parts, and the exact `documents.search(...)` request is shown as JSON and Python before running:

1. **Scoring** — either BM25 keywords (`text`) or Lucene (`query_string`). Lucene can be written raw or assembled row-by-row with the clause builder (MUST `+` / MUST NOT `-` / SHOULD, term / phrase / phrase prefix, slop `~N`, boost `^N`).
2. **Text-match filters** — `$match_phrase`, `$match_all`, `$match_any`, each optionally negated with `$not`, combined with `$and` or `$or`.
3. **Metadata filters** — ticker `$in` / `$nin`, year range, chunk-index range.

Pick an example from the dropdown to load it into the builder, then tweak it:

| Example | Demonstrates |
|---|---|
| Supply-chain shortages, excluding COVID | `text:(("supply chain" OR semiconductor) AND shortage) NOT text:(covid)` |
| Cyber incidents | `text:(+cybersecurity ransomware^3 breach -insurance)` |
| Rising interest rates | Proximity: `text:("interest rates increase"~5)` |
| China trade & tariffs | Term boost: `text:(tariffs^3 trade china)` |
| AI mentions | Phrase prefix `"artificial intel"*` + required term + year range |
| EV batteries at Ford & GM | BM25 + `$match_phrase` filter + ticker `$in` + year range |
| Cloud growth, no pandemic talk | `$not` + `$match_any` exclusion, ticker `$nin` |
| Regulators: EC or DOJ | `$or` across two `$match_phrase` filters |
| Inflation & input costs | `$match_all` + year range |
| Buybacks vs. dividends | Required OR-groups `+(a OR b) +(c OR d)` |

Things the server enforces (surfaced in the UI):

- One scoring type per request — `text` **or** `query_string`, never mixed.
- `query_string` clauses may not set `fields`; qualify terms inline (`text:(...)`).
- A Lucene query of only exclusions (`text:(-covid)`) is rejected.
- Proximity, boost and phrase prefix are scoring-only — they can't be used in `filter`.
- Phrase-prefix matches all receive the same constant score.

The query compilation logic and examples live in `fts_queries.py`.

## Comparing dense vs full-text

**Compare** mode runs both searches in parallel for the same question:

- **Dense side** — the query is embedded with `text-embedding-3-small` and ranked by cosine similarity.
- **Full-text side** — either the same query as BM25 keywords, or whatever complex query is currently set up in the **Query builder** (Lucene, text-match filters, metadata filters). When using the builder, you can optionally apply its filters to the dense side too, so only the ranking signal differs.

Sidebar ticker/year filters apply to both sides. The results show:

| View | What it tells you |
|---|---|
| Overlap / Jaccard, only-dense, only-full-text | How much the two retrieval methods agree in the top-k |
| Latency | Dense (embedding + search) vs full-text search time |
| Keyword coverage | Share of the query's keywords present in each result — dense results with low coverage are paraphrase / concept matches that BM25 can't find |
| Side by side | Both ranked lists, with badges showing each result's rank in the other list, and keyword highlighting on both |
| Rank comparison | One table of every retrieved chunk with its dense rank, full-text rank and Δ |
| Fused (RRF) | Client-side reciprocal rank fusion (k = 60) of the two lists — a preview of a two-query hybrid |

## Deploying to Hugging Face Spaces

The Space runs the **Gradio** app (`app.py`) on free ZeroGPU hardware — no Docker needed. The YAML block at the top of this README is the Space config, and `requirements.txt` holds the Space's Python dependencies (Gradio itself comes from `sdk_version`). ZeroGPU requires at least one `@spaces.GPU` function; `app.py` defines a no-op one since all compute happens in Pinecone and OpenAI.

1. Add a Hugging Face write token to `.env` as `HF_TOKEN`.
2. Preview what will be uploaded:
   ```bash
   uv run python deploy_space.py <owner>/<space-name> --dry-run
   ```
3. Deploy. The first run creates the Space; `--set-secrets` copies `PINECONE_API_KEY` and `OPENAI_API_KEY` from `.env` into the Space's secrets (only needed once, or when keys change):
   ```bash
   uv run python deploy_space.py <owner>/<space-name> --set-secrets [--private]
   ```

`deploy_space.py` uploads an explicit allowlist (`README.md`, `requirements.txt`, `app.py`, `search_core.py`, `fts_queries.py`), so `.env`, the example data, and the Streamlit app never leave your machine. Re-run it without `--set-secrets` to push code changes.

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
