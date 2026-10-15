# Vector Search Scripts (converted from Jupyter notebooks)

This folder contains two standalone Python scripts converted from the uploaded
Colab notebooks, plus the dependencies needed to run them locally in VS Code.

| Script | Converted from | What it does |
|---|---|---|
| `vector_search_faiss_pinecone.py` | `PineCone_inclass.ipynb` | Compares FAISS (local) index types (Flat, HNSW, IVF) and shows the same data stored/queried in Pinecone (cloud), including metadata filtering. |
| `dilsevector_matchmaking_search.py` | `PineConeDilseVector.ipynb` | An end-to-end "matchmaking search engine" walkthrough on Pinecone: dense (semantic) search, sparse (keyword) search, hybrid search (RRF), evaluation metrics (P/R/MRR/MAP/NDCG), reranking, and production ops (latency percentiles + caching). |

## 1. What changed from the notebooks

Both notebooks were written for Google Colab. To make them runnable from a
normal local Python environment / VS Code, the following was changed:

- **`!pip install ...` cells** → moved into `requirements.txt`.
- **Colab secrets (`google.colab.userdata.get(...)`)** → replaced with a
  plain environment variable (`PINECONE_API_KEY`), optionally loaded from a
  local `.env` file via `python-dotenv`.
- **A real Pinecone API key that appeared in one notebook's saved output**
  was **removed entirely** and replaced with an environment-variable read.
  Treat that original key as compromised — if it's still active, revoke/
  rotate it at [app.pinecone.io](https://app.pinecone.io/).
- **Cell-by-cell notebook structure** → reorganized into named functions, one
  per notebook section (A, B, C... / Section 2, Section 3...), called in
  order from a `main()` function, so each part can also be re-run or
  imported individually if you want to experiment in a Python REPL.
- Every original comment was kept and expanded where useful; new comments
  were added to explain the parts that only made sense inside a notebook
  (e.g. why batching happens, what each parameter tunes).

## 2. Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows PowerShell/CMD

# 2. Install dependencies
pip install -r requirements.txt
```

> **Python version:** `dilsevector_matchmaking_search.py` uses modern type
> hints (`list[dict]`, `dict | None`) that require **Python 3.10+**. If
> you're on an older Python, either upgrade or remove those hints.

## 3. Pinecone API key

Both scripts need a free Pinecone API key for their cloud-search sections.

1. Sign up at [pinecone.io](https://www.pinecone.io/) and copy your API key
   from [app.pinecone.io](https://app.pinecone.io/).
2. Provide it to the scripts using **one** of these methods:

   **Option A — environment variable**
   ```bash
   export PINECONE_API_KEY="your-key-here"        # macOS/Linux
   $env:PINECONE_API_KEY="your-key-here"           # Windows PowerShell
   ```

   **Option B — `.env` file** (recommended for local dev)
   Create a file named `.env` in this same folder:
   ```
   PINECONE_API_KEY=your-key-here
   ```
   The scripts auto-load this via `python-dotenv`. **Add `.env` to your
   `.gitignore`** so you never commit a real key.

`vector_search_faiss_pinecone.py` will run its FAISS section even with no
key set — the Pinecone section is simply skipped with a message. Every
Pinecone call in `dilsevector_matchmaking_search.py` requires a valid key.

## 4. Running the scripts

```bash
python vector_search_faiss_pinecone.py
python dilsevector_matchmaking_search.py
```

### Cost / time note for `dilsevector_matchmaking_search.py`
This script creates **two live Pinecone indexes** (`dilse-dense` and
`dilse-sparse`) and upserts 400 records into each using Pinecone's hosted
embedding models. A full run takes roughly 1–2 minutes and consumes your
Pinecone account quota (free tier is generally sufficient for this demo
size). When you're done experimenting, delete the indexes to avoid ongoing
storage charges:

```python
from dilsevector_matchmaking_search import connect_to_pinecone, cleanup_indexes
pc = connect_to_pinecone()
cleanup_indexes(pc)
```

or simply uncomment the `cleanup_indexes(pc)` line at the bottom of `main()`
in the script before running it.

## 5. File overview

```
.
├── vector_search_faiss_pinecone.py     # Script 1: FAISS + Pinecone basics
├── dilsevector_matchmaking_search.py   # Script 2: full matchmaking search pipeline
├── requirements.txt                    # pip dependencies for both scripts
└── README.md                           # this file
```

### `vector_search_faiss_pinecone.py` sections
- **Section 2 — FAISS (local):** builds embeddings for 5 toy documents with
  `sentence-transformers`, then queries them with three FAISS index types
  (`IndexFlatL2`, `IndexHNSWFlat`, `IndexIVFFlat`) to compare exact vs.
  approximate search.
- **Section 3 — Pinecone (cloud):** creates a Pinecone index, upserts 5
  documents with metadata, then runs a plain semantic search plus three
  metadata-filtered searches (`$eq`, `$in`, `$and`).

### `dilsevector_matchmaking_search.py` sections
- **A — Setup & Data:** connects to Pinecone, generates 400 synthetic dating
  profiles (3 are hand-planted for reliable demo results).
- **B — Dense Index:** creates a Pinecone index using a hosted embedding
  model and upserts all profiles.
- **C — Semantic Search:** shows plain vector search matches by meaning, not
  literal facts.
- **D — Metadata Filters:** shows how to combine a fuzzy query with an exact
  metadata `WHERE` clause in one call.
- **E — Sparse (Keyword) Index:** adds a BM25-style keyword index and
  contrasts it with dense search.
- **F — Hybrid Search:** merges dense + sparse rankings using Reciprocal
  Rank Fusion (RRF), with a tunable `alpha` weight.
- **G — Compression:** back-of-envelope math on vector storage size before/
  after quantization, plus an optional local FAISS Product Quantization demo.
- **H — Evaluation Metrics:** implements Precision@k, Recall@k, MRR@k,
  Average Precision@k, and NDCG@k from scratch, then applies them to real
  search results using synthetic ground truth.
- **I — Reranking:** adds a two-stage retrieve-then-rerank pipeline using a
  hosted cross-encoder model and compares metrics before/after.
- **J — Production Ops:** measures p50/p95/p99 search latency and
  demonstrates a simple in-memory result cache.
- **K — Wrap-up:** `cleanup_indexes()` deletes both Pinecone indexes.

## 6. Troubleshooting

- **`faiss` fails to install on Apple Silicon / some Linux distros:** try
  `pip install faiss-cpu --no-cache-dir`, or install via conda
  (`conda install -c pytorch faiss-cpu`) if pip continues to fail.
- **`RuntimeError: PINECONE_API_KEY not found`:** double-check your `.env`
  file is in the same folder you're running the script from, or that the
  environment variable is exported in the same terminal session you're
  running Python from.
- **Pinecone index creation seems to hang:** index creation is asynchronous;
  both scripts poll every second until Pinecone reports the index as ready,
  which normally takes well under a minute.
