# Vector Search Demo — FAISS (Local)

A small, self-contained demo comparing three FAISS index types for
similarity search over text embeddings:

| Index | Type            | Recall  | Speed at scale       |
|-------|-----------------|---------|-----------------------|
| Flat  | Exact           | 100%    | Slow (O(N))           |
| HNSW  | ANN (graph)     | ~95–99% | Fast                  |
| IVF   | ANN (cluster)   | ~95–99% | Fast, lower memory    |

On the tiny 5-document dataset used here, all three indexes will likely
return the same top matches — the differences only become meaningful at
scale (thousands to millions of vectors).

## Files

- `Vector_Database_1.py` — the demo script
- `requirements.txt` — Python dependencies
- `README.md` — this file

## Setup

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python Vector_Database_1.py
```

The script will:
1. Embed 5 sample text snippets with `all-MiniLM-L6-v2`.
2. Search for the query `"AI systems"` using a **Flat** (exact) index.
3. Repeat the same search using an **HNSW** (graph-based ANN) index.
4. Repeat the same search using an **IVF** (cluster-based ANN) index.

Each section prints the top-3 matching texts so you can compare results
across index types.

## Key tunable parameters (recall vs. speed trade-offs)

| Parameter         | Index         | Higher value →                                         | Lower value →                                |
|--------------------|--------------|----------------------------------------------------------|------------------------------------------------|
| `M`                | HNSW         | Better recall, more memory, slower build                 | Less memory, faster build, lower recall         |
| `efConstruction`   | HNSW (build) | Better neighbor quality, slower build                    | Faster build, lower recall                      |
| `efSearch`         | HNSW (query) | Better recall, slower queries                             | Faster queries, lower recall                    |
| `nlist`            | IVF          | Faster fine search, slower coarse search, more memory     | Slower fine search, cheaper coarse search       |
| `nprobe`           | IVF (query)  | Better recall, slower queries                              | Faster queries, risk of missing neighbors       |

**Rules of thumb:**
- HNSW: `M = 16–32` is a common production range; `efSearch` should be
  set to at least `k` (often 5–10× `k`) for high recall.
- IVF: `nlist ≈ sqrt(N)` for large datasets (`N` = total vectors);
  `nprobe` must be `<= nlist` and is tuned against your latency SLA.

In production, `M` / `nlist` are typically fixed at index-build time
based on dataset size, while `efSearch` / `nprobe` are tuned per query
to balance latency against recall.

## Notes

- Vectors must be `float32` for FAISS.
- IVF requires a training step (`.train()`) before adding vectors, since
  it needs to run k-means to find cluster centroids.
- HNSW supports incremental inserts without retraining; IVF supports
  O(1) appends to existing cluster lists once trained.
- For datasets under ~10,000 vectors, brute-force (Flat) search is often
  simpler and just as fast as maintaining an ANN index.
