"""
dilsevector_matchmaking_search.py

DilSeVector — Production-Style Matchmaking Search with Pinecone
=================================================================
Converted from: PineConeDilseVector.ipynb

A guided, end-to-end walkthrough of building a "matchmaking search engine"
on top of Pinecone (a managed vector database), using 400 synthetic dating
profiles. Each section solves one realistic production problem:

  A. Setup & Data              - connect to Pinecone, generate 400 fake profiles
  B. Build the Dense Index     - create a semantic ("dense vector") index and upsert profiles
  C. Basic Semantic Search     - search by MEANING, and see where it fails on exact facts
  D. Metadata Filters          - use filters for exact facts instead of embeddings
  E. Keyword Search (Sparse)   - add a BM25-style keyword index to catch exact-word matches
  F. Hybrid Search             - combine dense + sparse results with Reciprocal Rank Fusion
  G. Scaling / Compression     - understand vector quantization (why Pinecone shrinks memory use)
  H. Evaluation Metrics        - measure search quality with Precision, Recall, MRR, MAP, NDCG
  I. Reranking                 - add a second-pass reranker to fix poor ranking
  J. Production Ops            - measure latency percentiles (p50/p95/p99) and add a simple cache
  K. Wrap-up / Cleanup         - delete indexes when done, to avoid ongoing charges

HOW TO RUN (from VS Code / terminal)
-------------------------------------
1. Create and activate a virtual environment (recommended):
       python -m venv .venv
       .venv\\Scripts\\activate      (Windows)
       source .venv/bin/activate    (macOS/Linux)

2. Install dependencies:
       pip install -r requirements.txt

3. Set your Pinecone API key as an environment variable.
   Get a free key at https://app.pinecone.io/
       Windows (PowerShell):  $env:PINECONE_API_KEY="your-key-here"
       macOS/Linux (bash):    export PINECONE_API_KEY="your-key-here"

   Alternatively, create a ".env" file next to this script containing:
       PINECONE_API_KEY=your-key-here
   (python-dotenv, included in requirements.txt, will auto-load it.)

   IMPORTANT: Never commit a real API key to source control. The original
   notebook this script was converted from had a Colab-only secret manager
   call (google.colab.userdata) — that has been replaced below with a plain
   environment-variable read so it works in any local Python environment.

4. Run the script:
       python dilsevector_matchmaking_search.py

   NOTE ON COST/TIME: This script creates two live Pinecone indexes and
   upserts 400 records into each (with Pinecone's hosted/integrated
   embedding models), which takes roughly 1-2 minutes and uses your
   Pinecone account quota. Section K shows the cleanup calls to delete
   both indexes when you're done experimenting.
"""

import math
import os
import random
import time

import numpy as np
from dotenv import load_dotenv           # lets us keep the API key in a local .env file
from pinecone import Pinecone            # the client class used to talk to the Pinecone service

load_dotenv()  # loads variables from a ".env" file in the current directory, if present

# Namespace: a logical partition inside a Pinecone index (similar to a table/collection)
NS = "users"

# Index names used throughout the script
DENSE_INDEX_NAME = "dilse-dense1"     # semantic ("dense vector") index
SPARSE_INDEX_NAME = "dilse-sparse1"   # keyword ("sparse vector" / BM25-style) index


# ---------------------------------------------------------------------------
# SECTION A — SETUP & DATA
# ---------------------------------------------------------------------------

def connect_to_pinecone() -> Pinecone:
    """Create a Pinecone client using the PINECONE_API_KEY environment variable."""

    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise RuntimeError(
            "PINECONE_API_KEY not found. Set it as an environment variable, "
            "or add PINECONE_API_KEY=... to a .env file next to this script."
        )

    pc = Pinecone(api_key=pinecone_key)  # from app.pinecone.io
    print("Connected to Pinecone \u2705")
    return pc


def build_user_profiles(n_users: int = 400, seed: int = 42) -> list[dict]:
    """
    Build `n_users` fake dating profiles as plain Python dictionaries.

    Each dict is one "record" that will later be uploaded ("upserted") to
    Pinecone. Three profiles are hand-edited (planted) afterwards so every
    later demo section has a guaranteed, predictable result to point at:

      P0001 - the exact profile a later "biodata" query is looking for
              (IIT Bombay - 2019 - Marathi - Dadar)
      P0002 - a "paraphrase twin": means almost the same thing as a later
              query but shares ZERO exact words with it ("sorted but chill"
              vs "fun-loving but settled") - used to contrast dense vs
              keyword search.
      P0003 - the "golden match" used later to demonstrate reranking
              (non-smoker, golden retriever, Dadar).
    """

    random.seed(seed)  # fixed seed => the "random" profiles are identical every run

    # Pools of values used to randomly build each fake profile
    LOCALITIES = ["Dadar", "Thane", "Byculla", "Dombivli", "South Bombay", "Bandra", "Andheri", "Powai"]
    COLLEGES = ["IIT Bombay", "VJTI", "NMIMS", "St Xaviers", "Mumbai University", "SPIT", "ICT Mumbai"]
    JOBS = [
        "software engineer at TCS", "chartered accountant", "doctor", "startup founder",
        "graphic designer", "data scientist", "teacher", "investment banker", "architect",
    ]
    VIBES = [
        "adventurous and loves trekking in the Sahyadris",
        "fun-loving but settled, enjoys quiet weekends",
        "foodie who hunts for the best vada pav in the city",
        "bookworm, poetry evenings at Prithvi cafe",
        "gym at 6am, marathon runner, discipline is life",
        "stand-up comedy fan, drags friends to open mics",
        "loves dogs, volunteers at animal shelters on weekends",
        "sorted but chill, big on family, small on drama",
        "travel junkie, 14 countries and counting",
        "classical music and long walks at Marine Drive",
    ]
    LANGS = ["Marathi", "Gujarati", "Hindi", "Konkani", "Tamil"]

    # Build n_users random profiles as plain Python dictionaries
    users = []
    for i in range(n_users):
        u = {
            "_id": f"P{i:04d}",
            "locality": random.choice(LOCALITIES),
            "college": random.choice(COLLEGES),
            "batch": random.choice(range(2014, 2023)),
            "job": random.choice(JOBS),
            "vibe": random.choice(VIBES),
            "lang": random.choice(LANGS),
            "smoke": random.choice(["non-smoker", "occasional smoker"]),
        }
        users.append(u)

    # --- Planted demo profiles ---------------------------------------------
    # These 3 profiles are hand-edited afterwards so later demos have a
    # guaranteed, predictable result to point at.

    # P0001 (index 1): the "perfect" profile that exactly matches the
    # biodata-style query used later - IIT Bombay, batch 2019, Marathi,
    # lives in Dadar, non-smoker.
    users[1].update(
        college="IIT Bombay", batch=2019, lang="Marathi", locality="Dadar", smoke="non-smoker",
        vibe="loves dogs, volunteers at animal shelters on weekends", job="data scientist",
    )

    # P0002 (index 2): a "paraphrase twin" - means almost the same thing as
    # another query but shares ZERO exact words with it ("sorted but chill"
    # vs "fun-loving but settled"). Used to show dense/keyword search have
    # different blind spots.
    users[2].update(vibe="sorted but chill, big on family, small on drama", locality="Bandra")

    # P0003 (index 3): the "golden match" used later to demonstrate reranking.
    users[3].update(
        smoke="non-smoker", locality="Dadar", job="architect",
        vibe="loves dogs, has a golden retriever named Simba",
    )

    # Build a human-readable "bio" string for each user. This is the text
    # field that Pinecone will actually turn into a vector (embed), and it's
    # also what the app would display on a profile card.
    for u in users:
        u["bio"] = (
            f"{u['job'].capitalize()}, {u['college']} {u['batch']} batch. {u['lang']} speaking, "
            f"lives in {u['locality']}. {u['vibe'].capitalize()}. {u['smoke'].capitalize()}."
        )

    # Sanity check: print the 3 planted profiles + total count
    for i in (1, 2, 3):
        print(users[i]["_id"], "\u2192", users[i]["bio"])
    print("Total users:", len(users))

    return users


# ---------------------------------------------------------------------------
# SECTION B — BUILD THE DENSE (SEMANTIC) SEARCH INDEX
# ---------------------------------------------------------------------------

def create_dense_index(pc: Pinecone):
    """
    Create (or reuse) the dense/semantic index.

    Notice what's NOT here: no model download, no .encode(), no numpy. We
    tell Pinecone which hosted embedding model to use and WHICH FIELD to
    embed, and Pinecone handles the rest server-side ("integrated embedding").
    """

    existing_indexes = [i.name for i in pc.list_indexes()]
    if DENSE_INDEX_NAME not in existing_indexes:
        pc.create_index_for_model(
            name=DENSE_INDEX_NAME, 
            cloud="aws", 
            region="us-east-1",
            # "Integrated embedding": Pinecone will embed text FOR us.
            embed={
                "model": "llama-text-embed-v2",     # Pinecone-hosted embedder ("the Romantic")
                "field_map": {"text": "bio"},        # embed the 'bio' field of each record
            },
        )
    else:
        print(f"Index '{DENSE_INDEX_NAME}' already exists. Skipping creation.")

    # Creating an index is async - poll until Pinecone reports it's ready to use
    while not pc.describe_index(DENSE_INDEX_NAME).status["ready"]:
        time.sleep(1)

    dense = pc.Index(DENSE_INDEX_NAME)   # a handle we'll use to upsert/search this index
    print("Dense index ready \u2705")
    return dense


def upsert_dense(dense, users: list[dict]):
    """
    Upsert (insert-or-update) all user profiles into the dense index.

    Records go up as plain dicts. Pinecone embeds `bio` server-side; the
    other fields become filterable metadata. 96 records per call is the
    integrated-embedding batch limit, hence the batching loop.
    """

    for i in range(0, len(users), 96):
        dense.upsert_records(namespace=NS, records=users[i:i + 96])  # sends plain dicts; Pinecone embeds 'bio' server-side

    time.sleep(10)  # give the async embedding a moment (embedding happens in the background)
    print(dense.describe_index_stats())  # confirms how many vectors are now stored


# ---------------------------------------------------------------------------
# SECTION C — BASIC SEMANTIC SEARCH (AND WHERE IT FAILS)
# ---------------------------------------------------------------------------

def search(index, text: str, k: int = 5, flt: dict | None = None):
    """
    Run a semantic/keyword search against `index`.

    text : the natural-language query
    k    : how many results to return (top_k)
    flt  : optional metadata filter dict, e.g. {"college": "IIT Bombay"}
    Returns a list of (profile_id, score, bio) tuples.
    """

    q = {"inputs": {"text": text}, "top_k": k}
    if flt:
        q["filter"] = flt  # attach the filter only if one was given

    res = index.search(namespace=NS, query=q, fields=["bio"])
    return [
        (h["id"], round(h["score"], 4), h["fields"]["bio"])
        for h in res["result"]["hits"]
    ]


def show(hits, star: set = ()):
    """Pretty-print search results. `star` is a set of profile ids to mark with a star."""
    for r, (pid, s, bio) in enumerate(hits, 1):
        marker = "  *" if pid in star else ""
        print(f"{r:>2}. {pid}  {s}{marker}\n    {bio[:105]}")


def fact_check(hits):
    """
    Print a check-mark table showing whether each of the 4 required facts
    (college, batch, language, locality) literally appears in each hit's bio.

    Used to illustrate that plain semantic search matches on VIBE, not on
    literal, structured facts.
    """
    CRISIS2_FACTS = ("IIT Bombay", "2019", "Marathi", "Dadar")
    print(f"{'rank':<5}{'id':<7}{'IIT-B':<7}{'2019':<7}{'Marathi':<9}{'Dadar':<7}")

    for r, (pid, s, bio) in enumerate(hits, 1):
        f = ["Y" if w in bio else "N" for w in CRISIS2_FACTS]
        perfect = " *** PERFECT" if f == ["Y"] * 4 else ""
        print(f"{r:<5}{pid:<7}{f[0]:<7}{f[1]:<7}{f[2]:<9}{f[3]:<7}{perfect}")


def run_semantic_search_demo(dense):
    """Section C: show that plain semantic search matches by MEANING, not literal facts."""

    # Try a plain semantic search: notice the query text doesn't need to
    # match the bio word-for-word - matches come from MEANING, not exact keywords.
    show(search(dense, "someone adventurous who loves trekking"))

    # Now run a "biodata"-style query (structured facts) through PURE semantic
    # search. Expect near-misses: dense search chases "vibe", not literal facts.
    crisis2_query = "IIT Bombay 2019 batch, Marathi speaking, lives in Dadar"
    fact_check(search(dense, crisis2_query))
    print("\n(The perfect profile P0001 exists. Did pure semantic search even surface it?)")


# ---------------------------------------------------------------------------
# SECTION D — PRODUCTION FIX #1: METADATA FILTERS
# ---------------------------------------------------------------------------
# Structured facts don't belong in embeddings - they belong in FILTERS.
# Split any query into:
#   1. the FUZZY part (feelings/vibe)  -> handled by the embedding
#   2. the EXACT facts (college/batch/lang/locality) -> handled by a metadata
#      filter (a WHERE clause), applied in the SAME search call.
# ---------------------------------------------------------------------------

def run_metadata_filter_demo(dense):
    """Section D: combine a fuzzy semantic query with a hard metadata filter."""

    exact_facts_filter = {
        "college": "IIT Bombay",
        "batch": 2019,          # the facts, as a WHERE clause
        "lang": "Marathi",
        "locality": "Dadar",
    }

    # Example 1: fuzzy text ("looking for a genuine connection") + exact filter
    hits = search(dense, "looking for a genuine connection", flt=exact_facts_filter)
    fact_check(hits)  # now every returned hit is GUARANTEED to satisfy all 4 facts

    # Example 2: a different fuzzy text, same filter - still guaranteed correct facts
    hits = search(dense, "someone adventurous who loves trekking", flt=exact_facts_filter)
    fact_check(hits)


# ---------------------------------------------------------------------------
# SECTION E — KEYWORD SEARCH (SPARSE / BM25 INDEX)
# ---------------------------------------------------------------------------
# Filters are binary: in or out. Sometimes you want exact words to BOOST
# ranking, not gatekeep - e.g. "golden retriever" should rank higher, not
# exclude everyone else. That's a sparse index: Pinecone hosts a BM25-family
# lexical model. Same API, second index, zero manual math required.
# ---------------------------------------------------------------------------

def create_sparse_index(pc: Pinecone, users: list[dict]):
    """Create (or reuse) the sparse/keyword index and upsert the same user profiles into it."""

    if not pc.has_index(SPARSE_INDEX_NAME):
        pc.create_index_for_model(
            name=SPARSE_INDEX_NAME, cloud="aws", region="us-east-1",
            embed={
                "model": "pinecone-sparse-english-v0",   # hosted BM25-style lexical model
                "field_map": {"text": "bio"},
            },
        )
    while not pc.describe_index(SPARSE_INDEX_NAME).status["ready"]:
        time.sleep(1)

    sparse = pc.Index(SPARSE_INDEX_NAME)

    # Upload the same user profiles into the sparse index, in batches of 96 again
    for i in range(0, len(users), 96):
        sparse.upsert_records(namespace=NS, records=users[i:i + 96])
    time.sleep(10)
    print("Sparse index ready \u2705")
    return sparse


def run_sparse_vs_dense_demo(dense, sparse):
    """
    Section E: contrast sparse (keyword) vs dense (semantic) search.

    - Sparse search nails exact facts easily (good at literal word matches).
    - Sparse search struggles on paraphrases with zero shared words.
    - Dense search still finds the paraphrase, because it matches on MEANING.
    They have complementary blind spots -> motivates hybrid search (Section F).
    """

    crisis2_query = "IIT Bombay 2019 batch, Marathi speaking, lives in Dadar"

    print("=== Sparse (keyword) search on the biodata query ===")
    fact_check(search(sparse, crisis2_query))

    # Paraphrase query: P0002's bio shares ZERO exact words with this text
    # ("sorted but chill" vs "fun-loving but settled").
    paraphrase_query = "fun-loving but settled, enjoys a quiet weekend"

    print("\n=== Sparse (keyword) search on a paraphrase query ===")
    show(search(sparse, paraphrase_query), star={"P0002"})

    print("\n=== Dense (semantic) search on the same paraphrase query ===")
    show(search(dense, paraphrase_query), star={"P0002"})

    print("\nComplementary blind spots -> use both together. That's hybrid search.")


# ---------------------------------------------------------------------------
# SECTION F — HYBRID SEARCH (DENSE + SPARSE, MERGED WITH RRF)
# ---------------------------------------------------------------------------

def hybrid_search(dense, sparse, users: list[dict], text: str, alpha: float = 0.5, k: int = 5):
    """
    Combine dense + sparse search results using Reciprocal Rank Fusion (RRF).

    alpha : how much to trust the dense (semantic) results vs sparse (keyword) results.
            alpha = 1.0 -> pure dense
            alpha = 0.0 -> pure sparse
            0.5         -> balanced
    RRF works on RANKS (1st place, 2nd place, ...) rather than raw scores,
    because dense and sparse scores live on different, incomparable scales.
    """

    d = [pid for pid, _, _ in search(dense, text, k=25)]   # top 25 ids from dense search
    s = [pid for pid, _, _ in search(sparse, text, k=25)]  # top 25 ids from sparse search

    score = {}
    # RRF formula: each result contributes 1/(60 + rank) to its fused score,
    # weighted by alpha for dense and (1 - alpha) for sparse.
    for rank, pid in enumerate(d):
        score[pid] = score.get(pid, 0) + alpha / (60 + rank)
    for rank, pid in enumerate(s):
        score[pid] = score.get(pid, 0) + (1 - alpha) / (60 + rank)

    top = sorted(score, key=score.get, reverse=True)[:k]  # highest fused score first
    bio = {u["_id"]: u["bio"] for u in users}
    return [(pid, round(score[pid], 5), bio[pid]) for pid in top]


def run_hybrid_search_demo(dense, sparse, users: list[dict]):
    """Section F: sweep alpha from pure-sparse to pure-dense and see the ranking shift."""

    mixed_query = "IIT Bombay 2019 Marathi Dadar, adventurous trekking type"  # facts AND vibes in one query

    for a in (0.0, 0.5, 1.0):
        label = "pure sparse" if a == 0 else "pure dense" if a == 1 else "balanced"
        print(f"\n---- alpha = {a}  ({label}) ----")
        show(hybrid_search(dense, sparse, users, mixed_query, alpha=a, k=3), star={"P0001"})


# ---------------------------------------------------------------------------
# SECTION G — SCALING: VECTOR COMPRESSION (QUANTIZATION)
# ---------------------------------------------------------------------------
# Full-precision vectors get expensive at scale. Product Quantization (PQ)
# compresses stored vectors by roughly 32x with a small accuracy cost.
# In Pinecone, this compression is fully managed - you write none of it -
# but it helps to understand what's happening under the hood.
# ---------------------------------------------------------------------------

def run_compression_math_demo():
    """Section G (part 1): back-of-envelope RAM math before/after quantization."""

    N, dim, fbytes = 100_000_000, 1024, 4   # llama-text-embed-v2 default: 1024 dims
    # fbytes = 4 because each dimension is stored as a 32-bit (4-byte) float
    print(f"{N:,} users x {dim} dims x {fbytes} B = {N * dim * fbytes / 1e9:,.0f} GB of RAM (uncompressed)")
    # Product quantization (PQ) can shrink storage by roughly 32x with small accuracy loss
    print(f"After ~32x product quantization: {N * dim * fbytes / 1e9 / 32:,.0f} GB")


def run_faiss_pq_peek():
    """
    Section G (part 2): optional "under the hood" peek at FAISS Product
    Quantization, for the curious. NOT used in the production Pinecone flow -
    Pinecone manages compression for you. Requires the optional
    'faiss-cpu' dependency; skipped gracefully if it's not installed.
    """

    try:
        import faiss
    except ImportError:
        print("faiss-cpu not installed - skipping the FAISS PQ peek (optional).")
        return

    X = np.random.rand(100_000, 128).astype("float32")  # 100k fake 128-dim vectors
    pq = faiss.IndexPQ(128, 16, 8)          # 16 chunks, 2^8 = 256 possible "types" per chunk
    pq.train(X)
    pq.add(X)                                # learn the "codebook", then compress+store X
    print(f"raw: {X.nbytes / 1e6:.0f} MB  ->  PQ codes: {100_000 * 16 / 1e6:.1f} MB  (32x)")


# ---------------------------------------------------------------------------
# SECTION H — EVALUATION METRICS (HOW GOOD IS THE SEARCH?)
# ---------------------------------------------------------------------------
# ranked : list of ids in the order the search engine returned them, e.g. ["A","B","C","D","E"]
# rel    : the SET of ids that are actually "relevant"/correct (ground truth), e.g. {"B","E","G"}
# k      : how many top results to consider
# ---------------------------------------------------------------------------

def precision_at_k(ranked, rel, k):
    """Of the top k results, what fraction are actually relevant?"""
    return sum(x in rel for x in ranked[:k]) / k


def recall_at_k(ranked, rel, k):
    """Of ALL relevant items, what fraction did we manage to surface in the top k?"""
    return sum(x in rel for x in ranked[:k]) / max(1, len(rel))


def mrr_at_k(ranked, rel, k):
    """
    Mean Reciprocal Rank: 1 / (rank of the FIRST relevant hit).
    Rewards finding a good match EARLY - 0 if there's no relevant hit in the top k.
    """
    return next((1 / i for i, x in enumerate(ranked[:k], 1) if x in rel), 0.0)


def ap_at_k(ranked, rel, k):
    """
    Average Precision: average of precision@i, evaluated only at each rank
    where a relevant item appears. Divides by the TOTAL number of relevant
    items (not just the ones found), so missed matches still hurt the score.
    """
    hits = ap = 0
    for i, x in enumerate(ranked[:k], 1):
        if x in rel:
            hits += 1
            ap += hits / i
    return ap / max(1, len(rel))


def ndcg_at_k(ranked, gains, k):
    """
    Normalized Discounted Cumulative Gain: like precision, but supports
    "graded" relevance (some matches are better than others) and discounts
    items that appear lower in the ranking.
    """
    dcg = sum(gains.get(x, 0) / math.log2(i + 1) for i, x in enumerate(ranked[:k], 1))
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(sorted(gains.values(), reverse=True)[:k], 1))
    return dcg / idcg if idcg else 0.0


def run_metrics_sanity_check():
    """Quick sanity check of the metric functions on a toy example, doable by hand."""

    card_ranked, card_rel = ["A", "B", "C", "D", "E"], {"B", "E", "G"}
    print(
        "TOY EXAMPLE -> "
        f"P@5={precision_at_k(card_ranked, card_rel, 5):.2f} "
        f"R@5={recall_at_k(card_ranked, card_rel, 5):.2f}  "
        f"MRR@5={mrr_at_k(card_ranked, card_rel, 5):.2f}  "
        f"AP@5={ap_at_k(card_ranked, card_rel, 5):.2f}"
    )
    print("Why does AP divide by 3, not 2? The relevant item we FAILED to show (G) still hurts the score.")


def run_metrics_on_app(dense, users: list[dict]):
    """
    Evaluate real search quality against ground truth we can compute directly
    (because we generated the synthetic user data ourselves). A real app
    would need human-labeled ground truth for this.
    """

    query3 = "non-smoker who loves dogs, based in Dadar"

    truly = {
        u["_id"] for u in users
        if u["smoke"] == "non-smoker" and "dog" in u["vibe"] and u["locality"] == "Dadar"
    }
    print(f"Ground truth: {len(truly)} truly compatible -> {sorted(truly)}\n")

    # Run plain dense search and see how it ranks against the ground truth
    stage1 = [pid for pid, _, _ in search(dense, query3, k=20)]
    first = next((i for i, x in enumerate(stage1, 1) if x in truly), ">20")  # rank of first true match
    print("Dense top-20:", stage1[:10], "...")
    print(
        f"P@5={precision_at_k(stage1, truly, 5):.2f}  R@10={recall_at_k(stage1, truly, 10):.2f}  "
        f"MRR@10={mrr_at_k(stage1, truly, 10):.2f}  |  first TRUE match at rank {first}"
    )

    return stage1, truly, query3


# ---------------------------------------------------------------------------
# SECTION I — RERANKING (TWO-STAGE RETRIEVAL)
# ---------------------------------------------------------------------------
# Stage 1 (retrieval): fetch a wider candidate set quickly, using dense search.
# Stage 2 (rerank): a hosted cross-encoder model re-reads the query against
# EACH candidate profile TOGETHER (much more expensive per-item, but much
# more accurate), then re-sorts the top candidates down to the best few.
# This is typically a single extra parameter on the same search call.
# ---------------------------------------------------------------------------

def run_reranking_demo(dense, stage1, truly, query3):
    """Section I: compare metrics before (stage1) vs after (stage2) reranking."""

    # Stage 1 (retrieval): fetch 25 candidates fast, using the same dense search as before.
    # Stage 2 (rerank): ask a hosted cross-encoder model to re-sort the top 25
    # down to the best 10. This single extra parameter is the whole diff.
    res = dense.search(
        namespace=NS,
        query={"inputs": {"text": query3}, "top_k": 25},      # Stage 1: fast retrieval
        fields=["bio"],
        rerank={
            "model": "bge-reranker-v2-m3",                     # Stage 2: hosted cross-encoder reranker
            "top_n": 10,
            "rank_fields": ["bio"],
        },
    )
    stage2 = [h["_id"] for h in res["result"]["hits"]]   # ids, now reranked

    # Compare the same metrics BEFORE (stage1, plain retrieval) vs AFTER (stage2, reranked)
    print(f"{'metric':<8}{'before':>9}{'after':>9}")
    for name, fn, k in [("MRR@10", mrr_at_k, 10), ("P@5", precision_at_k, 5), ("AP@10", ap_at_k, 10)]:
        print(f"{name:<8}{fn(stage1, truly, k):>9.2f}{fn(stage2, truly, k):>9.2f}")

    print("\nHybrid blends scores AT retrieval time. Reranking is a second pass AFTER retrieval.")


# ---------------------------------------------------------------------------
# SECTION J — PRODUCTION OPS: LATENCY & CACHING
# ---------------------------------------------------------------------------

def run_latency_demo(dense, n_calls: int = 60):
    """
    Section J (part 1): measure end-to-end search latency percentiles.

    Percentiles matter more than the average: p99 tells you how slow the
    UNLUCKIEST 1% of requests are - that's what "the tail" means, and it's
    usually what determines whether users perceive the product as fast.
    """

    lat = []   # will hold `n_calls` latency measurements, in milliseconds

    for _ in range(n_calls):
        t0 = time.perf_counter()
        search(dense, "dog lover in Dadar", k=10)
        lat.append((time.perf_counter() - t0) * 1000)  # convert seconds -> milliseconds

    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    print(f"end-to-end latency  p50={p50:.0f} ms   p95={p95:.0f} ms   p99={p99:.0f} ms")
    print("In production, don't hand-roll this: the Pinecone console charts QPS + percentiles live,")
    print("and teams typically pipe metrics to Prometheus/Datadog.")


def run_caching_demo(dense):
    """
    Section J (part 2): a minimal in-memory cache to avoid repeat searches.

    In a real production system this dict would typically be Redis instead.
    Gotchas to be aware of: cache invalidation (new users join -> stale
    cache) and near-duplicate queries ("matches near Dadar" vs "Dadar-area
    matches") not sharing a cache entry.
    """

    cache = {}   # in-memory cache: {normalized query text -> search results}

    def cached_search(text, k=5):
        key = text.strip().lower()  # normalize so "Dadar" and "dadar " hit the same cache entry
        if key not in cache:
            cache[key] = search(dense, text, k)  # cache MISS: actually call the search API
        return cache[key]                          # cache HIT: instant, no network call

    # First call: cache miss (has to embed + search over the network)
    t0 = time.perf_counter()
    cached_search("matches near Dadar")
    miss = (time.perf_counter() - t0) * 1000

    # Second call, same text: cache hit (just a dict lookup)
    t0 = time.perf_counter()
    cached_search("matches near Dadar")
    hit = (time.perf_counter() - t0) * 1000

    print(f"MISS {miss:8.2f} ms  (embed + search, over the network)")
    print(f"HIT  {hit:8.4f} ms  ->  {miss / max(hit, 1e-9):,.0f}x faster")


# ---------------------------------------------------------------------------
# SECTION K — WRAP-UP & CLEANUP
# ---------------------------------------------------------------------------

def cleanup_indexes(pc: Pinecone):
    """
    Delete both Pinecone indexes created by this script, to avoid ongoing
    storage/compute charges. Call this explicitly when you're done
    experimenting - it is NOT called automatically by main().
    """
    pc.delete_index(DENSE_INDEX_NAME)
    pc.delete_index(SPARSE_INDEX_NAME)
    print(f"Deleted indexes '{DENSE_INDEX_NAME}' and '{SPARSE_INDEX_NAME}'.")


# ---------------------------------------------------------------------------
# MAIN — run all sections in order
# ---------------------------------------------------------------------------

def main():
    # --- A: Setup & Data ---
    pc = connect_to_pinecone()
    users = build_user_profiles()

    # --- B: Build the Dense Index ---
    dense = create_dense_index(pc)
    upsert_dense(dense, users)

    # --- C: Basic Semantic Search ---
    run_semantic_search_demo(dense)

    # --- D: Metadata Filters ---
    run_metadata_filter_demo(dense)
    '''
    # --- E: Keyword Search (Sparse Index) ---
    sparse = create_sparse_index(pc, users)
    run_sparse_vs_dense_demo(dense, sparse)

    # --- F: Hybrid Search ---
    run_hybrid_search_demo(dense, sparse, users)

    # --- G: Scaling / Compression ---
    run_compression_math_demo()
    run_faiss_pq_peek()  # optional; skipped automatically if faiss-cpu isn't installed

    # --- H: Evaluation Metrics ---
    run_metrics_sanity_check()
    stage1, truly, query3 = run_metrics_on_app(dense, users)

    # --- I: Reranking ---
    run_reranking_demo(dense, stage1, truly, query3)

    # --- J: Production Ops (latency & caching) ---
    run_latency_demo(dense)
    run_caching_demo(dense)

    # --- K: Wrap-up ---
    print("\nAll sections complete.")
    print("Reminder: call cleanup_indexes(pc) to delete the Pinecone indexes and avoid charges.")
    # Uncomment the next line to delete both indexes automatically at the end of a run:
    # cleanup_indexes(pc)
    '''


if __name__ == "__main__":
    main()
