"""
vector_search_faiss_pinecone.py

Vector Search Demo: FAISS (Local) + Pinecone (Cloud)
=====================================================
Converted from: PineCone_inclass.ipynb

This script demonstrates two approaches to vector similarity search:

  1. FAISS      - a local, in-memory vector search library (no internet / API key needed)
  2. Pinecone   - a managed cloud vector database (requires a free API key)

We embed a small set of text documents into vectors using a Sentence-Transformers
model, then run similarity search using different FAISS index types, and finally
show how to store/query the same data in Pinecone with metadata filtering.

HOW TO RUN (from VS Code / terminal)
-------------------------------------
1. Create and activate a virtual environment (recommended):
       python -m venv .venv
       .venv\\Scripts\\activate      (Windows)
       source .venv/bin/activate    (macOS/Linux)

2. Install dependencies:
       pip install -r requirements.txt

3. Set your Pinecone API key as an environment variable (Section 3 needs this).
   Get a free key at https://www.pinecone.io/
       Windows (PowerShell):  $env:PINECONE_API_KEY="your-key-here"
       macOS/Linux (bash):    export PINECONE_API_KEY="your-key-here"

   Alternatively, create a ".env" file next to this script containing:
       PINECONE_API_KEY=your-key-here
   (python-dotenv, included in requirements.txt, will auto-load it.)

4. Run the script:
       python vector_search_faiss_pinecone.py

   Section 2 (FAISS) always runs - no setup required.
   Section 3 (Pinecone) only runs if a valid PINECONE_API_KEY is found; otherwise
   it is skipped with a friendly message so the rest of the script still works.
"""

import os
import time

import numpy as np                                       # numerical arrays for embeddings
import faiss                                              # local vector search library (Facebook AI Similarity Search)
from sentence_transformers import SentenceTransformer     # turns text into embeddings

# python-dotenv lets us keep secrets in a local .env file instead of hardcoding them
from dotenv import load_dotenv

load_dotenv()  # loads variables from a ".env" file in the current directory, if present


# ---------------------------------------------------------------------------
# SECTION 2 — FAISS (Local Vector Search)
# ---------------------------------------------------------------------------
# FAISS lets you build and query vector indexes entirely on your own machine -
# no network calls, no API keys. We compare three index types on the same
# tiny dataset:
#
#   Flat  (IndexFlatL2)   -> brute-force exact search, compares query to every
#                            vector. Best for small datasets / ground truth.
#   HNSW  (IndexHNSWFlat) -> graph-based approximate search. Fast + accurate,
#                            a common production choice for large datasets.
#   IVF   (IndexIVFFlat)  -> clusters vectors, searches only nearby clusters.
#                            Trades a little accuracy for a lot of speed at scale.
# ---------------------------------------------------------------------------

def run_faiss_demo():
    """Run the full FAISS section: build embeddings, then query 3 index types."""

    print("\n========== FAISS DEMO ==========\n")

    # --- 2.1 Prepare the dataset and embeddings ----------------------------

    # A small toy dataset of documents to search over
    texts = [
        "Document about AI simulations.",
        "Deep learning and neural networks.",
        "Vector databases and similarity search.",
        "Graph-based indexing methods.",
        "Clustering algorithms in machine learning.",
    ]

    # Load a lightweight pretrained embedding model.
    # "all-MiniLM-L6-v2" maps sentences to 384-dimensional dense vectors.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert each text document into a vector embedding
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")  # FAISS requires float32

    # All FAISS indexes need to know the vector dimensionality up front
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    # Embed the search query the same way we embedded the documents.
    # This same query vector is reused across all three index types below so
    # the results are directly comparable.
    query = "AI systems"
    query_vector = model.encode([query]).astype("float32")

    k = 3  # number of nearest neighbors to retrieve

    # --- 2.2 Flat Index — Exact Search --------------------------------------
    # IndexFlatL2 computes the exact Euclidean (L2) distance between the query
    # and EVERY stored vector. Slow at scale, but guarantees the most accurate
    # results - good as a ground-truth baseline.
    print("---- FAISS: Flat Index ----")

    flat_index = faiss.IndexFlatL2(dimension)
    flat_index.add(embeddings)  # add our document vectors to the index

    distances, indices = flat_index.search(query_vector, k)  # returns distances + indices of k closest vectors
    for i in indices[0]:
        print(texts[i])

    # --- 2.3 HNSW Index — Graph-Based Approximate Search --------------------
    # IndexHNSWFlat builds a navigable small-world graph over the vectors.
    # Much faster than Flat search on large datasets while staying highly
    # accurate - a common production choice.
    # The "32" parameter controls the number of graph connections per node
    # (higher = more accurate but more memory).
    print("\n---- FAISS: HNSW Index ----")

    hnsw_index = faiss.IndexHNSWFlat(dimension, 32)
    hnsw_index.add(embeddings)

    distances, indices = hnsw_index.search(query_vector, k)  # reuse the same query vector from before
    for i in indices[0]:
        print(texts[i])

    # --- 2.4 IVF Index — Cluster-Based Search --------------------------------
    # IndexIVFFlat first clusters all vectors into `nlist` groups (via k-means),
    # then at search time only looks inside the closest clusters (`nprobe`
    # controls how many). Trades a little accuracy for a lot of speed on large
    # datasets.
    # NOTE: with only 5 documents and 2 clusters, this is purely illustrative -
    # IVF really shines with much larger datasets.
    print("\n---- FAISS: IVF Index ----")

    nlist = 2  # number of clusters to partition the vectors into

    quantizer = faiss.IndexFlatL2(dimension)              # IVF needs a "quantizer" index to assign vectors to clusters
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    ivf_index.train(embeddings)  # IVF indexes must be trained (to learn cluster centroids) before adding data
    ivf_index.add(embeddings)

    ivf_index.nprobe = 1  # how many clusters to search at query time (higher = more accurate, slower)

    distances, indices = ivf_index.search(query_vector, k)
    for i in indices[0]:
        print(texts[i])

    # Return the embedding model so Section 3 can reuse it without reloading
    return model


# ---------------------------------------------------------------------------
# SECTION 3 — Pinecone (Cloud Vector Database)
# ---------------------------------------------------------------------------
# Unlike FAISS, Pinecone is a managed cloud service - your vectors are stored
# remotely and queried over the network. This gives you persistence,
# scalability, and metadata filtering without managing infrastructure yourself.
#
# You need a free Pinecone API key: sign up at https://www.pinecone.io/ and
# set it as the PINECONE_API_KEY environment variable (see file header).
# ---------------------------------------------------------------------------

def run_pinecone_demo(model):
    """Run the full Pinecone section: create index, upsert docs, run 4 kinds of queries."""

    # Imported here (not at module level) so the script can still run the FAISS
    # section even in environments where the "pinecone" package is not installed.
    from pinecone import Pinecone, ServerlessSpec

    print("\n========== PINECONE DEMO ==========\n")

    # Read the API key from an environment variable instead of hardcoding it.
    # This keeps secrets out of source control (see .env / PINECONE_API_KEY above).
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY not set - skipping Pinecone section.")
        print("Set it as an environment variable or in a .env file to run this part.")
        return

    # Initialize the Pinecone client
    pc = Pinecone(api_key=api_key)

    index_name = "demo-index"

    # --- 3.1 Connect to Pinecone and create an index ------------------------
    # Only create the index if it doesn't already exist (avoids errors on re-run)
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,              # must match our embedding model's output size (MiniLM-L6-v2)
            metric="cosine",            # similarity metric: cosine, euclidean, or dotproduct
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # serverless = pay-per-use, no infra to manage
        )

    # Get a handle to the index so we can upsert/query it
    index = pc.Index(index_name)

    # --- 3.2 Upload (upsert) documents with metadata -------------------------
    # Each vector in Pinecone can carry arbitrary metadata (a dict of key/value
    # pairs) alongside it. This lets us filter search results later, e.g.
    # "only show me AI-related documents".
    pinecone_texts = [
        "AI is transforming healthcare",
        "Robotics in industrial automation",
        "Machine learning in finance",
        "AI for simulation and modeling",
        "Cloud computing basics",
    ]

    # Embed each document and upload it with metadata attached
    for i, text in enumerate(pinecone_texts):
        embedding = model.encode(text).tolist()  # Pinecone expects a plain list, not a numpy array

        # Metadata used later for filtering search results
        metadata = {
            "category": "ai" if "AI" in text else "other",  # simple category tag
            "created_at": int(time.time()),                 # unix timestamp of insertion
            "keywords": text.lower().split(),                # list of lowercase words for keyword filtering
        }

        # upsert = "insert or update" - each vector needs a unique ID
        index.upsert(vectors=[(f"doc-{i}", embedding, metadata)])

    print(f"Upserted {len(pinecone_texts)} documents into '{index_name}'")

    # --- 3.3 Basic Semantic Search --------------------------------------------
    # A plain nearest-neighbor query with no filters - returns the `top_k`
    # most semantically similar documents to the query.
    print("---- Pinecone: Semantic Search ----")

    query = "AI in simulations"
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=3,               # return the 3 closest matches
        include_metadata=True,  # also return each match's metadata dict
    )

    for match in results["matches"]:
        print(match["id"], match["score"])  # id = doc identifier, score = similarity score
        print(match["metadata"])
        print("-----")

    # --- 3.4 Filter by Category -------------------------------------------------
    # Metadata filters narrow results to only vectors whose metadata matches a
    # condition - here, category == "ai". This runs ALONGSIDE the similarity
    # search, so you still get semantically relevant results, just restricted
    # to a subset.
    print("\n---- Pinecone: Filter by Category ----")

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
        filter={
            "category": {"$eq": "ai"}   # only return docs where category equals "ai"
        },
    )

    for match in results["matches"]:
        print(match["id"], match["metadata"])

    # --- 3.5 Filter by Keyword ---------------------------------------------------
    # The $in operator checks whether the `keywords` metadata list contains any
    # of the given values - here, only documents whose keyword list includes
    # "finance" are returned.
    print("\n---- Pinecone: Filter by Keyword ----")

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
        filter={
            "keywords": {"$in": ["finance"]}   # only return docs that mention "finance"
        },
    )

    for match in results["matches"]:
        print(match["id"], match["metadata"])

    # --- 3.6 Combined Filter (AND logic) -----------------------------------------
    # Multiple filter conditions can be combined with $and to require that ALL
    # conditions match - here: category is "ai" AND the keyword list includes
    # "simulation".
    print("\n---- Pinecone: Combined Filter ----")

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
        filter={
            "$and": [
                {"category": {"$eq": "ai"}},           # condition 1: category must be "ai"
                {"keywords": {"$in": ["simulation"]}},  # condition 2: keywords must include "simulation"
            ]
        },
    )

    for match in results["matches"]:
        print(match["id"], match["metadata"])


# ---------------------------------------------------------------------------
# MAIN — run both sections in order
# ---------------------------------------------------------------------------

def main():
    embedding_model = run_faiss_demo()   # Section 2: always runs, no API key needed
    run_pinecone_demo(embedding_model)   # Section 3: needs PINECONE_API_KEY, else skipped

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()
