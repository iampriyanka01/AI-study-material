# ============================================
# VECTOR SEARCH DEMO
# FAISS (Local)
# ============================================
#
# Demonstrates three FAISS index types for similarity search over text
# embeddings:
#   1. Flat  — exact brute-force search
#   2. HNSW  — graph-based approximate nearest neighbor (ANN) search
#   3. IVF   — cluster-based approximate nearest neighbor (ANN) search
#
# See README.md for setup instructions and a summary of tunable parameters.

# ---------------------------
# 1. INSTALL (Run once)
# ---------------------------
# pip install -r requirements.txt


# ---------------------------
# 2. IMPORTS
# ---------------------------
import numpy as np          # for array/vector handling
import faiss                # local vector search library
import time                 # (optional) useful for timing comparisons
from sentence_transformers import SentenceTransformer  # text -> embedding model

print("FAISS version:", faiss.__version__)


# ============================================
# PART 1 — SAMPLE DATASET + EMBEDDINGS
# ============================================

print("\n========== FAISS DEMO ==========\n")

# Small toy dataset — 5 short documents
texts = [
    "Document about AI simulations.",
    "Deep learning and neural networks.",
    "Vector databases and similarity search.",
    "Graph-based indexing methods.",
    "Clustering algorithms in machine learning."
]

# Load embedding model (384-dim output for all-MiniLM-L6-v2)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text -> vectors
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")  # FAISS requires float32

dimension = embeddings.shape[1]  # size of each embedding vector
print("Embedded", len(texts), "docs into", dimension, "-dim vectors")


# --------------------------------------------
# 1️⃣ FLAT INDEX (Exact Search)
# --------------------------------------------
print("\n---- FAISS: Flat Index ----")

# IndexFlatL2 = exact brute-force search using L2 (Euclidean) distance
flat_index = faiss.IndexFlatL2(dimension)
flat_index.add(embeddings)  # add all document vectors to the index

# Embed the query the same way as the documents
query = "AI systems"
query_vector = model.encode([query]).astype("float32")

k = 3  # number of nearest neighbors to retrieve (top-k).
# Larger k just returns more results / slightly more compute; it doesn't
# change per-result accuracy the way M, efSearch, nlist, nprobe do.

distances, indices = flat_index.search(query_vector, k)  # search against every vector

for i in indices[0]:
    print(texts[i])


# --------------------------------------------
# 2️⃣ HNSW INDEX (Graph-based ANN)
# --------------------------------------------
print("\n---- FAISS: HNSW Index ----")

# IndexHNSWFlat = graph-based ANN index (multi-layer navigable graph)
#
# PERFORMANCE-AFFECTING PARAMETERS:
# M (set to 32 below) = max number of neighbor links per node per layer.
#   - Bigger M  -> denser graph, higher recall, but MORE memory & SLOWER build.
#   - Smaller M -> sparser/"stringy" graph, less memory, faster build, but LOWER recall.
#   - Typical production range: M = 16-32.
hnsw_index = faiss.IndexHNSWFlat(dimension, 32)

# efConstruction = how many candidate nodes are explored while BUILDING the graph
# (i.e. when choosing each node's M neighbors during insertion).
#   - Bigger efConstruction  -> better-quality neighbor selection, higher recall, SLOWER build.
#   - Smaller efConstruction -> faster build, but may pick suboptimal neighbors -> LOWER recall.
# (Not set explicitly here, so FAISS uses its default; typical tuned value is ~100-200.)
# Example: hnsw_index.hnsw.efConstruction = 200

hnsw_index.add(embeddings)  # inserts vectors one by one, building the graph

# efSearch = how many candidate nodes are explored at QUERY time.
#   - Bigger efSearch  -> higher recall (closer to exact search), but SLOWER queries.
#   - Smaller efSearch -> faster queries, but may miss some true nearest neighbors.
#   - Rule of thumb: efSearch should be >= k (often 5-10x k for high recall).
# (Not set explicitly here, so FAISS uses its default.)
# Example: hnsw_index.hnsw.efSearch = 64

# Reuse the same query_vector from the Flat Index section above
distances, indices = hnsw_index.search(query_vector, k)

for i in indices[0]:
    print(texts[i])


# --------------------------------------------
# 3️⃣ IVF INDEX (Cluster-based ANN)
# --------------------------------------------
print("\n---- FAISS: IVF Index ----")

# PERFORMANCE-AFFECTING PARAMETERS:
# nlist = number of k-means clusters ("centroids") the vector space is split into.
#   - More clusters -> smaller lists per cluster -> FASTER fine search inside a list,
#     but SLOWER coarse search (comparing query to all centroids) and more memory
#     for centroid storage. Also needs enough data per cluster to train well.
#   - Fewer clusters  -> bigger lists -> slower fine search, but cheaper coarse search.
#   - Rule of thumb: nlist ~ sqrt(N) for large datasets (N = total vectors).
#   - Kept very small here (2) since we only have 5 toy documents.
nlist = 2

# The quantizer decides which cluster (centroid) each vector belongs to
quantizer = faiss.IndexFlatL2(dimension)
ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# IVF must be trained first (runs k-means to find cluster centroids).
# Training quality/time also scales with nlist (more clusters = more k-means work).
ivf_index.train(embeddings)
ivf_index.add(embeddings)  # assign each vector to its nearest centroid's list

# nprobe = number of nearest clusters to actually search at query time.
#   - Higher nprobe -> checks more clusters -> BETTER recall, but SLOWER queries
#     (more candidate vectors to compare against).
#   - Lower nprobe  -> fewer clusters checked -> faster queries, but risk of missing
#     the true nearest neighbor if it landed in a cluster you didn't probe.
#   - nprobe must be <= nlist. In production this is tuned against your latency SLA
#     and recall target (e.g. nprobe = 8-32 out of nlist = 1024).
ivf_index.nprobe = 1

distances, indices = ivf_index.search(query_vector, k)

for i in indices[0]:
    print(texts[i])


# ============================================
# SUMMARY OF TUNABLE PARAMETERS
# ============================================
# | Parameter        | Index         | Higher value ->                          | Lower value ->                             |
# |-------------------|--------------|-------------------------------------------|---------------------------------------------|
# | M                 | HNSW         | Better recall, more memory, slower build  | Less memory, faster build, lower recall     |
# | efConstruction    | HNSW (build) | Better neighbor quality, slower build     | Faster build, lower recall                  |
# | efSearch          | HNSW (query) | Better recall, slower queries             | Faster queries, lower recall                |
# | nlist             | IVF          | Faster fine search, slower coarse search, more memory | Slower fine search, cheaper coarse search |
# | nprobe            | IVF (query)  | Better recall, slower queries             | Faster queries, risk of missing neighbors   |
#
# In production, M / nlist are typically fixed at build time based on dataset
# size, while efSearch / nprobe are tuned per query to balance latency against
# recall.
