"""
RAG Chunking & Retrieval Strategies Demo
=========================================

This script demonstrates two halves of a RAG (Retrieval-Augmented Generation)
pipeline using a mock "personal loan policy" document:

1. Chunking Techniques (splitting documents into pieces)
   - Fixed-size chunking
   - Hierarchical chunking
   - Semantic chunking

2. Retrieval Strategies (picking the right chunks for a query)
   - Vanilla Top-K
   - MMR (Maximum Marginal Relevance)
   - Reranking (by recency/year)

Run with:  python chunking_demo.py
"""

import re

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Make sure the NLTK sentence tokenizer data is available.
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def display_chunks(chunks, title):
    """Pretty-print a list of chunks with a title/header."""
    print(f"\n{title} -- Total Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---\n{chunk}")


# ---------------------------------------------------------------------------
# 1. Chunking Techniques
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
Section 1: Loan Eligibility
To qualify for a personal loan, the customer must be between 21 and 60 years old, have a minimum salary of ₹25,000, and a credit score above 700.

Section 2: Disbursal and Timelines
Loan disbursal typically occurs within 48 hours of approval. Delays can occur due to incomplete documentation.

Section 3: EMI Defaults
If the customer misses 2 or more EMIs, penalties apply. Further defaults may lead to legal action or freezing of accounts.

Section 4: Foreclosure
Customers can foreclose their loan after 6 EMIs have been paid. A foreclosure fee of 2% is applicable.
"""


def fixed_size_chunking(text, chunk_size=300, overlap=50):
    """
    Split text into fixed-size chunks with optional overlap (measured in tokens).
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def hierarchical_chunking(text):
    """
    Split document using structured section headers (e.g., "Section 1:", "Section 2:").
    """
    sections = re.split(r"\n(?=Section \d+:)", text.strip())
    return [sec.strip() for sec in sections if sec.strip()]


def semantic_chunking(text, model, threshold=0.75):
    """
    Group nearby sentences based on semantic similarity (using cosine similarity).
    """
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i]], [embeddings[i - 1]])[0][0]
        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# ---------------------------------------------------------------------------
# 2. Retrieval Strategies
# ---------------------------------------------------------------------------

def vanilla_top_k(chunks, chunk_embeddings, query_embedding, k=3):
    """Return the top-k chunks purely by cosine similarity to the query."""
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_k_idx = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k_idx], top_k_idx, similarities[top_k_idx]


def mmr(chunks, chunk_embeddings, query_embedding, k=3, lambda_param=0.7):
    """
    Return k chunks using Maximum Marginal Relevance (MMR), balancing
    relevance to the query with diversity from already-selected chunks.
    """
    similarities = cosine_similarity(chunk_embeddings, [query_embedding]).flatten()
    selected = []
    selected_idx = []
    candidate_idx = list(range(len(chunks)))

    for _ in range(k):
        if len(selected_idx) == 0:
            idx = int(np.argmax(similarities))
            selected_idx.append(idx)
            selected.append(chunks[idx])
            candidate_idx.remove(idx)
            continue

        # For each candidate, calculate max similarity with already selected
        diversity = np.array([
            max(
                cosine_similarity([chunk_embeddings[i]], [chunk_embeddings[j]])[0][0]
                for j in selected_idx
            )
            for i in candidate_idx
        ])
        mmr_score = lambda_param * similarities[candidate_idx] - (1 - lambda_param) * diversity
        idx = candidate_idx[int(np.argmax(mmr_score))]
        selected_idx.append(idx)
        selected.append(chunks[idx])
        candidate_idx.remove(idx)

    return selected


def rerank_by_year(top_k_idx, chunks_metadata):
    """Re-rank a set of retrieved chunk indices by 'year', newest first."""
    idx_years = [(idx, chunks_metadata[idx]["year"]) for idx in top_k_idx]
    reranked = sorted(idx_years, key=lambda x: x[1], reverse=True)
    return [chunks_metadata[idx]["text"] for idx, _ in reranked]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PART 1: DOCUMENT-SIDE STRATEGIES - CHUNKING TECHNIQUES")
    print("=" * 70)

    # --- Fixed-size chunking ---
    fixed_chunks = fixed_size_chunking(SAMPLE_TEXT, chunk_size=40, overlap=10)

    display_chunks(fixed_chunks, "Fixed-Size Chunking")

    # --- Hierarchical chunking ---
    hierarchical_chunks = hierarchical_chunking(SAMPLE_TEXT)
    display_chunks(hierarchical_chunks, "Hierarchical Chunking")

    # --- Semantic chunking ---
    print("\nLoading sentence-transformer model (this may take a moment on first run)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    semantic_chunks = semantic_chunking(SAMPLE_TEXT, model, threshold=0.75)
    display_chunks(semantic_chunks, "Semantic Chunking")

    

    print("\n" + "=" * 70)
    print("PART 2: RETRIEVAL-SIDE STRATEGIES - TOP-K, MMR, RERANKER")
    print("=" * 70)

    chunks = [
        "Applicants must be at least 21 years old.",
        "Minimum monthly salary is ₹25,000.",
        "Applicants must have a credit score above 700.",
        "Applicants must be at least 21 years old and earn ₹25,000.",
        "Minimum credit score of 700 is required.",
        "Personal loans are not available to self-employed applicants.",
        "EMI defaults attract a penalty of 2% per month.",
    ]

    query = "What are the age, salary, and credit score requirements for a personal loan?"

    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode([query])[0]

    vanilla_results, _, vanilla_scores = vanilla_top_k(chunks, chunk_embeddings, query_embedding, k=7)
    print("\nVanilla Top-K Results:")
    for text, score in zip(vanilla_results, vanilla_scores):
        print(f"  - {text}  [score: {score:.2f}]")

    mmr_results = mmr(chunks, chunk_embeddings, query_embedding, k=3, lambda_param=0.7)
    print("\nMMR Top-3 Results:")
    for text in mmr_results:
        print(f"  - {text}")
    

    print("\n" + "=" * 70)
    print("PART 3: RERANKING BY METADATA (RECENCY)")
    print("=" * 70)

    chunks_metadata = [
        {"text": "Applicants must be at least 21 years old.", "year": 2018, "type": "policy"},
        {"text": "Minimum monthly salary is ₹25,000.", "year": 2024, "type": "policy"},
        {"text": "Applicants must have a credit score above 650.", "year": 2024, "type": "policy"},
        {"text": "Applicants must be at least 21 years old and earn ₹25,000.", "year": 2019, "type": "policy"},
        {"text": "Minimum credit score of 700 is required.", "year": 2018, "type": "faq"},
        {"text": "Personal loans are not available to self-employed applicants.", "year": 2023, "type": "faq"},
        {"text": "EMI defaults attract a penalty of 2% per month.", "year": 2022, "type": "policy"},
    ]

    meta_chunks = [item["text"] for item in chunks_metadata]
    meta_chunk_embeddings = model.encode(meta_chunks)
    meta_query_embedding = model.encode([query])[0]

    vanilla_results, top_k_idx, vanilla_scores = vanilla_top_k(
        meta_chunks, meta_chunk_embeddings, meta_query_embedding, k=3
    )
    print("\nVanilla Top-3 Results (before re-ranking):")
    for text, score in zip(vanilla_results, vanilla_scores):
        print(f"  - {text}  [score: {score:.2f}]")

    mmr_results = mmr(meta_chunks, meta_chunk_embeddings, meta_query_embedding, k=3, lambda_param=0.7)
    print("\nMMR Top-3 Results:")
    for text in mmr_results:
        print(f"  - {text}")

    reranked_results = rerank_by_year(top_k_idx, chunks_metadata)
    print("\nRe-ranked Top-3 Results (by recency):")
    for text in reranked_results:
        print(f"  - {text}")


if __name__ == "__main__":
    main()
