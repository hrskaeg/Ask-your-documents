"""Hybrid retrieval: combine vector search (Voyage + Chroma) with BM25 keyword search."""
from rank_bm25 import BM25Okapi
import re


def tokenize(text: str) -> list[str]:
    """Simple tokenizer that preserves § and other special characters as their own tokens."""
    # Lowercase, but keep § / numbers / Danish chars
    text = text.lower()
    # Split on whitespace and most punctuation, but keep § attached to following number
    # Replace § with " § " so it becomes its own token, then split
    text = text.replace("§", " § ")
    tokens = re.findall(r"[\wæøå§]+", text)
    return tokens


def build_bm25_index(collection):
    """Pull all docs from Chroma and build a BM25 index over them."""
    data = collection.get()
    docs = data["documents"]
    metadatas = data["metadatas"]
    ids = data["ids"]
    tokenized_corpus = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, docs, metadatas, ids


def hybrid_retrieve(
    question: str,
    voyage,
    collection,
    bm25,
    bm25_docs,
    bm25_metadatas,
    bm25_ids,
    k: int = 5,
    alpha: float = 0.5,
):
    """
    Retrieve top-k chunks combining vector and BM25 scores.

    alpha controls the weighting:
      alpha = 1.0  -> pure vector search
      alpha = 0.0  -> pure BM25
      alpha = 0.5  -> equal weight
    """
    # --- Vector search ---
    embed_result = voyage.embed([question], model="voyage-3", input_type="query")
    query_embedding = embed_result.embeddings[0]

    # Get more results than k so we have candidates to rerank
    n_candidates = max(k * 4, 20)
    vec_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
    )
    vec_ids = vec_results["ids"][0]
    vec_distances = vec_results["distances"][0]
    # Convert distance to similarity (1 - normalized_distance) so higher = better
    max_dist = max(vec_distances) if vec_distances else 1.0
    vec_scores = {
        doc_id: 1 - (dist / max_dist) for doc_id, dist in zip(vec_ids, vec_distances)
    }

    # --- BM25 search ---
    tokenized_query = tokenize(question)
    bm25_raw_scores = bm25.get_scores(tokenized_query)
    # Normalize BM25 scores to 0-1
    max_bm25 = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1.0
    bm25_scores = {
        doc_id: score / max_bm25
        for doc_id, score in zip(bm25_ids, bm25_raw_scores)
    }

    # --- Combine ---
    all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
    combined = {
        doc_id: alpha * vec_scores.get(doc_id, 0) + (1 - alpha) * bm25_scores.get(doc_id, 0)
        for doc_id in all_ids
    }

    # Top k
    top_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:k]

    # Build results in same shape as your existing retrieve() function
    id_to_idx = {doc_id: i for i, doc_id in enumerate(bm25_ids)}
    results = []
    for doc_id in top_ids:
        idx = id_to_idx[doc_id]
        results.append((
            bm25_docs[idx],
            bm25_metadatas[idx],
            1 - combined[doc_id],  # convert back to "distance" for display consistency
        ))
    return results
