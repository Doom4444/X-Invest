# pipeline/rag_retriever.py
#
# UPDATED: Uses nomic-embed-text via Ollama instead of sentence-transformers.
#
# WHY NOMIC-EMBED-TEXT:
#   - Already running locally on your Ollama instance
#   - No extra Python dependency (removes sentence-transformers)
#   - nomic-embed-text supports multilingual text reasonably well
#   - Consistent: same Ollama server handles both chat AND embeddings
#
# HOW OLLAMA EMBEDDINGS WORK:
#   POST http://localhost:11434/api/embeddings
#   Body: { "model": "nomic-embed-text:latest", "prompt": "your text" }
#   Response: { "embedding": [0.123, -0.456, ...] }  ← 768-dim vector
#
# DISTANCE THRESHOLD:
#   nomic-embed-text produces 768-dim vectors (vs 384 for MiniLM).
#   Cosine distance range is still 0-2. Keep threshold at 1.2.

import requests
import chromadb
from config import OLLAMA_URL, EMBED_MODEL, CHROMA_PATH, COLLECTION_NAME

_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            _collection = None  # not ingested yet
    return _collection

def embed(text: str) -> list[float]:
    """
    Get embedding vector for a text string using nomic-embed-text via Ollama.
    Returns empty list on failure.
    """
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30
        )
        r.raise_for_status()
        return r.json()["embedding"]
    except Exception as e:
        print(f"[rag_retriever] Embedding error: {e}")
        return []

def retrieve(query: str, n_results: int = 3) -> list[dict]:
    """
    Find the most relevant financial concepts using MMR reranking.
    MMR balances relevance to the query with diversity between results.
    Returns [] if ChromaDB not ready or nothing relevant found.
    """
    try:
        col = _get_collection()
        if col is None:
            return []

        emb = embed(query)
        if not emb:
            return []

        # Fetch wider candidate pool for MMR to select from
        pool_size = min(20, col.count())
        res = col.query(
            query_embeddings=[emb],
            n_results=pool_size,
            include=["documents", "metadatas", "distances", "embeddings"]
        )

        docs       = res["documents"][0]
        metas      = res["metadatas"][0]
        distances  = res["distances"][0]
        embeddings = res["embeddings"][0]

        if not docs:
            return []

        # ── MMR reranking ──────────────────────────────────────────────────
        import numpy as np

        def cosine(a, b):
            a, b = np.array(a), np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        selected   = []
        candidates = list(range(len(docs)))
        lambda_    = 0.5  # 0 = pure diversity, 1 = pure relevance

        while len(selected) < n_results and candidates:
            if not selected:
                best = max(candidates, key=lambda i: cosine(emb, embeddings[i]))
            else:
                def mmr_score(i):
                    relevance  = cosine(emb, embeddings[i])
                    redundancy = max(cosine(embeddings[i], embeddings[s]) for s in selected)
                    return lambda_ * relevance - (1 - lambda_) * redundancy
                best = max(candidates, key=mmr_score)
            selected.append(best)
            candidates.remove(best)

        return [
            {"text": docs[i], "meta": metas[i], "distance": distances[i]}
            for i in selected
            if distances[i] < 1.2
        ]

    except Exception as e:
        print(f"[rag/retriever] Error: {e}")
        return []
