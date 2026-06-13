import numpy as np
from rank_bm25 import BM25Okapi


class HybridNewsStore:
    def __init__(self, dense_weight: float = 0.7):
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self.articles: list[dict] = []
        self._matrix: np.ndarray | None = None  # (N, D) L2-normalized
        self._bm25: BM25Okapi | None = None

    def build(self, articles: list[dict], embeddings: list[list[float]]) -> None:
        self.articles = articles

        matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = matrix / norms

        tokenized = [a["text"].lower().split() for a in articles]
        self._bm25 = BM25Okapi(tokenized)

    def search(
        self, query_embedding: list[float], query_text: str, top_k: int = 5
    ) -> list[dict]:
        if not self.articles or self._matrix is None or self._bm25 is None:
            return []

        # Dense cosine similarity via dot product (embeddings are already L2-normalized)
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm
        dense_scores: np.ndarray = self._matrix @ q_vec  # (N,)

        # BM25 sparse scores, normalized to [0, 1]
        sparse_scores = np.array(
            self._bm25.get_scores(query_text.lower().split()), dtype=np.float32
        )
        max_sparse = sparse_scores.max()
        if max_sparse > 0:
            sparse_scores /= max_sparse

        combined = self.dense_weight * dense_scores + self.sparse_weight * sparse_scores

        top_indices = np.argsort(combined)[::-1][:top_k]
        results = []
        for idx in top_indices:
            article = dict(self.articles[idx])
            article["score"] = float(combined[idx])
            results.append(article)
        return results
