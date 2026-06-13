from .embedder import embed_texts
from .news_fetcher import fetch_news
from .vector_store import HybridNewsStore

_SCORE_THRESHOLD = 0.30


class NewsRAGPipeline:
    def __init__(self):
        self._store = HybridNewsStore()
        self._built = False

    def build_index(self, tickers: list[str]) -> int:
        articles = fetch_news(tickers)
        if not articles:
            return 0
        embeddings = embed_texts([a["text"] for a in articles])
        self._store.build(articles, embeddings)
        self._built = True
        return len(articles)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._built:
            return []
        query_emb = embed_texts([query])[0]
        return self._store.search(query_emb, query, top_k)

    def format_context(self, results: list[dict]) -> str | None:
        relevant = [r for r in results if r["score"] >= _SCORE_THRESHOLD]
        if not relevant:
            return None

        lines = []
        for i, article in enumerate(relevant, 1):
            lines.append(f"[{i}] {article['text']}")

        lines.append(
            "\nCite articles by number (e.g. [1], [2]). "
            "Do not reference or invent any news not listed above."
        )
        return "\n".join(lines)
