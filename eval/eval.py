"""
Evaluation script for the NewsRAG pipeline.

Usage:
    uv run python eval/eval.py --tickers AAPL NVDA TSLA MSFT
"""

import argparse
import json
import sys
from pathlib import Path

# Allow imports from the project root when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from rag.pipeline import NewsRAGPipeline  # noqa: E402 (after sys.path patch)


def build_questions(tickers: list[str]) -> list[dict]:
    t0 = tickers[0] if tickers else "AAPL"
    t1 = tickers[1] if len(tickers) > 1 else t0

    return [
        # happy_path — direct ticker mentions
        {"q": f"Why is {t0} moving?", "cat": "happy_path"},
        {"q": f"Latest news on {t1}?", "cat": "happy_path"},
        {"q": f"What's happening with {t0} stock today?", "cat": "happy_path"},
        {"q": f"Any recent announcements from {t0}?", "cat": "happy_path"},
        {"q": f"What is driving {t1}'s price action?", "cat": "happy_path"},
        # cross_ticker — spans multiple holdings
        {"q": f"What news is affecting {t0} and {t1}?", "cat": "cross_ticker"},
        {"q": f"Compare recent news for {t0} vs {t1}", "cat": "cross_ticker"},
        {"q": f"What do {t0} and {t1} have in common in the news?", "cat": "cross_ticker"},
        # semantic — concept-based, no ticker name
        {"q": "Any AI-related news in my holdings?", "cat": "semantic"},
        {"q": "Regulatory risks for my portfolio?", "cat": "semantic"},
        {"q": "Any earnings surprises in my holdings?", "cat": "semantic"},
        {"q": "What macroeconomic factors are affecting my stocks?", "cat": "semantic"},
        # out_of_corpus — should return nothing relevant
        {"q": "What did the Fed say about rates?", "cat": "out_of_corpus"},
        {"q": "Bitcoin news?", "cat": "out_of_corpus"},
        {"q": "NYC weather?", "cat": "out_of_corpus"},
    ]


def score_question(retrieved: list[dict], category: str) -> dict:
    top_score = retrieved[0]["score"] if retrieved else 0.0
    is_ooc = category == "out_of_corpus"

    if is_ooc:
        faithfulness = 1.0 if top_score < 0.30 else 0.0
        relevance = 1.0 if top_score <= 0.30 else 0.0
    else:
        faithfulness = min(1.0, top_score / 0.5)
        relevance = 1.0 if top_score > 0.30 else 0.0

    return {
        "top_score": top_score,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "retrieved_above_threshold": sum(1 for r in retrieved if r["score"] > 0.30),
    }


def print_table(rows: list[dict]) -> None:
    header = f"{'Question':<58}  {'Category':<15}  {'TopScore':>8}  {'Faith':>6}  {'Rel':>5}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        q = r["question"][:56] + ".." if len(r["question"]) > 58 else r["question"]
        print(
            f"{q:<58}  {r['category']:<15}  {r['top_score']:>8.4f}  "
            f"{r['faithfulness']:>6.2f}  {r['relevance']:>5.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the NewsRAG pipeline")
    parser.add_argument("--tickers", nargs="+", required=True, help="Stock tickers to index")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    questions = build_questions(tickers)

    print(f"Building index for: {', '.join(tickers)} ...")
    pipeline = NewsRAGPipeline()
    count = pipeline.build_index(tickers)
    print(f"Index built: {count} articles\n")

    results = []
    for item in questions:
        retrieved = pipeline.retrieve(item["q"], top_k=5)
        scores = score_question(retrieved, item["cat"])
        results.append({"question": item["q"], "category": item["cat"], **scores})

    print_table(results)

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["relevance"] for r in results) / len(results)
    print(f"\nAverages  —  faithfulness: {avg_faith:.3f}   relevance: {avg_rel:.3f}")

    output_path = Path(__file__).parent / "eval_results.json"
    payload = {
        "tickers": tickers,
        "questions": results,
        "averages": {"faithfulness": avg_faith, "relevance": avg_rel},
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
