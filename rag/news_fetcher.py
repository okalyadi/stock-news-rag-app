import time
from datetime import datetime, timezone

import yfinance as yf


def _parse_article(ticker: str, item: dict) -> dict | None:
    # yfinance >=0.2.x nests metadata under a "content" key; older versions are flat
    content = item.get("content") or item

    title = content.get("title", "").strip()
    if not title:
        return None

    summary = content.get("summary") or content.get("description") or ""
    summary = summary.strip()

    # URL: new shape uses canonicalUrl dict, old shape uses "link"
    canonical = content.get("canonicalUrl") or {}
    url = (
        canonical.get("url")
        or content.get("url")
        or item.get("link")
        or ""
    )

    # Published timestamp: new shape is ISO string, old shape is unix int
    raw_time = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
    if isinstance(raw_time, (int, float)):
        published_at = datetime.fromtimestamp(raw_time, tz=timezone.utc).isoformat()
    else:
        published_at = str(raw_time or "")

    text = f"{ticker}: {title}. {summary}" if summary else f"{ticker}: {title}."

    return {
        "ticker": ticker,
        "title": title,
        "summary": summary,
        "url": url,
        "published_at": published_at,
        "text": text,
    }


def fetch_news(tickers: list[str]) -> list[dict]:
    articles: list[dict] = []
    for ticker in tickers:
        try:
            raw = yf.Ticker(ticker).news or []
            for item in raw:
                parsed = _parse_article(ticker, item)
                if parsed:
                    articles.append(parsed)
        except Exception as exc:
            print(f"[news_fetcher] Warning: could not fetch news for {ticker}: {exc}")
        time.sleep(0.1)
    return articles
