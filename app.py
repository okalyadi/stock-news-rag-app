import os

import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

from rag.embedder import is_configured as nebius_ok
from rag.pipeline import NewsRAGPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Why Is My Stock Moving?",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a financial news analyst. Your job is to explain stock price movements using provided news articles.

Rules you must follow — no exceptions:
1. Only use information from the provided news articles in your answer.
2. Cite every factual claim with the article number, e.g. [1], [2].
3. If news articles ARE provided, always give an answer grounded in those articles — even if they only partially address the question. Summarise what the articles reveal about the stocks or market, and note any gaps.
4. If NO news context is provided at all, respond exactly:
   "I couldn't find relevant news for that question in the current index."
5. Never invent market events, price targets, earnings figures, analyst upgrades, or any financial data not present in the articles.
6. Be concise and specific."""

SUGGESTED_QUESTIONS = [
    "Why is my stock moving today?",
    "What news is affecting my tech stocks?",
    "Any earnings surprises in my holdings?",
    "What regulatory risks should I be aware of?",
    "Any AI-related developments in my portfolio?",
]

# ── Session state init ────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "pipeline": NewsRAGPipeline(),
        "article_count": 0,
        "messages": [],          # list of {"role", "content", "sources"?}
        "pending_question": None,
        "tickers": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _groq_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def _stream_groq(messages: list[dict]):
    """Yield text chunks from Groq; raises on missing key."""
    from groq import Groq  # lazy import so app starts without it installed

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=1024,
        temperature=0.2,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _build_llm_messages(history: list[dict], prompt: str, context: str | None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include the last 8 turns (4 exchanges) for conversational context
    for msg in history[-8:]:
        msgs.append({"role": msg["role"], "content": msg["content"]})

    if context:
        user_content = f"News context:\n{context}\n\nUser question: {prompt}"
    else:
        user_content = prompt

    msgs.append({"role": "user", "content": user_content})
    return msgs


# ── Stock data & chart helpers ────────────────────────────────────────────────
_EXCHANGE_MAP = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NNM": "NASDAQ",
    "NYQ": "NYSE",   "PCX": "NYSE",   "ASE": "NYSE",
}


@st.cache_data(ttl=300)
def _fetch_stock_info(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "name":         info.get("shortName") or info.get("longName") or ticker,
        "exchange":     info.get("exchange", ""),
        "currency":     info.get("currency", "USD"),
        "current":      info.get("currentPrice") or info.get("regularMarketPrice"),
        "prev_close":   info.get("previousClose") or info.get("regularMarketPreviousClose"),
        "day_high":     info.get("dayHigh") or info.get("regularMarketDayHigh"),
        "day_low":      info.get("dayLow") or info.get("regularMarketDayLow"),
        "week52_high":  info.get("fiftyTwoWeekHigh"),
        "week52_low":   info.get("fiftyTwoWeekLow"),
        "market_cap":   info.get("marketCap"),
        "volume":       info.get("volume") or info.get("regularMarketVolume"),
        "avg_volume":   info.get("averageVolume"),
        "pe_ratio":     info.get("trailingPE"),
    }


def _fmt_price(val: float | None, currency: str = "USD") -> str:
    if val is None:
        return "—"
    sym = "$" if currency == "USD" else f"{currency} "
    return f"{sym}{val:,.2f}"


def _fmt_mcap(val: float | None) -> str:
    if val is None:
        return "—"
    if val >= 1e12:
        return f"${val / 1e12:.2f}T"
    if val >= 1e9:
        return f"${val / 1e9:.2f}B"
    return f"${val / 1e6:.2f}M"


def _fmt_vol(val: float | None) -> str:
    if val is None:
        return "—"
    if val >= 1e6:
        return f"{val / 1e6:.2f}M"
    if val >= 1e3:
        return f"{val / 1e3:.1f}K"
    return f"{val:,.0f}"


def _pct_delta(current: float | None, prev: float | None) -> str | None:
    if not current or not prev or prev == 0:
        return None
    return f"{((current - prev) / prev) * 100:+.2f}%"


@st.cache_data(ttl=120)
def _fetch_market_data() -> dict:
    """Fetch today's OHLC for major indices + VIX in one yfinance download."""
    symbols = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^DJI":  "Dow Jones",
        "^RUT":  "Russell 2000",
        "^VIX":  "VIX",
    }
    try:
        raw = yf.download(
            list(symbols.keys()), period="5d", interval="1d",
            auto_adjust=True, progress=False, multi_level_index=True,
        )
        result = {}
        for sym, name in symbols.items():
            try:
                closes = raw["Close"][sym].dropna()
                highs  = raw["High"][sym].dropna()
                lows   = raw["Low"][sym].dropna()
                result[name] = {
                    "symbol":  sym,
                    "current": float(closes.iloc[-1]),
                    "prev":    float(closes.iloc[-2]) if len(closes) > 1 else None,
                    "day_high": float(highs.iloc[-1]),
                    "day_low":  float(lows.iloc[-1]),
                }
            except Exception:
                result[name] = None
        return result
    except Exception:
        return {}


@st.cache_data(ttl=300)
def _fetch_vix_detail() -> dict:
    """52-week range + 1-year history for VIX chart."""
    try:
        info = yf.Ticker("^VIX").info
        hist = yf.download("^VIX", period="1y", interval="1d",
                           auto_adjust=True, progress=False)
        close = hist["Close"].squeeze().dropna()
        history = close.to_frame(name="vix").reset_index().rename(columns={"Date": "date"})
        return {
            "week52_high": info.get("fiftyTwoWeekHigh"),
            "week52_low":  info.get("fiftyTwoWeekLow"),
            "history": history,
        }
    except Exception:
        return {}


def _vix_label(vix: float | None) -> str:
    if vix is None:
        return ""
    if vix < 15:
        return "😌  Low fear — market is calm"
    if vix < 20:
        return "🟡  Moderate — normal volatility"
    if vix < 30:
        return "🟠  Elevated fear — some uncertainty"
    return "🔴  High fear — extreme volatility"


def _tv_url(widget: str, **params) -> str:
    """Build a TradingView embed-widget iframe URL."""
    import json
    base = f"https://s.tradingview.com/embed-widget/{widget}/?locale=en"
    return f"{base}#{json.dumps(params)}"


def _tradingview_chart(ticker: str, exchange: str = "", height: int = 520) -> None:
    import urllib.parse
    symbol = f"{exchange}:{ticker}" if exchange else ticker
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": "D",
        "timezone": "America/New_York",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "allow_symbol_change": "false",
        "calendar": "false",
        "hide_side_toolbar": "false",
        "withdateranges": "1",
        "save_image": "1",
        "support_host": "https://www.tradingview.com",
    })
    url = f"https://s.tradingview.com/widgetembed/?{params}"
    st.iframe(url, height=height)


def _fetch_benzinga_catalyst(ticker: str) -> dict:
    import requests
    from bs4 import BeautifulSoup
    try:
        url = f"https://www.benzinga.com/quote/{ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)[:3000]

        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = (
            f"What recent news or catalyst is driving {ticker} stock today? "
            "Return a one-sentence summary, then up to 2 recent headlines verbatim. "
            "Just the data — no commentary.\n\n"
            f"Page content:\n{text}"
        )
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        result = response.choices[0].message.content.strip()
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        return {"catalyst": lines[0] if lines else None, "headlines": lines[1:3]}
    except Exception:
        return {"catalyst": None, "headlines": []}


def _run_gapper_scan() -> dict:
    from datetime import datetime
    from yfinance.screener.screener import screen, EquityQuery

    query = EquityQuery("and", [
        EquityQuery("gt",  ["percentchange", 5]),
        EquityQuery("eq",  ["region", "us"]),
        EquityQuery("gte", ["intradaymarketcap", 100_000_000]),
        EquityQuery("gt",  ["intradayprice", 3]),
        EquityQuery("gt",  ["dayvolume", 50_000]),
    ])
    result = screen(query, sortField="dayvolume", sortAsc=False, size=50)
    quotes = result.get("quotes", [])

    top10 = quotes[:10]

    gappers = []
    for i, q in enumerate(top10):
        ticker  = q.get("symbol", "")
        price   = q.get("preMarketPrice")         or q.get("regularMarketPrice")         or 0
        gap_pct = q.get("preMarketChangePercent")  or q.get("regularMarketChangePercent")  or 0
        volume  = q.get("preMarketVolume")         or q.get("regularMarketVolume")         or 0
        mkt_cap = q.get("marketCap")               or 0

        cat = _fetch_benzinga_catalyst(ticker)
        gappers.append({
            "rank":             i + 1,
            "symbol":           ticker,
            "price":            round(float(price), 2),
            "gap_pct":          round(float(gap_pct), 2),
            "premarket_volume": int(volume),
            "market_cap":       _fmt_mcap(mkt_cap),
            "catalyst":         cat["catalyst"],
            "headlines":        cat["headlines"],
        })

    return {"scanned_at": datetime.now().isoformat(), "gappers": gappers}


def _run_trending_scan(direction: str) -> dict:
    """direction: 'up' or 'down'"""
    from datetime import datetime
    from yfinance.screener.screener import screen, EquityQuery

    change_filter = EquityQuery("gt", ["percentchange",  5]) if direction == "up" \
               else EquityQuery("lt", ["percentchange", -5])

    query = EquityQuery("and", [
        change_filter,
        EquityQuery("eq",  ["region", "us"]),
        EquityQuery("gt",  ["intradayprice", 1]),
        EquityQuery("gt",  ["avgdailyvol3m", 1_000_000]),
        EquityQuery("gt",  ["dayvolume",     1_000_000]),
    ])
    result = screen(query, sortField="dayvolume", sortAsc=False, size=50)
    quotes = result.get("quotes", [])

    # Post-filter: relative volume > 1.5 and ATR >= 1
    tickers_for_atr = [q.get("symbol", "") for q in quotes if q.get("symbol")][:20]
    atr_map: dict[str, float] = {}
    if tickers_for_atr:
        try:
            hist = yf.download(
                tickers_for_atr, period="14d", interval="1d",
                auto_adjust=True, progress=False, multi_level_index=True,
            )
            for sym in tickers_for_atr:
                try:
                    high  = hist["High"][sym].dropna()
                    low   = hist["Low"][sym].dropna()
                    close = hist["Close"][sym].dropna()
                    tr = (high - low).combine(
                        (high - close.shift(1)).abs(), max
                    ).combine(
                        (low  - close.shift(1)).abs(), max
                    )
                    atr_map[sym] = float(tr.rolling(14).mean().iloc[-1])
                except Exception:
                    atr_map[sym] = 0.0
        except Exception:
            pass

    # Intraday time adjustment for RVOL (Finviz-style)
    # RVOL = current_volume / (avg_daily_volume × fraction_of_session_elapsed)
    # This prevents early-session stocks looking low-RVOL vs full-day average.
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    elapsed  = (now_et - market_open).total_seconds()
    session  = (market_close - market_open).total_seconds()
    day_frac = max(0.05, min(1.0, elapsed / session))  # clamp to [5%, 100%]

    filtered = []
    for q in quotes:
        sym     = q.get("symbol", "")
        price   = float(q.get("regularMarketPrice") or 0)
        chg_pct = float(q.get("regularMarketChangePercent") or 0)
        vol     = float(q.get("regularMarketVolume") or 0)
        avg_vol = float(q.get("averageDailyVolume3Month") or 0)
        mkt_cap = q.get("marketCap") or 0

        # Time-adjusted expected volume for the elapsed portion of the session
        expected_vol = avg_vol * day_frac
        rel_vol = (vol / expected_vol) if expected_vol > 0 else 0
        atr     = atr_map.get(sym, 999.0)  # if ATR not fetched, don't block

        if rel_vol >= 1.5 and atr >= 1.0:
            try:
                has_options = len(yf.Ticker(sym).options) > 0
            except Exception:
                has_options = False
            if has_options:
                filtered.append({
                    "symbol":      sym,
                    "price":       round(price, 2),
                    "chg_pct":     round(chg_pct, 2),
                    "volume":      int(vol),
                    "avg_volume":  _fmt_vol(avg_vol),
                    "rvol":        round(rel_vol, 2),
                    "atr":         round(atr, 2),
                    "market_cap":  _fmt_mcap(mkt_cap),
                })

    filtered.sort(key=lambda x: x["volume"], reverse=True)
    top10 = filtered[:10]
    for i, g in enumerate(top10):
        g["rank"] = i + 1

    return {"scanned_at": datetime.now().isoformat(), "direction": direction, "stocks": top10}


def _render_sources(sources: list[dict]) -> None:
    with st.expander("Sources", expanded=False):
        if not sources:
            st.caption("No articles cleared the relevance threshold.")
            return
        for article in sources:
            col_badge, col_body = st.columns([1, 9])
            col_badge.markdown(f"**`{article['ticker']}`**")
            with col_body:
                title = article["title"] or "(no title)"
                url = article.get("url", "")
                if url:
                    st.markdown(f"[{title}]({url})")
                else:
                    st.markdown(f"**{title}**")
                st.caption(f"Relevance score: {article['score']:.3f}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Why Is My Stock Moving?")
    st.caption("RAG-powered stock news assistant")
    st.divider()

    if not nebius_ok():
        st.warning(
            "**NEBIUS_API_KEY not set.**\n\n"
            "Add it to your `.env` file to enable embeddings and index building.",
            icon="⚠️",
        )

    if not _groq_configured():
        st.warning(
            "**GROQ_API_KEY not set.**\n\n"
            "Add it to your `.env` file to enable chat responses.",
            icon="⚠️",
        )

    st.subheader("Stock Tickers")
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        placeholder="AAPL, NVDA, TSLA, MSFT",
        label_visibility="collapsed",
    )

    build_disabled = not nebius_ok()
    if st.button("Build Index", type="primary", disabled=build_disabled, use_container_width=True):
        if not tickers_input.strip():
            st.warning("Please enter at least one stock ticker (e.g. AAPL, TSLA).")
        raw_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if raw_tickers:
            with st.spinner(f"Fetching & embedding news for {', '.join(raw_tickers)} …"):
                try:
                    count = st.session_state.pipeline.build_index(raw_tickers)
                    st.session_state.article_count = count
                    st.session_state.tickers = raw_tickers
                    st.session_state.messages = []  # reset chat for new index
                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    if st.session_state.article_count:
        st.success(
            f"Index ready — **{st.session_state.article_count}** articles"
            + (f" ({', '.join(st.session_state.tickers)})" if st.session_state.tickers else ""),
            icon="✅",
        )
    elif nebius_ok():
        st.info("Enter tickers above and click **Build Index** to start.", icon="ℹ️")

    st.divider()
    st.caption("Powered by Nebius · Groq · yfinance")
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This app aggregates publicly available market data and news "
        "from third-party sources for informational purposes only. Nothing here constitutes "
        "financial advice or a recommendation to buy or sell any security. Always do your "
        "own research before making investment decisions."
    )

# ── Main area: tabs ───────────────────────────────────────────────────────────
tab_chat, tab_browse, tab_overview, tab_pulse, tab_gappers, tab_trending = st.tabs(
    ["💬 Chat", "🔍 Browse Articles", "📊 Stock Overview", "📅 Market Pulse", "🚀 Premarket Gappers", "📈 Trending Stocks"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    index_ready = st.session_state.article_count > 0
    chat_enabled = index_ready and _groq_configured()

    if not index_ready:
        st.info("Build the news index first using the sidebar.", icon="👈")

    # Suggested question buttons (shown when index is ready)
    if index_ready:
        st.markdown("**Suggested questions**")
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i].button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_question = q

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_sources(msg.get("sources", []))

    # Resolve active prompt (typed or suggested)
    typed_prompt = st.chat_input(
        "Ask about your stocks…",
        disabled=not chat_enabled,
    )
    active_prompt = typed_prompt or st.session_state.pending_question
    if st.session_state.pending_question:
        st.session_state.pending_question = None  # consume it

    # Process prompt
    if active_prompt:
        if not _groq_configured():
            st.error("GROQ_API_KEY is not set — cannot generate a response.")
        elif not index_ready:
            st.error("Build the index first.")
        else:
            # Show user message
            with st.chat_message("user"):
                st.markdown(active_prompt)
            st.session_state.messages.append({"role": "user", "content": active_prompt})

            # Retrieve + format context
            retrieved = st.session_state.pipeline.retrieve(active_prompt, top_k=5)
            context = st.session_state.pipeline.format_context(retrieved)
            sources = [r for r in retrieved if r["score"] >= 0.30]

            # Build LLM messages (history excludes the message we just appended)
            llm_messages = _build_llm_messages(
                st.session_state.messages[:-1], active_prompt, context
            )

            # Stream assistant response
            with st.chat_message("assistant"):
                try:
                    full_response = st.write_stream(_stream_groq(llm_messages))
                except Exception as exc:
                    full_response = f"Error generating response: {exc}"
                    st.error(full_response)
                _render_sources(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "sources": sources}
            )
            st.rerun()

    # Clear chat button
    if st.session_state.messages:
        st.divider()
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# BROWSE TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_browse:
    if not st.session_state.article_count:
        st.info("Build the news index first using the sidebar.", icon="👈")
    else:
        col_search, col_k = st.columns([4, 1])
        with col_search:
            browse_query = st.text_input(
                "Search articles",
                placeholder="e.g. AI chips, earnings beat, regulatory",
                label_visibility="collapsed",
            )
        with col_k:
            top_k = st.slider("Top-k", min_value=1, max_value=20, value=5, label_visibility="collapsed")

        if browse_query:
            if not nebius_ok():
                st.error("NEBIUS_API_KEY is required for search.")
            else:
                with st.spinner("Searching …"):
                    try:
                        results = st.session_state.pipeline.retrieve(browse_query, top_k=top_k)
                    except Exception as exc:
                        st.error(f"Search failed: {exc}")
                        results = []

                if not results:
                    st.warning("No articles found.")
                else:
                    st.markdown(f"**{len(results)} results** for *{browse_query}*")

                    for article in results:
                        with st.container(border=True):
                            badge_col, title_col = st.columns([1, 10])
                            badge_col.markdown(
                                f"<span style='background:#1f77b4;color:white;padding:2px 8px;"
                                f"border-radius:4px;font-weight:bold;font-size:0.85rem'>"
                                f"{article['ticker']}</span>",
                                unsafe_allow_html=True,
                            )
                            url = article.get("url", "")
                            title = article["title"] or "(no title)"
                            with title_col:
                                if url:
                                    st.markdown(f"**[{title}]({url})**")
                                else:
                                    st.markdown(f"**{title}**")

                            summary = article.get("summary", "")
                            if summary:
                                preview = summary[:220] + "…" if len(summary) > 220 else summary
                                st.caption(preview)

                            score = article["score"]
                            st.progress(
                                min(1.0, max(0.0, score)),
                                text=f"Relevance: {score:.3f}",
                            )

                    # Show the exact context block the LLM would receive
                    context_preview = st.session_state.pipeline.format_context(results)
                    st.divider()
                    st.markdown("**Context block sent to the LLM**")
                    if context_preview:
                        st.code(context_preview, language="text")
                    else:
                        st.info(
                            "No articles cleared the 0.15 relevance threshold — "
                            "the model would receive no context and decline to answer.",
                            icon="ℹ️",
                        )
        else:
            st.caption("Enter a search query above to browse indexed articles.")

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if not st.session_state.tickers:
        st.info("Build the news index first using the sidebar.", icon="👈")
    else:
        selected = st.selectbox(
            "Ticker", st.session_state.tickers, label_visibility="collapsed", key="overview_ticker"
        )

        with st.spinner(f"Loading {selected} market data…"):
            try:
                info = _fetch_stock_info(selected)
            except Exception as exc:
                st.error(f"Could not load data for {selected}: {exc}")
                info = {}

        if info:
            currency = info["currency"]
            current  = info["current"]
            delta    = _pct_delta(current, info["prev_close"])

            st.subheader(f"{info['name']}  ·  {selected}")
            st.divider()

            # ── Row 1: price + key stats ──────────────────────────────────────
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Current Price",  _fmt_price(current, currency),     delta)
            c2.metric("Market Cap",     _fmt_mcap(info["market_cap"]))
            c3.metric("Today's High",   _fmt_price(info["day_high"],   currency))
            c4.metric("Today's Low",    _fmt_price(info["day_low"],    currency))
            c5.metric("52-Week High",   _fmt_price(info["week52_high"], currency))
            c6.metric("52-Week Low",    _fmt_price(info["week52_low"],  currency))

            # ── Row 2: volume + pe ────────────────────────────────────────────
            c7, c8, c9, *_ = st.columns(6)
            c7.metric("Volume",      _fmt_vol(info["volume"]))
            c8.metric("Avg Volume",  _fmt_vol(info["avg_volume"]))
            c9.metric("P/E (TTM)",   f"{info['pe_ratio']:.1f}" if info["pe_ratio"] else "—")

            st.divider()

            # ── TradingView chart ─────────────────────────────────────────────
            exchange = _EXCHANGE_MAP.get(info["exchange"], "")
            _tradingview_chart(selected, exchange=exchange)

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET PULSE TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pulse:

    # ── Market indices row ────────────────────────────────────────────────────
    st.subheader("Market Snapshot")
    with st.spinner("Loading market data…"):
        mkt = _fetch_market_data()

    if mkt:
        cols = st.columns(len(mkt))
        for col, (name, data) in zip(cols, mkt.items()):
            if data:
                delta = _pct_delta(data["current"], data["prev"])
                val = f"{data['current']:.2f}" if name == "VIX" else f"{data['current']:,.2f}"
                col.metric(name, val, delta)
            else:
                col.metric(name, "—")

    st.divider()

    # ── VIX detail ────────────────────────────────────────────────────────────
    st.subheader("CBOE Volatility Index (VIX)")
    vix_data = mkt.get("VIX") if mkt else None
    with st.spinner("Loading VIX history…"):
        vix_detail = _fetch_vix_detail()

    if vix_data:
        vix_52h = vix_detail.get("week52_high")
        vix_52l = vix_detail.get("week52_low")
        v = vix_data["current"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VIX",      f"{v:.2f}",           _pct_delta(v, vix_data["prev"]))
        c2.metric("Day High", f"{vix_data['day_high']:.2f}" if vix_data["day_high"] else "—")
        c3.metric("Day Low",  f"{vix_data['day_low']:.2f}"  if vix_data["day_low"]  else "—")
        c4.metric("52W Range",
                  f"{vix_52l:.1f} – {vix_52h:.1f}" if vix_52l and vix_52h else "—")
        st.caption(_vix_label(v))

    # Altair area chart with fear-level reference lines
    hist_df = vix_detail.get("history")
    if hist_df is not None and not hist_df.empty:
        import altair as alt
        import pandas as pd

        area = (
            alt.Chart(hist_df)
            .mark_area(line={"color": "#e05252", "strokeWidth": 1.5},
                       color="#e05252", opacity=0.25)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("vix:Q", title="VIX", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("vix:Q", format=".2f", title="VIX"),
                ],
            )
        )
        rules_df = pd.DataFrame({"level": [15, 20, 30], "label": ["Low fear", "Elevated", "Extreme"]})
        rules = (
            alt.Chart(rules_df)
            .mark_rule(strokeDash=[4, 4])
            .encode(
                y="level:Q",
                color=alt.Color("label:N", scale=alt.Scale(
                    domain=["Low fear", "Elevated", "Extreme"],
                    range=["#2ecc71", "#f39c12", "#e74c3c"],
                )),
            )
        )
        st.altair_chart((area + rules).properties(height=280), use_container_width=True)

    st.divider()

    # ── Top 10 movers ─────────────────────────────────────────────────────────
    st.subheader("Today's Top 10 Movers")
    st.caption("Powered by TradingView · US equities")
    st.iframe(
        _tv_url(
            "hotlists",
            colorTheme="light",
            exchange="US",
            showChart=True,
            locale="en",
            isTransparent=False,
            showSymbolLogo=True,
            showFloatingTooltip=False,
        ),
        height=800,
    )

    st.divider()

    # ── Economic calendar ─────────────────────────────────────────────────────
    st.subheader("Economic Calendar")
    st.caption("Key macro events for US, EU, GB, JP, AU, CA")
    st.iframe(
        _tv_url(
            "events",
            colorTheme="light",
            isTransparent=False,
            locale="en",
            importanceFilter="-1,0,1",
            countryFilter="us,eu,gb,jp,au,ca",
        ),
        height=520,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PREMARKET GAPPERS TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gappers:
    import json
    from datetime import date

    st.subheader("Premarket Gappers Scanner")

    if st.button("🔍 Scan Now", type="primary"):
        with st.spinner("Scanning premarket gappers and fetching catalysts… (60–90 s)"):
            try:
                result = _run_gapper_scan()
                st.session_state["gappers_result"] = result
                today = date.today().isoformat()
                with open(f"premarket_gappers_{today}.json", "w") as _f:
                    json.dump(result, _f, indent=2)
            except Exception as _exc:
                st.error(f"Scan failed: {_exc}")
                result = None
    else:
        result = st.session_state.get("gappers_result")

    if result:
        gappers = result["gappers"]
        scanned_at = result["scanned_at"][:19].replace("T", " ")

        if not gappers:
            st.info("No gappers matched the filters right now. Try again during pre-market hours (4–9:30 AM ET).", icon="ℹ️")
        else:
            top3 = gappers[:3]
            summary = ", ".join(
                f"{g['symbol']} ({g['gap_pct']:+.1f}%) — {g['catalyst'] or 'no catalyst'}"
                for g in top3
            )
            st.success(f"**Premarket Gappers: {len(gappers)} names.** Top: {summary}")
            st.caption(f"Scanned at {scanned_at}")
            st.divider()

            for g in gappers:
                with st.expander(
                    f"#{g['rank']}  {g['symbol']}  ·  {g['gap_pct']:+.1f}%  ·  ${g['price']:.2f}",
                    expanded=(g["rank"] == 1),
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Gap %", f"{g['gap_pct']:+.1f}%")
                    c2.metric("Price", f"${g['price']:.2f}")
                    c3.metric("Mkt Cap", g["market_cap"])
                    c4.metric("Pre-mkt Vol", _fmt_vol(g["premarket_volume"]))
                    if g.get("catalyst"):
                        st.markdown(f"**Catalyst:** {g['catalyst']}")
                    for h in g.get("headlines", []):
                        st.markdown(f"- {h}")

            st.divider()
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(result, indent=2),
                file_name=f"premarket_gappers_{date.today().isoformat()}.json",
                mime="application/json",
            )
    else:
        st.info("Click **Scan Now** to find today's premarket gappers.", icon="ℹ️")

# ═══════════════════════════════════════════════════════════════════════════════
# TRENDING STOCKS TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_trending:
    import json as _json
    from datetime import date as _date

    st.subheader("Trending Stocks Scanner")

    st.markdown("""
    <style>
    /* Target only the 2-column row in the last tabpanel (Trending Stocks).
       :first-child:nth-last-child(2) matches a column that is both first AND
       second-to-last, which is only true in a 2-column layout — not 4/6 col rows. */
    div[role="tabpanel"]:last-child
        [data-testid="stColumn"]:first-child:nth-last-child(2) button {
        background-color: #1a7a3c !important;
        color: white !important;
        border: none !important;
    }
    div[role="tabpanel"]:last-child
        [data-testid="stColumn"]:first-child:nth-last-child(2) button:hover {
        background-color: #145e2e !important;
    }
    div[role="tabpanel"]:last-child
        [data-testid="stColumn"]:last-child:nth-child(2) button {
        background-color: #b91c1c !important;
        color: white !important;
        border: none !important;
    }
    div[role="tabpanel"]:last-child
        [data-testid="stColumn"]:last-child:nth-child(2) button:hover {
        background-color: #991818 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col_up, col_dn = st.columns(2)
    scan_up = col_up.button("📈 Scan Trending Up",   use_container_width=True)
    scan_dn = col_dn.button("📉 Scan Trending Down", use_container_width=True)

    if scan_up or scan_dn:
        direction = "up" if scan_up else "down"
        label = "up" if scan_up else "down"
        with st.spinner(f"Scanning stocks trending {label}… (30–60 s)"):
            try:
                trend_result = _run_trending_scan(direction)
                st.session_state[f"trending_{direction}"] = trend_result
            except Exception as _exc:
                st.error(f"Scan failed: {_exc}")
                trend_result = None
    else:
        # Show whichever was last scanned
        trend_result = st.session_state.get("trending_up") or st.session_state.get("trending_dn")

    if trend_result:
        stocks    = trend_result["stocks"]
        direction = trend_result["direction"]
        scanned_at = trend_result["scanned_at"][:19].replace("T", " ")
        arrow = "📈" if direction == "up" else "📉"

        if not stocks:
            st.info(f"No stocks matched the filters for trending {direction} right now.", icon="ℹ️")
        else:
            st.success(f"{arrow} **{len(stocks)} stocks trending {direction}** — top by volume")
            st.caption(f"Scanned at {scanned_at}")
            st.divider()

            for g in stocks:
                chg_color = "🟢" if direction == "up" else "🔴"
                with st.expander(
                    f"#{g['rank']}  {g['symbol']}  ·  {chg_color} {g['chg_pct']:+.1f}%  ·  ${g['price']:.2f}",
                    expanded=(g["rank"] == 1),
                ):
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    c1.metric("Change %",  f"{g['chg_pct']:+.1f}%")
                    c2.metric("Price",     f"${g['price']:.2f}")
                    c3.metric("Volume",    _fmt_vol(g["volume"]))
                    c4.metric("Avg Vol",   g["avg_volume"])
                    c5.metric("RVOL",      f"{g['rvol']:.1f}×")
                    c6.metric("ATR",       f"{g['atr']:.2f}")

            st.divider()
            st.download_button(
                "⬇️ Download JSON",
                data=_json.dumps(trend_result, indent=2),
                file_name=f"trending_{direction}_{_date.today().isoformat()}.json",
                mime="application/json",
            )
    else:
        st.info("Click **Scan Trending Up** or **Scan Trending Down** to begin.", icon="ℹ️")

st.divider()
st.markdown(
    "<p style='color:#888; font-size:0.85rem; line-height:1.6;'>"
    "⚠️ <strong>Disclaimer:</strong> This app aggregates publicly available market data "
    "and news from third-party sources for informational purposes only. Nothing here "
    "constitutes financial advice or a recommendation to buy or sell any security. "
    "Always do your own research before making investment decisions."
    "</p>",
    unsafe_allow_html=True,
)
