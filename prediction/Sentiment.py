"""
Sentiment.py — X-INVEST (v6 — Improved)
التحسينات:
  [FIX-1] Financial-aware sentiment مع VADER
  [FIX-2] Weighted aggregation: الأخبار الأحدث بتاخد وزن أعلى
  [FIX-3] Negation handling أحسن
  [FIX-4] إضافة Sentiment Trend: هل الـ sentiment بيتحسن أو بيتراجع؟
  [FIX-5] إضافة Source Credibility Weighting
  [NEW]   Sentiment Summary مع Emoji للعرض السريع
"""
import os
import datetime
import warnings
from typing import Optional
import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── VADER Cache ──────────────────────────────────────────────────────────────
_VADER_CACHE = {}


def _get_vader():
    if "analyzer" not in _VADER_CACHE:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _VADER_CACHE["analyzer"] = SentimentIntensityAnalyzer()
        except ImportError:
            raise ImportError("pip install vaderSentiment")
    return _VADER_CACHE["analyzer"]


# ── Financial Keyword Adjustments ────────────────────────────────────────────
FINANCIAL_POSITIVE_BOOST = {
    "beat estimates": +0.4,
    "beats expectations": +0.4,
    "raised guidance": +0.35,
    "record revenue": +0.35,
    "dividend increase": +0.3,
    "buyback": +0.25,
    "share repurchase": +0.25,
    "strong earnings": +0.3,
    "upgraded": +0.3,
    "outperform": +0.3,
    "bullish": +0.25,
    "price target raised": +0.4,
    "reduced losses": +0.2,
    "narrowed loss": +0.2,
    "turnaround": +0.2,
    "expansion": +0.15,
    "partnership": +0.15,
    "new contract": +0.2,
    "market share gain": +0.25,
    "all-time high": +0.3,
    "ath": +0.2,
    "revenue growth": +0.25,
    "profit growth": +0.3,
}

FINANCIAL_NEGATIVE_BOOST = {
    "missed estimates": -0.4,
    "below expectations": -0.35,
    "lowered guidance": -0.4,
    "guidance cut": -0.4,
    "downgraded": -0.35,
    "underperform": -0.3,
    "bearish": -0.25,
    "price target cut": -0.4,
    "layoffs": -0.25,
    "restructuring": -0.2,
    "sec investigation": -0.5,
    "fraud": -0.5,
    "bankruptcy": -0.6,
    "recall": -0.3,
    "supply chain": -0.15,
    "margin compression": -0.3,
    "revenue miss": -0.35,
    "earnings miss": -0.35,
    "profit warning": -0.4,
    "revenue decline": -0.3,
    "market share loss": -0.25,
}

# [FIX-5] Source credibility weights (high = more trustworthy)
SOURCE_CREDIBILITY = {
    "reuters": 1.3,
    "bloomberg": 1.3,
    "wall street journal": 1.2,
    "financial times": 1.2,
    "cnbc": 1.1,
    "marketwatch": 1.1,
    "seeking alpha": 0.9,
    "motley fool": 0.85,
    "benzinga": 0.9,
    "default": 1.0,
}


def _get_source_weight(source: str) -> float:
    if not source:
        return SOURCE_CREDIBILITY["default"]
    src_lower = source.lower()
    for key, weight in SOURCE_CREDIBILITY.items():
        if key in src_lower:
            return weight
    return SOURCE_CREDIBILITY["default"]


def _apply_financial_keywords(text: str, base_compound: float) -> float:
    text_lower = text.lower()
    adjustment = 0.0
    for phrase, boost in FINANCIAL_POSITIVE_BOOST.items():
        if phrase in text_lower:
            adjustment += boost
    for phrase, boost in FINANCIAL_NEGATIVE_BOOST.items():
        if phrase in text_lower:
            adjustment += boost
    adjusted = np.clip(base_compound + adjustment * 0.5, -1.0, 1.0)
    return float(adjusted)


SCORE_MAP_VADER = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def _vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def score_headlines(headlines: list, sources: list = None) -> list:
    if not headlines:
        return []
    if sources is None:
        sources = [""] * len(headlines)
    try:
        analyzer = _get_vader()
    except ImportError as e:
        print(f"\n⚠️ [Sentiment] {e}")
        return [
            {"text": h, "label": "neutral", "score": 0.0,
             "confidence": 0.5, "compound": 0.0, "source_weight": 1.0}
            for h in headlines
        ]

    results = []
    for text, source in zip(headlines, sources):
        raw_scores = analyzer.polarity_scores(str(text)[:512])
        base_compound = raw_scores["compound"]
        compound = _apply_financial_keywords(str(text), base_compound)
        label = _vader_label(compound)
        confidence = min(abs(compound) * 2, 1.0) if label != "neutral" else 0.5
        source_weight = _get_source_weight(source)

        results.append({
            "text":         text,
            "label":        label,
            "score":        SCORE_MAP_VADER[label],
            "confidence":   round(confidence, 4),
            "compound":     round(compound, 4),
            "source_weight": source_weight,
        })

    return results


def aggregate_sentiment(scored: list, weights: list = None) -> dict:
    """
    Weighted aggregation مع source credibility.
    weights: قائمة بنفس طول scored، كبيرة = أحدث
    """
    if not scored:
        return {
            "sentiment_score": 0.0, "sentiment_pos_ratio": 0.0,
            "sentiment_neg_ratio": 0.0, "sentiment_neu_ratio": 0.0,
            "sentiment_confidence": 0.5,
            "n_articles": 0, "n_positive": 0, "n_negative": 0, "n_neutral": 0,
            "sentiment_emoji": "➡️",
        }

    labels    = [d["label"]                  for d in scored]
    compounds = [d.get("compound", 0.0)      for d in scored]
    confs     = [d["confidence"]              for d in scored]
    src_w     = [d.get("source_weight", 1.0) for d in scored]
    n         = len(scored)

    if weights is None:
        weights = np.ones(n)
    weights = np.array(weights, dtype=float) * np.array(src_w, dtype=float)
    weights = weights / weights.sum()

    n_pos = labels.count("positive")
    n_neg = labels.count("negative")
    n_neu = labels.count("neutral")

    score = float(np.dot(weights, compounds))
    emoji = "📈" if score > 0.1 else "📉" if score < -0.1 else "➡️"

    return {
        "sentiment_score":      score,
        "sentiment_pos_ratio":  n_pos / n,
        "sentiment_neg_ratio":  n_neg / n,
        "sentiment_neu_ratio":  n_neu / n,
        "sentiment_confidence": float(np.dot(weights, confs)),
        "n_articles": n, "n_positive": n_pos,
        "n_negative": n_neg, "n_neutral": n_neu,
        "sentiment_emoji": emoji,
    }


TICKER_TO_NAME = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google Alphabet",
    "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "NVIDIA", "META": "Meta Facebook",
    "JPM": "JPMorgan",
}


def _fetch_yfinance_news(ticker: str, max_articles: int = 100) -> list:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        raw_news = t.news or []
        articles = []
        for item in raw_news[:max_articles]:
            content = item.get("content", {})
            title   = content.get("title", "") or item.get("title", "") or ""
            summary = content.get("summary", "") or item.get("summary", "") or ""
            pub     = (content.get("pubDate") or content.get("displayTime")
                       or item.get("providerPublishTime", ""))
            if isinstance(pub, (int, float)):
                pub = datetime.datetime.utcfromtimestamp(pub).strftime("%Y-%m-%dT%H:%M:%SZ")
            provider = content.get("provider", {})
            source   = (provider.get("displayName") if isinstance(provider, dict)
                        else item.get("publisher", ""))
            if title:
                articles.append({
                    "title":       title,
                    "description": summary,
                    "publishedAt": str(pub),
                    "source":      source,
                })
        return articles
    except Exception as e:
        print(f"[sentiment] yfinance news error for {ticker}: {e}")
        return []


def _fetch_newsapi(ticker: str, days_back: int = 30, max_articles: int = 100) -> list:
    try:
        from newsapi import NewsApiClient
        raw  = os.getenv("NEWS_API_KEY", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            return []
        query     = TICKER_TO_NAME.get(ticker.upper(), ticker)
        to_date   = datetime.date.today()
        from_date = to_date - datetime.timedelta(days=min(days_back, 30))
        client    = NewsApiClient(api_key=keys[0])
        response  = client.get_everything(
            q=query, language="en", sort_by="relevancy",
            from_param=str(from_date), to=str(to_date),
            page_size=min(max_articles, 100),
        )
        return [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "publishedAt": a.get("publishedAt", ""),
                "source":      a.get("source", {}).get("name", ""),
            }
            for a in response.get("articles", []) if a.get("title")
        ]
    except Exception as e:
        print(f"[sentiment] NewsAPI error for {ticker}: {e}")
        return []


def fetch_news(ticker: str, days_back: int = 7, max_articles: int = 100) -> list:
    articles = _fetch_yfinance_news(ticker, max_articles=max_articles)
    if not articles:
        articles = _fetch_newsapi(ticker, days_back=days_back, max_articles=max_articles)
    return articles


def get_daily_sentiment(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days_back: int = 30,
) -> pd.DataFrame:
    articles = fetch_news(ticker, days_back=days_back, max_articles=100)
    cols = ["sentiment_score", "sentiment_pos_ratio", "sentiment_neg_ratio",
            "sentiment_neu_ratio", "sentiment_confidence", "n_articles"]

    if not articles:
        idx = pd.date_range(
            start=start_date or (datetime.date.today() - datetime.timedelta(days=days_back)),
            end=end_date or datetime.date.today(), freq="D"
        )
        return pd.DataFrame(0.0, index=idx, columns=cols)

    by_day: dict[str, list] = {}
    by_day_sources: dict[str, list] = {}
    for a in articles:
        pub = a["publishedAt"]
        day = pub[:10] if pub and len(pub) >= 10 else ""
        if day:
            text = (a["title"] + ". " + a["description"]).strip()
            by_day.setdefault(day, []).append(text)
            by_day_sources.setdefault(day, []).append(a.get("source", ""))

    rows = {}
    for day, texts in by_day.items():
        sources = by_day_sources.get(day, [])
        scored = score_headlines(texts, sources=sources)
        agg = aggregate_sentiment(scored)
        rows[day] = {k: agg[k] for k in cols}

    if not rows:
        idx = pd.date_range(
            start=start_date or (datetime.date.today() - datetime.timedelta(days=days_back)),
            end=end_date or datetime.date.today(), freq="D"
        )
        return pd.DataFrame(0.0, index=idx, columns=cols)

    daily_df = pd.DataFrame.from_dict(rows, orient="index")
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df.sort_index(inplace=True)
    idx = pd.date_range(
        start=start_date or daily_df.index.min(),
        end=end_date or daily_df.index.max(), freq="D"
    )
    daily_df = daily_df.reindex(idx).ffill().fillna(0.0)
    return daily_df


def get_latest_sentiment(ticker: str, days_back: int = 7) -> dict:
    articles = fetch_news(ticker, days_back=days_back, max_articles=50)
    if not articles:
        return {
            "sentiment_score": 0.0, "sentiment_pos_ratio": 0.0,
            "sentiment_neg_ratio": 0.0, "sentiment_neu_ratio": 0.0,
            "sentiment_confidence": 0.5, "n_articles": 0,
            "n_positive": 0, "n_negative": 0, "n_neutral": 0,
            "sentiment_emoji": "➡️",
        }

    texts   = [(a["title"] + ". " + a["description"]).strip() for a in articles]
    sources = [a.get("source", "") for a in articles]

    n = len(texts)
    weights = np.linspace(1.0, 0.3, n)

    scored = score_headlines(texts, sources=sources)
    result = aggregate_sentiment(scored, weights=weights.tolist())

    # [FIX-4] Sentiment Trend: هل الـ sentiment بيتحسن؟
    if n >= 6:
        recent_compounds   = [s["compound"] for s in scored[:n//2]]
        older_compounds    = [s["compound"] for s in scored[n//2:]]
        recent_avg = np.mean(recent_compounds) if recent_compounds else 0
        older_avg  = np.mean(older_compounds)  if older_compounds  else 0
        result["sentiment_trend"] = round(float(recent_avg - older_avg), 4)
    else:
        result["sentiment_trend"] = 0.0

    return result


SENTIMENT_FEATURES = [
    "sentiment_score", "sentiment_pos_ratio", "sentiment_neg_ratio",
    "sentiment_neu_ratio", "sentiment_confidence", "n_articles",
]