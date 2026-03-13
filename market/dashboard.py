# market/dashboard.py
#
# PURPOSE: Fetches complete company data for the Market page dashboard panel.
#
# HOW IT FITS:
#   Called by api/market.py for GET /api/market/{ticker}
#   Returns structured JSON that market.js renders directly
#
# WHY DIFFERENT FROM data_fetcher.py:
#   data_fetcher -> optimized for LLM (Arabic text strings, minimal fields)
#   dashboard.py -> optimized for UI (all fields, structured JSON, news links)

import yfinance as yf
from market.companies import COMPANIES

_MAP = {c["ticker"]: c for c in COMPANIES}

def get_dashboard_data(ticker: str) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price: return None
        meta = _MAP.get(ticker, {})
        return {
            "ticker":     ticker,
            "name_en":    info.get("longName", meta.get("name_en", ticker)),
            "name_ar":    meta.get("name_ar", ""),
            "market":     meta.get("market", ""),
            "flag":       meta.get("flag", ""),
            "sector":     info.get("sector", meta.get("sector", "")),
            "website":    info.get("website"),
            "price":      price,
            "currency":   info.get("currency", "USD"),
            "change":     info.get("regularMarketChange"),
            "change_pct": info.get("regularMarketChangePercent"),
            "market_cap": info.get("marketCap"),
            "pe_ratio":   info.get("trailingPE"),
            "pb_ratio":   info.get("priceToBook"),
            "eps":        info.get("trailingEps"),
            "dividend":   info.get("dividendYield"),
            "week52_high":info.get("fiftyTwoWeekHigh"),
            "week52_low": info.get("fiftyTwoWeekLow"),
            "day_high":   info.get("dayHigh"),
            "day_low":    info.get("dayLow"),
            "volume":     info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "news": [{"title": n["title"], "link": n.get("link","")}
                     for n in (stock.news or [])[:5]
                     if isinstance(n, dict) and "title" in n],
        }
    except Exception as e:
        print(f"[dashboard] {ticker}: {e}")
        return None
