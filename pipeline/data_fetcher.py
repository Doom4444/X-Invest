# pipeline/data_fetcher.py
#
# PURPOSE: Fetches real-time stock data from Yahoo Finance.
#
# HOW IT FITS:
#   context_builder.py calls get_stock_data(ticker) for each detected ticker
#   Then calls format_for_prompt(data) to get Arabic text for the LLM
#   market/dashboard.py also uses this for the Market page (different format)
#
# NEVER CRASHES: returns None on any error so pipeline continues

import yfinance as yf

def get_stock_data(ticker: str) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return None
        return {
            "ticker":      ticker,
            "name":        info.get("longName", ticker),
            "price":       price,
            "currency":    info.get("currency", "USD"),
            "pe_ratio":    info.get("trailingPE"),
            "market_cap":  info.get("marketCap"),
            "week52_high": info.get("fiftyTwoWeekHigh"),
            "week52_low":  info.get("fiftyTwoWeekLow"),
            "change_pct":  info.get("regularMarketChangePercent"),
            "volume":      info.get("volume"),
            "sector":      info.get("sector"),
            "news":        [n["title"] for n in (stock.news or [])[:3]
                           if isinstance(n, dict) and "title" in n],
        }
    except Exception as e:
        print(f"[data_fetcher] {ticker}: {e}")
        return None

def format_for_prompt(d: dict) -> str:
    """Format as Arabic text for LLM context injection"""
    if not d: return ""
    lines = [
        f"Stock: {d['name']} ({d['ticker']})",
        f"Current Price: {d['price']} {d['currency']}",
    ]
    if d.get("pe_ratio"):    lines.append(f"P/E Ratio: {d['pe_ratio']:.2f}")
    if d.get("change_pct"):  lines.append(f"Today Change: {d['change_pct']:.2f}%")
    if d.get("week52_high"): lines.append(f"52W High: {d['week52_high']}")
    if d.get("week52_low"):  lines.append(f"52W Low: {d['week52_low']}")
    if d.get("market_cap"):  lines.append(f"Market Cap: {d['market_cap']:,}")
    if d.get("news"):        lines.append(f"Latest News: {'; '.join(d['news'])}")
    return "\n".join(lines)
