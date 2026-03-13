# pipeline/entity_extractor.py
#
# PURPOSE: Scans the user message for stock tickers (Arabic or English).
#
# HOW IT FITS:
#   Called by context_builder.py for every message
#   Returns ["AAPL", "COMI.CA"] etc.
#   context_builder passes these to data_fetcher.get_stock_data()
#
# TWO DETECTION METHODS:
#   1. Regex: finds explicit uppercase tickers like AAPL, TSLA, COMI.CA
#      Minimum 2 chars to avoid catching single letters (E from P/E)
#   2. Name lookup: "apple" or "ابل" in query -> "AAPL"
#
# TICKER FORMATS:
#   US:     AAPL, TSLA, MSFT (standard)
#   Egypt:  COMI.CA, TMGH.CA (.CA suffix required by yfinance)
#   Saudi:  2222.SR, 1120.SR (.SR suffix required by yfinance)

import re

KNOWN_TICKERS = {
    "apple": "AAPL",        "أبل": "AAPL",
    "tesla": "TSLA",        "تسلا": "TSLA",
    "microsoft": "MSFT",    "مايكروسوفت": "MSFT",
    "google": "GOOGL",      "جوجل": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",       "أمازون": "AMZN",
    "nvidia": "NVDA",       "نفيديا": "NVDA",
    "meta": "META",         "ميتا": "META",
    "netflix": "NFLX",      "نتفليكس": "NFLX",
    "jpmorgan": "JPM",
    "cib": "COMI.CA",               "التجاري الدولي": "COMI.CA",
    "talaat moustafa": "TMGH.CA",   "طلعت مصطفى": "TMGH.CA",
    "orascom": "OCDI.CA",           "اوراسكوم": "OCDI.CA",
    "mobco": "MOPCO.CA",            "موبكو": "MOPCO.CA",
    "e-finance": "EFIH.CA",         "إي فاينانس": "EFIH.CA",
    "aramco": "2222.SR",            "أرامكو": "2222.SR",
    "al rajhi": "1120.SR",          "الراجحي": "1120.SR",
    "stc": "7010.SR",               "اتصالات السعودية": "7010.SR",
    "sabic": "2010.SR",             "سابك": "2010.SR",
}

def extract_tickers(query: str) -> list[str]:
    tickers = set()
    # 2+ uppercase letters, optional .XX suffix (for EGX .CA, Saudi .SR)
    tickers.update(re.findall(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b", query))
    query_lower = query.lower()
    for name, ticker in KNOWN_TICKERS.items():
        if name in query_lower or name in query:
            tickers.add(ticker)
    return list(tickers)
