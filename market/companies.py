# market/companies.py
#
# PURPOSE: Curated company list for the Market page (Option A -- hardcoded).
#
# HOW IT FITS:
#   Imported by api/market.py for GET /api/market/companies
#   market.js fetches this and renders company cards
#   Each ticker must match exactly what yfinance expects
#
# UPGRADE TO OPTION B (dynamic):
#   Replace this list with a yfinance screener call
#   Keep the same data structure so market.js does not change

COMPANIES = [
    {"ticker": "AAPL",    "name_en": "Apple",           "name_ar": "أبل",              "market": "NASDAQ",  "flag": "🇺🇸", "sector": "Technology"},
    {"ticker": "MSFT",    "name_en": "Microsoft",        "name_ar": "مايكروسوفت",     "market": "NASDAQ",  "flag": "🇺🇸", "sector": "Technology"},
    {"ticker": "GOOGL",   "name_en": "Alphabet",         "name_ar": "ألفابيت",          "market": "NASDAQ",  "flag": "🇺🇸", "sector": "Technology"},
    {"ticker": "AMZN",    "name_en": "Amazon",           "name_ar": "أمازون",           "market": "NASDAQ",  "flag": "🇺🇸", "sector": "Consumer"},
    {"ticker": "NVDA",    "name_en": "NVIDIA",           "name_ar": "نفيديا",           "market": "NASDAQ",  "flag": "🇺🇸", "sector": "Technology"},
    {"ticker": "TSLA",    "name_en": "Tesla",            "name_ar": "تسلا",             "market": "NASDAQ",  "flag": "🇺🇸", "sector": "Automotive"},
    {"ticker": "META",    "name_en": "Meta",             "name_ar": "ميتا",              "market": "NASDAQ",  "flag": "🇺🇸", "sector": "Technology"},
    {"ticker": "NFLX",    "name_en": "Netflix",          "name_ar": "نتفليكس",          "market": "NASDAQ",  "flag": "🇺🇸", "sector": "Entertainment"},
    {"ticker": "JPM",     "name_en": "JPMorgan",         "name_ar": "جي بي مورجان",      "market": "NYSE",    "flag": "🇺🇸", "sector": "Finance"},
    {"ticker": "COMI.CA", "name_en": "CIB Egypt",        "name_ar": "التجاري الدولي",      "market": "EGX",     "flag": "🇪🇬", "sector": "Finance"},
    {"ticker": "TMGH.CA", "name_en": "Talaat Moustafa",  "name_ar": "طلعت مصطفى",       "market": "EGX",     "flag": "🇪🇬", "sector": "Real Estate"},
    {"ticker": "OCDI.CA", "name_en": "Orascom",          "name_ar": "اوراسكوم",          "market": "EGX",     "flag": "🇪🇬", "sector": "Construction"},
    {"ticker": "EFIH.CA", "name_en": "e-Finance",        "name_ar": "إي فاينانس",       "market": "EGX",     "flag": "🇪🇬", "sector": "Fintech"},
    {"ticker": "MOPCO.CA","name_en": "MOBCO",            "name_ar": "موبكو",              "market": "EGX",     "flag": "🇪🇬", "sector": "Chemicals"},
    {"ticker": "2222.SR", "name_en": "Saudi Aramco",     "name_ar": "أرامكو",           "market": "Tadawul", "flag": "🇸🇦", "sector": "Energy"},
    {"ticker": "1120.SR", "name_en": "Al Rajhi Bank",    "name_ar": "مصرف الراجحي",     "market": "Tadawul", "flag": "🇸🇦", "sector": "Finance"},
    {"ticker": "7010.SR", "name_en": "STC",              "name_ar": "الاتصالات السعودية",  "market": "Tadawul", "flag": "🇸🇦", "sector": "Telecom"},
    {"ticker": "2010.SR", "name_en": "SABIC",            "name_ar": "سابك",              "market": "Tadawul", "flag": "🇸🇦", "sector": "Chemicals"},
    {"ticker": "1180.SR", "name_en": "Saudi National Bank","name_ar": "البنك الأهلي",       "market": "Tadawul", "flag": "🇸🇦", "sector": "Finance"},
]
