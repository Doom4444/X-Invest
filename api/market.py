# api/market.py
#
# Market page data endpoints — called by static/js/market.js and related scripts.
#
# GET /api/market/dashboard          -> macro strip + ticker matrix (live)
# GET /api/market/{ticker}/history   -> price history for chart
# GET /api/market/{ticker}/forecast  -> model forecast panel
# GET /api/market/companies          -> curated company list (legacy)
# GET /api/market/{ticker}           -> yfinance company snapshot (legacy)

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from market.companies import COMPANIES
from market.dashboard import get_dashboard_data
from market.dashboard_feed import (
    get_dashboard_snapshot,
    get_forecast_panel,
    get_price_history,
)

router = APIRouter()


@router.get("/api/market/dashboard")
async def market_dashboard():
    """Full dashboard snapshot: macro indicators + all ticker rows."""
    try:
        data = get_dashboard_snapshot()
        if not data["tickers"]:
            return JSONResponse(
                {"error": "No ticker data available", "macro": data.get("macro", {})},
                status_code=503,
            )
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/market/{ticker}/history")
async def ticker_history(
    ticker: str,
    year_from: int = Query(2020, alias="from", ge=2010, le=2030),
    year_to: int = Query(2026, alias="to", ge=2010, le=2030),
):
    data = get_price_history(ticker, year_from, year_to)
    if not data:
        return JSONResponse({"error": f"No history for {ticker.upper()}"}, status_code=404)
    return JSONResponse(data)


@router.get("/api/market/{ticker}/forecast")
async def ticker_forecast(ticker: str):
    data = get_forecast_panel(ticker)
    if not data:
        return JSONResponse({"error": f"No forecast for {ticker.upper()}"}, status_code=404)
    return JSONResponse(data)


@router.get("/api/market/companies")
async def list_companies():
    """Returns curated company list."""
    return JSONResponse(COMPANIES)


@router.get("/api/market/{ticker}")
async def company_dashboard(ticker: str):
    """Returns yfinance snapshot for one company."""
    data = get_dashboard_data(ticker.upper())
    if not data:
        return JSONResponse({"error": f"No data for {ticker}"}, status_code=404)
    return JSONResponse(data)
