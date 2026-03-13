# api/market.py
#
# PURPOSE: Market page data endpoints.
#
# HOW IT FITS:
#   Registered in main.py via app.include_router(market_router)
#   Called by static/js/market.js
#   Completely independent of the chat pipeline
#   If Ollama is down, market page still works
#
# GET /api/market/companies  -> company list for the sidebar
# GET /api/market/{ticker}   -> full dashboard data for one company

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from market.companies import COMPANIES
from market.dashboard import get_dashboard_data

router = APIRouter()

@router.get("/api/market/companies")
async def list_companies():
    """Returns curated company list. market.js renders these as cards."""
    return JSONResponse(COMPANIES)

@router.get("/api/market/{ticker}")
async def company_dashboard(ticker: str):
    """Returns full dashboard data. Called when user clicks a company card."""
    data = get_dashboard_data(ticker.upper())
    if not data:
        return JSONResponse({"error": f"No data for {ticker}"}, status_code=404)
    return JSONResponse(data)
