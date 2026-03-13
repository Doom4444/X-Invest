# api/signal.py
#
# PURPOSE: Prediction signal endpoint.
#
# HOW IT FITS:
#   Registered in main.py via app.include_router(signal_router)
#   Called by market.js when viewing a company dashboard
#   Also callable from chat.js when a ticker is detected
#   Delegates entirely to prediction/signal_engine.py (Teammates 2+3)
#
# CONTRACT with signal_engine:
#   get_signal() MUST return all keys, MUST NEVER raise exceptions

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/api/signal/{ticker}")
async def get_signal(ticker: str):
    try:
        from prediction.signal_engine import get_signal as _get
        return JSONResponse(_get(ticker.upper()))
    except NotImplementedError:
        return JSONResponse({
            "ticker": ticker.upper(), "signal": "unavailable",
            "confidence": 0, "rsi": None, "sma_cross": None,
            "rf_signal": None, "disclaimer": "Prediction module in development",
            "error": "not implemented"
        })
    except Exception as e:
        return JSONResponse({"ticker": ticker.upper(), "signal": "error", "error": str(e)}, status_code=500)
