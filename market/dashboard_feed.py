# market/dashboard_feed.py
#
# Aggregates live market data for the /market dashboard page.
# Uses yfinance + prediction/train.get_features + prediction/predict.py.

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from market.companies import COMPANIES

logger = logging.getLogger("x-invest.market")

SIGNAL_MAP = {
    "BUY": "bullish",
    "HOLD": "neutral",
    "SELL": "bearish",
}

_COMPANY_MAP = {c["ticker"]: c for c in COMPANIES}

_MACRO_SYMBOLS = {
    "vix": "^VIX",
    "sp500": "^GSPC",
    "tnx_10y": "^TNX",
    "oil": "CL=F",
    "gold": "GC=F",
    "btc": "BTC-USD",
    "dxy": "DX-Y.NYB",
}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def _latest_close(symbol: str) -> tuple[float, float]:
    """Return (last_close, day_change_pct)."""
    try:
        hist = yf.download(symbol, period="5d", progress=False, auto_adjust=True)
        if hist.empty:
            return 0.0, 0.0
        hist = _flatten_yf(hist)
        closes = hist["Close"].dropna()
        if len(closes) < 1:
            return 0.0, 0.0
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else last
        chg = ((last - prev) / prev * 100) if prev else 0.0
        return last, chg
    except Exception as exc:
        logger.warning("macro fetch failed for %s: %s", symbol, exc)
        return 0.0, 0.0


def _derive_signal_from_features(latest) -> str:
    rsi = _safe_float(latest.get("rsi_14"), 50)
    macd_hist = _safe_float(latest.get("macd_hist"))
    momentum = _safe_float(latest.get("momentum_score"), 0.5)
    if rsi >= 55 and macd_hist > 0 and momentum >= 0.5:
        return "bullish"
    if rsi <= 45 and macd_hist < 0 and momentum <= 0.5:
        return "bearish"
    return "neutral"


def _run_prediction(ticker: str) -> dict | None:
    try:
        from prediction.predict import _load_model, predict_signal

        if _load_model() is None:
            return None
        pred = predict_signal(ticker)
        if pred.get("signal") == "ERROR":
            return None
        return pred
    except Exception as exc:
        logger.warning("prediction unavailable for %s: %s", ticker, exc)
        return None


def _simple_forecast(ticker: str, company: dict, n_days: int = 22) -> dict:
    hist = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"no price history for {ticker}")
    hist = _flatten_yf(hist)
    closes = hist["Close"].dropna()
    current = float(closes.iloc[-1])
    log_ret = np.log(closes / closes.shift(1)).dropna()
    vol = float(log_ret.tail(60).std()) if len(log_ret) > 1 else 0.02
    drift = float(log_ret.tail(5).mean()) if len(log_ret) > 5 else 0.0

    from prediction.train import get_features

    df = get_features(ticker, period="1y")
    latest = df.iloc[-1] if not df.empty else {}
    ui_signal = _derive_signal_from_features(latest)
    raw_signal = {"bullish": "BUY", "bearish": "SELL", "neutral": "HOLD"}[ui_signal]
    direction = {"bullish": "UP", "bearish": "DOWN", "neutral": "NEUTRAL"}[ui_signal]

    forecasts = []
    price = current
    today = datetime.now().date()
    trading_day = 0
    cal = 0
    while trading_day < n_days:
        cal += 1
        d = today + timedelta(days=cal)
        if d.weekday() >= 5:
            continue
        trading_day += 1
        price = float(price * np.exp(drift))
        band = price * vol * np.sqrt(trading_day) * 1.645
        forecasts.append({
            "date": d.strftime("%Y-%m-%d"),
            "forecast_price": round(price, 2),
            "upper_band": round(float(price + band), 2),
            "lower_band": round(float(max(price - band, 0.01)), 2),
        })

    price_target = forecasts[-1]["forecast_price"] if forecasts else current
    return {
        "meta": {
            "ticker": ticker,
            "sector": company.get("sector") or "",
            "current_price": round(current, 2),
            "signal": raw_signal,
            "direction": direction,
            "price_target": round(price_target, 2),
        },
        "forecasts": forecasts,
    }


def get_macro_snapshot() -> dict:
    macro: dict = {
        "vix": 0.0,
        "vix_chg": 0.0,
        "sp500": 0.0,
        "sp500_chg": 0.0,
        "tnx_10y": 0.0,
        "tnx_10y_chg": 0.0,
        "oil": 0.0,
        "oil_chg": 0.0,
        "gold": 0.0,
        "gold_chg": 0.0,
        "btc": 0.0,
        "btc_chg": 0.0,
        "dxy": 0.0,
        "dxy_chg": 0.0,
        "fed_rate": 5.25,
        "cpi": 3.2,
    }
    for key, sym in _MACRO_SYMBOLS.items():
        val, chg = _latest_close(sym)
        macro[key] = round(val, 2)
        macro[f"{key}_chg"] = round(chg, 2)
    return macro


def _build_ticker_row(company: dict) -> dict | None:
    ticker = company["ticker"]
    try:
        from prediction.train import get_features

        df = get_features(ticker, period="2y")
        if df.empty:
            logger.warning("no features for %s", ticker)
            return None

        latest = df.iloc[-1]
        signal = _derive_signal_from_features(latest)
        exp_return = round(_safe_float(latest.get("return_5d")) * 100, 3)

        return {
            "ticker": ticker,
            "name": company.get("name_en", ticker),
            "sector": company.get("sector") or "",
            "flag": company.get("flag", ""),
            "close": round(_safe_float(latest.get("close")), 2),
            "return_1d": round(_safe_float(latest.get("return_1d")) * 100, 3),
            "return_5d": round(_safe_float(latest.get("return_5d")) * 100, 3),
            "signal": signal,
            "rsi_14": round(_safe_float(latest.get("rsi_14")), 2),
            "macd": round(_safe_float(latest.get("macd")), 3),
            "macd_hist": round(_safe_float(latest.get("macd_hist")), 3),
            "bb_percent": round(_safe_float(latest.get("bb_percent")), 4),
            "atr_14": round(_safe_float(latest.get("atr_14")), 2),
            "stoch_k": round(_safe_float(latest.get("stoch_k")), 2),
            "trend_strength": round(_safe_float(latest.get("trend_strength")), 3),
            "momentum_score": round(_safe_float(latest.get("momentum_score")), 3),
            "dist_52w_high": round(_safe_float(latest.get("dist_52w_high")), 2),
            "dist_52w_low": round(_safe_float(latest.get("dist_52w_low")), 2),
            "exp_return": round(exp_return, 3),
        }
    except Exception as exc:
        logger.exception("ticker row failed for %s: %s", ticker, exc)
        return None


def get_dashboard_snapshot(max_workers: int = 4) -> dict:
    tickers: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_build_ticker_row, c): c for c in COMPANIES}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                tickers.append(row)

    tickers.sort(key=lambda t: t["ticker"])
    return {
        "macro": get_macro_snapshot(),
        "tickers": tickers,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_price_history(ticker: str, year_from: int = 2020, year_to: int = 2026) -> dict | None:
    ticker = ticker.upper()
    if ticker not in _COMPANY_MAP:
        return None

    start = f"{year_from}-01-01"
    end = f"{min(year_to, datetime.now().year)}-12-31"
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            return None
        raw = _flatten_yf(raw)
        history = []
        for idx, row in raw.iterrows():
            history.append({
                "date": pd.Timestamp(idx).strftime("%Y-%m-%d"),
                "close": round(_safe_float(row.get("Close")), 4),
                "vol": round(_safe_float(row.get("Volume")) / 1e7, 4),
            })
        return {"ticker": ticker, "history": history}
    except Exception as exc:
        logger.exception("history failed for %s: %s", ticker, exc)
        return None


def get_forecast_panel(ticker: str) -> dict | None:
    ticker = ticker.upper()
    company = _COMPANY_MAP.get(ticker)
    if not company:
        return None

    try:
        pred = _run_prediction(ticker)
        if pred:
            direction = str(pred.get("direction") or "NEUTRAL").upper()
            if direction == "FLAT":
                direction = "NEUTRAL"

            forecasts = []
            for point in pred.get("forecast_series") or []:
                forecasts.append({
                    "date": point["date"],
                    "forecast_price": _safe_float(point.get("price")),
                    "upper_band": _safe_float(point.get("upper")),
                    "lower_band": _safe_float(point.get("lower")),
                })

            return {
                "meta": {
                    "ticker": ticker,
                    "sector": company.get("sector") or "",
                    "current_price": _safe_float(pred.get("current_price")),
                    "signal": str(pred.get("signal") or "HOLD").upper(),
                    "direction": direction,
                    "price_target": _safe_float(pred.get("price_target")),
                },
                "forecasts": forecasts,
            }

        return _simple_forecast(ticker, company)
    except Exception as exc:
        logger.exception("forecast failed for %s: %s", ticker, exc)
        return None
