# prediction/indicators.py
# Owner: Prediction A (branch: pred-A/indicators)
#
# WHAT THIS FILE DOES:
#   Fetches 1 year of daily price history from yfinance for a given ticker,
#   computes all technical indicators, labels each row as bullish/neutral/bearish
#   based on the 5-day forward return, and returns a clean DataFrame.
#
# WHAT OTHER FILES IMPORT FROM HERE:
#   prediction/model.py  → get_features() for training + FEATURES list
#   prediction/train.py  → get_features() for training + FEATURES list
#   prediction/signal_engine.py → get_features() for live inference
#
# COLUMN CONTRACT (do not rename these — model.py depends on them):
#   SMA_cross     float  1.0 if SMA_20 > SMA_50 else 0.0
#   RSI           float  0–100, 14-period
#   MACD          float  EMA12 - EMA26
#   Volatility    float  10-day rolling std of daily returns
#   Volume_Change float  day-over-day volume % change
#   signal        str    "bullish" | "neutral" | "bearish"

import pandas as pd
import numpy as np
import yfinance as yf

# FEATURES must match the column names above exactly.
# model.py and train.py import this list — do not rename or reorder.
FEATURES = ["SMA_cross", "RSI", "MACD", "Volatility", "Volume_Change"]


def get_features(ticker: str) -> pd.DataFrame:
    """
    Fetch 1 year of daily data for ticker and compute all indicators.
    Returns a DataFrame with FEATURES columns + 'signal' column.
    Returns empty DataFrame if ticker is invalid or data unavailable.

    STUB — Prediction A replaces this body with the real implementation.
    The column names, types, and return shape must stay exactly as shown.
    """
    # ── STUB: returns random data with correct shape so B and C can start ──
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    return pd.DataFrame({
        "SMA_cross":     np.random.randint(0, 2, 100).astype(float),
        "RSI":           np.random.uniform(30, 70, 100),
        "MACD":          np.random.uniform(-2, 2, 100),
        "Volatility":    np.random.uniform(0.01, 0.05, 100),
        "Volume_Change": np.random.uniform(-0.2, 0.2, 100),
        "signal":        np.random.choice(["bullish", "neutral", "bearish"], 100),
    }, index=dates)
    # ── END STUB ────────────────────────────────────────────────────────────
