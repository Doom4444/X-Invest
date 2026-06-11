import argparse
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import mstats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    mean_absolute_error, precision_recall_curve,
    precision_score, r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from prediction.indicators import get_features, FEATURES

MODEL_PATH = "prediction/saved_models/signal_model.pkl"
TICKERS    = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]


def train():
    """
    Train the Random Forest classifier and save to MODEL_PATH.

    Steps to implement:
      1. Call get_features(ticker) for each ticker in TICKERS
      2. Concatenate all DataFrames — drop any rows with NaN
      3. X = df[FEATURES],  y = df["signal"]
      4. Use TimeSeriesSplit(n_splits=5) to evaluate — print classification_report
      5. Fit final model on all data
      6. Save to MODEL_PATH using pickle

    TODO: Prediction B — implement these steps below.
    """
    # TODO: Prediction B
    raise NotImplementedError("Prediction B: implement train() in train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-INVEST Training v14")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--period",  default="5y")
    parser.add_argument("--cutoff",  default=None,
                        help="Train only on data before this date (YYYY-MM-DD). "
                             "Use this to avoid leakage when backtesting a future period.")
    args = parser.parse_args()
    train_and_save(args.tickers, args.period, cutoff_date=args.cutoff)