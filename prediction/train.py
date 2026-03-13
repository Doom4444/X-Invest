# prediction/train.py
# Owner: Prediction B (branch: pred-B/model)
#
# WHAT THIS FILE DOES:
#   Trains the Random Forest classifier on historical data from multiple tickers,
#   evaluates it with TimeSeriesSplit, and saves the model to saved_models/.
#   This is a one-time script — run it once before the server starts.
#
# HOW TO RUN:
#   python prediction/train.py
#
# OUTPUT:
#   prediction/saved_models/signal_model.pkl
#
# CRITICAL RULE:
#   ALWAYS use TimeSeriesSplit — NEVER random train_test_split.
#   Random split leaks future data into training and produces fake high accuracy.
#   Expected accuracy: 50–55% across 3 classes.
#   If you see 90%+ accuracy, you are leaking data — fix it before opening the PR.

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    os.makedirs("prediction/saved_models", exist_ok=True)
    train()
