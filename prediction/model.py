# prediction/model.py
# Owner: Prediction B (branch: pred-B/model)
#
# WHAT THIS FILE DOES:
#   Loads the trained model from saved_models/signal_model.pkl and
#   exposes predict_signal(ticker) for live inference.
#   Called only by prediction/signal_engine.py at runtime.
#
# TRAINING IS IN A SEPARATE FILE:
#   prediction/train.py — run this once to produce the .pkl file
#   python prediction/train.py
#
# CRITICAL RULE:
#   predict_signal() must NEVER raise an exception.
#   If the model file is missing or inference fails, return the error in the dict.

import os
import pickle
from prediction.indicators import get_features, FEATURES

MODEL_PATH = "prediction/saved_models/signal_model.pkl"

_model = None


def _load_model():
    """Load model from disk once and cache it in memory."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python prediction/train.py"
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def predict_signal(ticker: str) -> dict:
    """
    Load trained model and predict signal for the given ticker.

    Returns:
        {
            "signal":     "bullish" | "neutral" | "bearish" | "unknown",
            "confidence": float (0–100),
            "rsi":        float | None,
            "sma_cross":  bool  | None,
        }

    Never raises — errors returned as {"signal": "unknown", ...}.

    STUB — Prediction B replaces this body with the real implementation.
    The return structure must stay exactly as shown.
    """
    # ── STUB: returns fixed data so C can build signal_engine.py ──────────
    return {
        "signal":     "bullish",
        "confidence": 65.0,
        "rsi":        55.0,
        "sma_cross":  True,
    }
    # ── END STUB ────────────────────────────────────────────────────────────
