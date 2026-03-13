# prediction/signal_engine.py
# Owner: Prediction C (branch: pred-C/signal-engine)
#
# WHAT THIS FILE DOES:
#   The ONLY file Adham calls from the prediction module.
#   Wires indicators.py and model.py together into one public function.
#
# CALLED BY:
#   api/signal.py              → GET /api/signal/{ticker}  (market page badge)
#   pipeline/context_builder.py → injects signal into chat context
#
# CRITICAL RULES:
#   1. get_signal() must NEVER raise an exception under any circumstances.
#      Catch everything internally. Errors go in the "error" key as a string.
#   2. All 8 keys in the return dict are required — never omit any.
#   3. "signal" must be exactly one of: bullish | neutral | bearish | unknown

from prediction.model import predict_signal


def get_signal(ticker: str) -> dict:
    """
    Public interface for the prediction module.
    Takes a ticker string, returns a signal dict.

    Return structure (ALL 8 keys required, NEVER raise):
    {
        "ticker":     str,
        "signal":     "bullish" | "neutral" | "bearish" | "unknown",
        "confidence": float (0–100),
        "rsi":        float | None,
        "sma_cross":  bool  | None,
        "rf_signal":  str   | None,
        "disclaimer": str,           # must be non-empty
        "error":      str,           # empty string "" if no error
    }

    STUB — Prediction C replaces this body with the real implementation.
    The return structure must stay exactly as shown.
    """
    # ── STUB: returns unavailable so the API works without crashing ────────
    return {
        "ticker":     ticker.upper(),
        "signal":     "unavailable",
        "confidence": 0,
        "rsi":        None,
        "sma_cross":  None,
        "rf_signal":  None,
        "disclaimer": "Prediction module is under development.",
        "error":      "not implemented",
    }
    # ── END STUB ────────────────────────────────────────────────────────────
