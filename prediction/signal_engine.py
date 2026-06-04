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

# prediction/signal_engine.py
from prediction.predict import predict_signal as _predict


def get_signal(ticker: str) -> dict:
    try:
        result = _predict(ticker)
        if result is None:
            return {
                "ticker":     ticker.upper(),
                "signal":     "unknown",
                "confidence": 0,
                "rsi":        None,
                "sma_cross":  None,
                "rf_signal":  None,
                "disclaimer": "Prediction model is not trained yet. Run train.py first.",
                "error":      "model not loaded",
            }

        # Map their signal names to the contract
        # predict.py returns BUY/HOLD/SELL — contract requires bullish/neutral/bearish
        signal_map = {
            "BUY":  "bullish",
            "HOLD": "neutral",
            "SELL": "bearish",
        }

        return {
            "ticker":     ticker.upper(),
            "signal":     signal_map.get(result.get("signal", ""), "unknown"),
            "confidence": round(result.get("confidence", 0) * 100, 1),
            "rsi":        result.get("rsi", None),
            "sma_cross":  result.get("sma_cross", None),
            "rf_signal":  signal_map.get(result.get("rf_signal", ""), None),
            "disclaimer": "Technical analysis only. Not financial advice.",
            "error":      "",
        }

    except Exception as e:
        return {
            "ticker":     ticker.upper(),
            "signal":     "unknown",
            "confidence": 0,
            "rsi":        None,
            "sma_cross":  None,
            "rf_signal":  None,
            "disclaimer": "Technical analysis only. Not financial advice.",
            "error":      str(e),
        }
    # ── END STUB ────────────────────────────────────────────────────────────
