import os
import sys
import pickle
import traceback
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# ── Path Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
for p in [ROOT_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from prediction.train import get_features, get_features_until_date, FEATURES
except ImportError:
    pred_dir = os.path.join(ROOT_DIR, "prediction")
    if pred_dir not in sys.path:
        sys.path.insert(0, pred_dir)
    from train import get_features, get_features_until_date, FEATURES

try:
    from Sentiment import get_latest_sentiment
    _SENTIMENT_AVAILABLE = True
except ImportError:
    _SENTIMENT_AVAILABLE = False

MODEL_PATH = os.path.join(SCRIPT_DIR, "saved_models", "signal_model.pkl")
_model = None


# ── Model Loading ─────────────────────────────────────────────────────────────
def _load_model() -> dict | None:
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        return _model
    except Exception:
        return None


def reload_model():
    global _model
    _model = None
    return _load_model()


# ── [FIX-5] Per-Ticker Ensemble Model Selector ───────────────────────────────
def _get_ticker_model(ticker: str, payload: dict) -> dict:
    
    ticker_models = payload.get("ticker_models", {})

    if ticker in ticker_models:
        tm = ticker_models[ticker]
        return {
            "classifier":            tm["classifier"],
            "classifier_rf":         tm.get("classifier_rf", tm["classifier"]),
            "classifier_xgb":        tm.get("classifier_xgb", tm["classifier"]),
            "regressor":             tm["regressor"],
            "clf_name":              tm["clf_name"],
            "reg_name":              tm["reg_name"],
            "feature_names":         tm["feature_names"],
            "optimal_clf_threshold": tm["optimal_clf_threshold"],
            "rf_threshold":          tm.get("rf_threshold", tm["optimal_clf_threshold"]),
            "xgb_threshold":         tm.get("xgb_threshold", tm["optimal_clf_threshold"]),
            "up_threshold":          tm.get("up_threshold", payload.get("up_threshold", 0.008)),
            "is_per_ticker":         True,
        }

    return {
        "classifier":            payload["classifier"],
        "classifier_rf":         payload.get("classifier_rf", payload["classifier"]),
        "classifier_xgb":        payload.get("classifier_xgb", payload["classifier"]),
        "regressor":             payload["regressor"],
        "clf_name":              payload.get("clf_name", "unknown"),
        "reg_name":              payload.get("reg_name", "unknown"),
        "feature_names":         payload["feature_names"],
        "optimal_clf_threshold": payload.get("optimal_clf_threshold", 0.55),
        "rf_threshold":          payload.get("rf_threshold", payload.get("optimal_clf_threshold", 0.55)),
        "xgb_threshold":         payload.get("xgb_threshold", payload.get("optimal_clf_threshold", 0.55)),
        "up_threshold":          payload.get("up_threshold", 0.008),
        "is_per_ticker":         False,
    }


def _get_ticker_threshold(ticker: str, payload: dict) -> float:
    thresholds = payload.get("ticker_thresholds", {})
    if ticker in thresholds:
        return float(thresholds[ticker])
    return float(payload.get("up_threshold", 0.005))


# ── Ticker Validation ─────────────────────────────────────────────────────────
def validate_ticker(ticker: str) -> tuple[bool, str]:
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 6:
        return False, f"Invalid ticker format: '{ticker}'"
    try:
        test = yf.Ticker(ticker)
        info = test.fast_info
        if not hasattr(info, 'last_price') or info.last_price is None:
            hist = test.history(period="5d")
            if hist.empty:
                return False, f"No price data found for '{ticker}'. Check the symbol."
        return True, ticker
    except Exception as e:
        return False, f"Ticker validation failed for '{ticker}': {e}"


# ── Feature Builder ───────────────────────────────────────────────────────────
def _build_inference_row(ticker: str, feature_names: list) -> tuple[np.ndarray, pd.DataFrame] | None:
    feat_df = get_features(ticker, period="2y")
    if feat_df.empty:
        return None

    latest = feat_df.iloc[[-1]].copy()
    row_data = {
        col: float(latest[col].values[0]) if col in latest.columns else 0.0
        for col in feature_names
    }
    row = pd.DataFrame([row_data], index=latest.index)
    return row.values, latest


# ── [FIX-5] Ensemble Prediction ───────────────────────────────────────────────
def _ensemble_predict(tm: dict, X: np.ndarray) -> tuple[float, float, str]:
   
    rf_clf  = tm.get("classifier_rf")
    xgb_clf = tm.get("classifier_xgb")
    rf_thr  = float(tm.get("rf_threshold", tm["optimal_clf_threshold"]))
    xgb_thr = float(tm.get("xgb_threshold", tm["optimal_clf_threshold"]))

    
    if rf_clf is None or xgb_clf is None:
        clf = tm["classifier"]
        proba    = clf.predict_proba(X)[0]
        up_proba = float(proba[1])
        opt_thr  = float(tm["optimal_clf_threshold"])
        return up_proba, opt_thr, "single"

    rf_proba  = float(rf_clf.predict_proba(X)[0][1])
    xgb_proba = float(xgb_clf.predict_proba(X)[0][1])

    
    avg_proba = 0.45 * rf_proba + 0.55 * xgb_proba
    avg_thr   = 0.45 * rf_thr + 0.55 * xgb_thr

    
    rf_buy  = rf_proba >= rf_thr
    xgb_buy = xgb_proba >= xgb_thr

    if rf_buy and xgb_buy:
        signal_type = "consensus_buy"
    elif rf_buy or xgb_buy:
        signal_type = "partial_buy"
    else:
        signal_type = "no_buy"

    return avg_proba, avg_thr, signal_type


# ── Jump Diffusion Monte Carlo ─────────────────────────────────────────────────
def _monte_carlo_forecast(
    current_price: float,
    predicted_log_return: float,
    hist_vol: float,
    n_trading_days: int,
    n_simulations: int = 2000,
    jump_intensity: float = 0.05,
    jump_mean: float = 0.0,
    jump_std: float = 0.03,
) -> list[dict]:
    rng = np.random.default_rng()
    daily_drift = predicted_log_return / max(n_trading_days, 1)

    Z = rng.standard_normal((n_simulations, n_trading_days))
    gbm_component = (daily_drift - 0.5 * hist_vol**2) + hist_vol * Z

    jumps_occur = rng.poisson(jump_intensity, (n_simulations, n_trading_days))
    jump_sizes  = rng.normal(jump_mean, jump_std, (n_simulations, n_trading_days))
    jump_component = jumps_occur * jump_sizes

    daily_log_ret = gbm_component + jump_component
    paths = current_price * np.exp(np.cumsum(daily_log_ret, axis=1))

    forecast = []
    today    = datetime.date.today()
    trading_day = 0
    cal_day     = 0

    while trading_day < n_trading_days:
        cal_day += 1
        future_date = today + timedelta(days=cal_day)
        if future_date.weekday() >= 5:
            continue
        pt = paths[:, trading_day]
        forecast.append({
            "date":  future_date.strftime("%Y-%m-%d"),
            "price": round(float(np.mean(pt)), 2),
            "upper": round(float(np.percentile(pt, 90)), 2),
            "lower": round(float(np.percentile(pt, 10)), 2),
            "p50":   round(float(np.median(pt)), 2),
            "p75":   round(float(np.percentile(pt, 75)), 2),
            "p25":   round(float(np.percentile(pt, 25)), 2),
        })
        trading_day += 1

    return forecast


# ── Cost Simulation ───────────────────────────────────────────────────────────
def _simulate_costs(hist_vol: float, current_price: float,
                    predicted_pct_return: float) -> dict:
    commission = 0.0010
    slippage   = min(0.0020, 0.0004 + hist_vol * 1.5)
    spread     = 0.0002
    total_cost = commission + slippage + spread
    net_return = predicted_pct_return - (total_cost * 100)
    return {
        "commission":           round(commission * 100, 3),
        "slippage":             round(slippage * 100, 3),
        "spread":               round(spread * 100, 3),
        "total_cost":           round(total_cost * 100, 3),
        "net_return":           round(net_return, 3),
        "net_target":           round(current_price * (1 + net_return / 100), 4),
        "break_even":           round(total_cost * 100, 3),
        "cost_adjusted_signal": (
            "HOLD (net return < cost)" if net_return <= 0 else "PASS"
        ),
    }


# ── Confidence Interpretation ─────────────────────────────────────────────────
def _interpret_confidence(confidence: float, opt_threshold: float) -> str:
    if confidence >= opt_threshold + 0.12:
        return "HIGH"
    elif confidence >= opt_threshold + 0.06:
        return "MEDIUM"
    elif confidence >= opt_threshold:
        return "LOW"
    else:
        return "BELOW THRESHOLD"


# ── Main Prediction ───────────────────────────────────────────────────────────
def predict_signal(ticker: str) -> dict:
    result = {
        "ticker":           ticker,
        "signal":           "ERROR",
        "direction":        None,
        "confidence":       0.0,
        "confidence_level": None,
        "price_target":     None,
        "current_price":    None,
        "expected_return":  None,
        "lookahead_days":   None,
        "model_clf":        None,
        "model_reg":        None,
        "model_version":    None,
        "trained_at":       None,
        "is_per_ticker":    False,
        "error":            None,
        "sentiment":        {},
        "forecast_series":  [],
        "hist_vol":         None,
        "up_threshold":     None,
        "opt_threshold":    None,
        "costs":            {},
        "ensemble_type":    None,
        "reg_r2":           None,
        "reg_quality":      None,
    }

    try:
        payload = _load_model()
        if payload is None:
            result["error"] = f"Model not found at '{MODEL_PATH}'. Run Train.py first."
            return result

        lookahead = int(payload.get("lookahead", 5))

        tm            = _get_ticker_model(ticker, payload)
        reg           = tm["regressor"]
        feature_names = tm["feature_names"]
        opt_threshold = tm["optimal_clf_threshold"]
        up_thresh     = tm["up_threshold"]
        is_per_ticker = tm["is_per_ticker"]

        result["lookahead_days"]  = lookahead
        result["model_clf"]       = tm["clf_name"]
        result["model_reg"]       = tm["reg_name"]
        result["model_version"]   = payload.get("version", "unknown")
        result["trained_at"]      = payload.get("trained_at", "unknown")
        result["opt_threshold"]   = opt_threshold
        result["is_per_ticker"]   = is_per_ticker
        result["up_threshold"]    = up_thresh

        build_result = _build_inference_row(ticker, feature_names)
        if build_result is None:
            result["error"] = f"Feature build failed for '{ticker}'."
            return result

        X, latest = build_result

        # Sentiment
        live_sentiment = {}
        if _SENTIMENT_AVAILABLE:
            try:
                live_sentiment = get_latest_sentiment(ticker, days_back=7)
            except Exception:
                pass
        result["sentiment"] = live_sentiment

        # ── [FIX-5] Ensemble Classifier ───────────────────────────────────────
        avg_proba, ens_thr, ensemble_type = _ensemble_predict(tm, X)

        up_proba   = avg_proba
        down_proba = 1.0 - up_proba
        direction  = "UP" if up_proba > down_proba else "DOWN"
        confidence = up_proba if direction == "UP" else down_proba

        result["direction"]        = direction
        result["confidence"]       = round(confidence, 4)
        result["up_probability"]   = round(up_proba, 4)
        result["down_probability"] = round(down_proba, 4)
        result["ensemble_type"]    = ensemble_type

        # ── [v13] Regressor مع تحقق من جودته ─────────────────────────────────
        reg_r2 = float(tm.get("reg_r2", 0.0))
        predicted_log_return = float(reg.predict(X)[0])

        if reg_r2 < -0.3:
            # [v13] R² سالب جداً → النموذج أسوأ من المتوسط
            # نستخدم إشارة من الـ classifier بدلاً منه:
            # up_proba > 0.5 → نتوقع عائد موجب صغير، وإلا سالب صغير
            sign = 1.0 if up_proba > 0.5 else -1.0
            # نقدّر حجم الحركة بالـ volatility التاريخية (سيُحسب لاحقاً)
            # نضع قيمة صفر مؤقتاً ونحدّثها بعد حساب hist_vol
            predicted_log_return = sign * 0.003  # افتراض حركة محدودة ±0.3%
            result["reg_quality"] = f"poor (R²={reg_r2:.3f}) — using classifier direction"
        else:
            result["reg_quality"] = f"good (R²={reg_r2:.3f})"

        predicted_pct_return = (np.exp(predicted_log_return) - 1) * 100
        result["expected_return"] = round(predicted_pct_return, 4)
        result["reg_r2"]          = round(reg_r2, 4)

        # ── Price & Historical Vol ─────────────────────────────────────────────
        hist = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [col[0] for col in hist.columns]
            current_price = float(hist["Close"].iloc[-1])
            log_ret       = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            hist_vol      = float(log_ret.tail(60).std()) if len(log_ret) > 1 else 0.02
        else:
            current_price = 0.0
            hist_vol      = 0.02
            log_ret       = pd.Series(dtype=float)

        # GARCH Vol Forecast
        if len(log_ret) > 30:
            try:
                from arch import arch_model as _arch_model
                _garch  = _arch_model(log_ret * 100, vol='Garch', p=1, q=1)
                _gfit   = _garch.fit(disp='off')
                _fcast  = _gfit.forecast(horizon=1)
                hist_vol = float(np.sqrt(_fcast.variance.values[-1, 0]) / 100)
            except Exception:
                hist_vol = float(log_ret.tail(60).std())

        result["current_price"] = round(current_price, 4)
        result["hist_vol"]      = round(hist_vol, 6)
        result["price_target"]  = round(current_price * np.exp(predicted_log_return), 4)

        # ── [FIX-5] Signal Logic: Consensus أو Partial ────────────────────────
        result["confidence_level"] = _interpret_confidence(up_proba, ens_thr)

        sell_threshold = max(0.50, 1.0 - ens_thr)

        if ensemble_type == "consensus_buy":
            # كلا النموذجين متفقان → BUY
            result["signal"] = "BUY"
        elif ensemble_type == "partial_buy":
            # نموذج واحد فقط → HOLD (أكثر حذراً)
            result["signal"] = "HOLD"
        elif up_proba >= ens_thr:
            # fallback للنموذج الواحد
            result["signal"] = "BUY"
        elif up_proba <= (1.0 - sell_threshold):
            result["signal"] = "SELL"
        else:
            result["signal"] = "HOLD"

        # ── Monte Carlo ────────────────────────────────────────────────────────
        result["forecast_series"] = _monte_carlo_forecast(
            current_price=current_price,
            predicted_log_return=predicted_log_return,
            hist_vol=hist_vol,
            n_trading_days=lookahead,
            n_simulations=2000,
        )

        # ── Cost Simulation ────────────────────────────────────────────────────
        result["costs"] = _simulate_costs(hist_vol, current_price, predicted_pct_return)

        result["error"] = None

    except Exception as exc:
        result["error"]  = f"Inference error: {exc}\n{traceback.format_exc()}"
        result["signal"] = "ERROR"

    return result


def predict_signals_batch(tickers: list[str]) -> list[dict]:
    results = []
    for t in tickers:
        print(f"  Predicting {t}...")
        r = predict_signal(t)
        results.append(r)

    def sort_key(r):
        sig   = r.get("signal", "HOLD")
        conf  = r.get("confidence", 0.0)
        etype = r.get("ensemble_type", "")
        # Consensus BUY أعلى أولوية
        order = {"BUY": 0, "HOLD": 1, "SELL": 2, "ERROR": 3}
        consensus_bonus = -0.1 if etype == "consensus_buy" else 0
        return (order.get(sig, 3) + consensus_bonus, -conf)

    results.sort(key=sort_key)
    return results


def model_info() -> dict:
    payload = _load_model()
    if payload is None:
        return {"error": f"No model at {MODEL_PATH}"}

    ticker_metrics  = payload.get("ticker_metrics", {})
    tickers_trained = payload.get("tickers_trained_on", [])

    return {
        "version":               payload.get("version"),
        "trained_at":            payload.get("trained_at"),
        "tickers_trained_on":    tickers_trained,
        "n_tickers":             len(tickers_trained),
        "failed_tickers":        payload.get("failed_tickers", []),
        "n_folds":               payload.get("n_folds"),
        "fbeta":                 payload.get("fbeta"),
        "min_precision":         payload.get("min_precision"),
        "lookahead_days":        payload.get("lookahead"),
        "up_threshold":          payload.get("up_threshold"),
        "optimal_clf_threshold": payload.get("optimal_clf_threshold"),
        "use_sentiment":         payload.get("use_sentiment"),
        "ticker_metrics": {
            t: {
                "precision": round(m.get("overall_precision", 0), 3),
                "accuracy":  round(m.get("overall_accuracy", 0), 3),
                "up_pct":    round(m.get("up_pct", 0), 3),
                "n_rows":    m.get("n_rows", 0),
            }
            for t, m in ticker_metrics.items()
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="X-INVEST Predict v13")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--batch",  nargs="+", default=None)
    parser.add_argument("--info",   action="store_true")
    args = parser.parse_args()

    if args.info:
        info = model_info()
        if "error" in info:
            print(f"❌ {info['error']}")
        else:
            print(f"\n{'='*60}")
            print(f"  X-INVEST Model Info (v{info['version']})")
            print(f"{'='*60}")
            print(f"  Trained At    : {str(info['trained_at'])[:16]}")
            print(f"  Tickers       : {info['n_tickers']} stocks")
            print(f"  Lookahead     : {info['lookahead_days']}d")
            print(f"  CV Folds      : {info['n_folds']}")
            print(f"  F-beta        : {info['fbeta']}")
            print(f"  Min Precision : {info.get('min_precision', 'N/A')}")
            if info['ticker_metrics']:
                print(f"\n  Per-Ticker Performance:")
                print(f"  {'Ticker':<8} {'Prec':>7} {'Acc':>7} {'UP%':>6} {'Rows':>6}")
                print(f"  {'-'*38}")
                for t, m in sorted(info['ticker_metrics'].items(),
                                   key=lambda x: x[1]['precision'], reverse=True):
                    print(f"  {t:<8} {m['precision']:>7.3f} {m['accuracy']:>7.3f} "
                          f"{m['up_pct']:>6.1%} {m['n_rows']:>6}")
        import sys; sys.exit(0)

    if args.batch:
        results = predict_signals_batch(args.batch)
        print(f"\n{'Ticker':<8} {'Signal':<6} {'Ensemble':<16} {'Conf':>6} {'UP%':>6} {'Ret%':>7}")
        print("-" * 65)
        for r in results:
            icon  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "ERROR": "❌"}.get(r["signal"], "⚪")
            etype = r.get("ensemble_type", "N/A")
            print(f"{r['ticker']:<8} {icon}{r['signal']:<5} {etype:<16} "
                  f"{r.get('confidence', 0):>6.3f} "
                  f"{r.get('up_probability', 0):>6.3f} "
                  f"{r.get('expected_return', 0):>+7.2f}%")
        import sys; sys.exit(0)

    raw_ticker = args.ticker or input("\n📈 Ticker Symbol (e.g. AAPL, TSLA): ").strip()
    print(f"  🔍 Validating '{raw_ticker.upper()}'...")
    valid, ticker_or_err = validate_ticker(raw_ticker)
    if not valid:
        print(f"\n❌ {ticker_or_err}")
        import sys; sys.exit(0)

    ticker = ticker_or_err
    print(f"\n{'='*60}\n  📊 Prediction Report: {ticker}\n{'='*60}")

    try:
        r = predict_signal(ticker)

        if r["signal"] == "ERROR":
            print(f"❌ {r['error']}")
        else:
            icon  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(r["signal"], "⚪")
            lk    = r.get("lookahead_days", "?")
            opt   = r.get("opt_threshold", 0.55)
            etype = r.get("ensemble_type", "N/A")

            print(f"\n  {icon} Signal               : {r['signal']} [{etype}]")
            print(f"  📉 Direction             : {r['direction']}")
            print(f"  📊 UP Probability        : {r.get('up_probability', 0):.1%}")
            print(f"  📊 DOWN Probability      : {r.get('down_probability', 0):.1%}")
            print(f"  🎚  Optimal Threshold     : {opt:.3f}")
            print(f"  ⭐ Confidence Level      : {r.get('confidence_level', 'N/A')}")
            print(f"  ⏱  Forecast Horizon       : {lk} trading day(s)")
            print(f"  💰 Current Price         : ${r['current_price']:.2f}")
            print(f"  🎯 Price Target          : ${r['price_target']:.2f}")
            print(f"  📈 Expected Return        : {r['expected_return']:+.2f}%")
            print(f"  📐 Hist. Volatility (60d) : {(r['hist_vol'] or 0)*100:.2f}% daily")
            print(f"  🎚  Adaptive Threshold     : {(r['up_threshold'] or 0)*100:.2f}%")

            info = model_info()
            tm   = info.get("ticker_metrics", {}).get(ticker, {})
            print(f"\n  🤖 Model Info:")
            print(f"     Version      : {r.get('model_version', 'N/A')}")
            print(f"     Classifier   : {r['model_clf']}")
            print(f"     Regressor    : {r['model_reg']} [{r.get('reg_quality','N/A')}]")
            print(f"     Trained At   : {str(r.get('trained_at', 'N/A'))[:16]}")
            if tm:
                print(f"     Precision    : {tm.get('precision', 0):.3f} (Walk-Forward)")
                print(f"     Accuracy     : {tm.get('accuracy', 0):.3f} (Walk-Forward)")

            costs = r.get("costs", {})
            if costs:
                print(f"\n  💸 Cost Simulation:")
                print(f"     Commission : {costs['commission']:.3f}%")
                print(f"     Slippage   : {costs['slippage']:.3f}%")
                print(f"     Total Cost : {costs['total_cost']:.3f}%")
                print(f"     Net Return : {costs['net_return']:+.3f}%")
                print(f"     Decision   : {costs['cost_adjusted_signal']}")

            sent = r.get("sentiment", {})
            if sent and sent.get("n_articles", 0) > 0:
                print(f"\n  🗞️  Sentiment (last 7d):")
                score  = sent.get('sentiment_score', 0)
                icon_s = "📈" if score > 0.05 else "📉" if score < -0.05 else "➡️"
                print(f"     Score    : {score:+.3f} {icon_s}")
                print(f"     Articles : {sent['n_articles']}")

            fc = r.get("forecast_series", [])
            if fc:
                last = fc[-1]
                print(f"\n  🔮 {lk}-Day Forecast (Jump Diffusion):")
                print(f"     P50 : ${last.get('p50', last['price']):.2f}  (median)")
                print(f"     P75 : ${last.get('p75', last['upper']):.2f}")
                print(f"     P25 : ${last.get('p25', last['lower']):.2f}")
                print(f"     P90 : ${last['upper']:.2f}")
                print(f"     P10 : ${last['lower']:.2f}")

            if r["signal"] == "BUY":
                if etype == "consensus_buy":
                    print(f"\n  ✅ Consensus BUY — كلا النموذجين RF و XGB متفقان.")
                else:
                    print(f"\n  ⚠️  Partial signal — اعتبره HOLD إذا كنت محافظاً.")
            conf_level = r.get('confidence_level', '')
            if "LOW" in conf_level or "BELOW" in conf_level:
                print(f"  ⚠️  Confidence {conf_level} — consider paper trading first.")

    except Exception as e:
        print(f"⚠️  Error: {e}")
        traceback.print_exc()

    print(f"{'='*60}\n")