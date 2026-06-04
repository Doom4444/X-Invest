"""
export_predictions.py — X-INVEST
بيولّد تنبؤات يومية حقيقية لآخر 30 يوم تداول لكل شركة،
ويصدّرها في CSV واحد مرتب بالتاريخ.

المنهجية:
  - لكل يوم من آخر 30 يوم تداول، بنبني الـ features من البيانات المتاحة
    حتى ذلك اليوم فقط (بدون lookahead).
  - ده يعكس بالضبط إيه اللي كان المودل سيقوله لو شغّلته كل يوم فعلاً.
  - النتيجة: تاريخ تنبؤي واقعي وغير ثابت لكل شركة.
"""

import os
import sys
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Path Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
for p in [ROOT_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── [FIX] Dynamic path detection ─────────────────────────────────────────────
# الملف ممكن يكون في الـ Root أو داخل prediction/ — بنكتشف المكان الصح تلقائياً
def _find_model_path() -> str:
    candidates = [
        os.path.join(SCRIPT_DIR, "prediction", "saved_models", "signal_model.pkl"),
        os.path.join(SCRIPT_DIR, "saved_models", "signal_model.pkl"),
        os.path.join(ROOT_DIR,   "prediction", "saved_models", "signal_model.pkl"),
        os.path.join(ROOT_DIR,   "saved_models", "signal_model.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # لو مش موجود في أي مكان، نرجع الـ path المتوقع الأول للرسالة
    return candidates[0]

MODEL_PATH = _find_model_path()
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "predictions_30d.csv")

# ── [FIX] Smart Train import — بيجرب كل المسارات الممكنة ────────────────────
def _import_get_features():
    import importlib, types

    # المسارات الممكنة لـ Train.py
    search_dirs = [
        os.path.join(SCRIPT_DIR, "prediction"),
        SCRIPT_DIR,
        ROOT_DIR,
        os.path.join(ROOT_DIR, "prediction"),
    ]
    for d in search_dirs:
        train_file = os.path.join(d, "Train.py")
        if os.path.exists(train_file):
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                import prediction.train
                importlib.reload(prediction.train)
                return prediction.train.get_features
            except Exception as e:
                print(f"  ⚠ Import from {d}: {e}")

    sys.exit("❌ Train.py مش موجود. تأكد من هيكل المشروع.")

get_features = _import_get_features()

# ── الشركات المطلوبة ──────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "JPM",'BRK-B', 'UNH', 'JNJ',
            'V', 'PG', 'HD', 'MA',
            'CVX', 'MRK', 'PEP', 'KO', 'ABBV']

# ── تحميل المودل ─────────────────────────────────────────────────────────────
def load_model() -> dict:
    if not os.path.exists(MODEL_PATH):
        print("❌ المودل مش موجود. تم البحث في:")
        print(f"   • {os.path.join(SCRIPT_DIR, 'prediction', 'saved_models', 'signal_model.pkl')}")
        print(f"   • {os.path.join(SCRIPT_DIR, 'saved_models', 'signal_model.pkl')}")
        print(f"   • {os.path.join(ROOT_DIR, 'prediction', 'saved_models', 'signal_model.pkl')}")
        sys.exit("   شغّل Train.py أولاً.")
    print(f"✓ Model loaded from: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ── استخراج آخر 30 يوم تداول ─────────────────────────────────────────────────
def get_last_trading_days(n: int = 30) -> list[datetime.date]:
    """بيرجع آخر n يوم تداول فعلي (بيستثني الويكند والإجازات من السوق)."""
    # نحمّل بيانات SPY كمرجع للأيام الفعلية
    spy = yf.download("SPY", period="3mo", progress=False, auto_adjust=True)
    if spy.empty:
        # fallback: آخر 30 يوم تقويمي بدون weekends
        days = []
        d = datetime.date.today() - datetime.timedelta(days=1)
        while len(days) < n:
            if d.weekday() < 5:
                days.append(d)
            d -= datetime.timedelta(days=1)
        return sorted(days)
    dates = spy.index.normalize().unique().tolist()
    dates = sorted([d.date() for d in dates])
    return dates[-n:]

# ── تنبؤ يوم واحد باستخدام features حتى ذلك اليوم ──────────────────────────
def predict_for_date(
    ticker: str,
    target_date: datetime.date,
    feat_df: pd.DataFrame,
    payload: dict,
) -> dict | None:
    """
    بيبني features من البيانات المتاحة حتى target_date فقط
    (بدون أي نظرة للمستقبل) ويرجع التنبؤ.
    """
    clf        = payload["classifier"]
    reg        = payload["regressor"]
    feat_names = payload["feature_names"]
    opt_thr    = float(payload.get("optimal_clf_threshold", 0.50))

    # بنستخدم البيانات المتاحة حتى target_date فقط
    target_ts = pd.Timestamp(target_date)
    hist      = feat_df[feat_df.index <= target_ts]

    if hist.empty or len(hist) < 20:
        return None

    # آخر صف متاح = البيانات المعروفة في نهاية target_date
    latest = hist.iloc[[-1]]

    row_data = {
        col: float(latest[col].values[0]) if col in latest.columns else 0.0
        for col in feat_names
    }
    X = np.array([[row_data[c] for c in feat_names]])

    try:
        proba       = clf.predict_proba(X)[0]
        up_prob     = float(proba[1])
        down_prob   = float(proba[0])
        log_ret     = float(reg.predict(X)[0])
        pct_ret     = float(np.exp(log_ret) - 1) * 100

        # السعر الفعلي في target_date
        close_col = "close" if "close" in hist.columns else None
        price = float(hist["close"].iloc[-1]) if close_col else None

        # Signal logic (نفس backtest.py)
        if up_prob >= opt_thr:
            signal = "BUY"
        elif down_prob >= opt_thr:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Confidence level
        diff = up_prob - opt_thr
        if diff >= 0.15:
            conf_level = "HIGH"
        elif diff >= 0.07:
            conf_level = "MEDIUM"
        elif diff >= 0:
            conf_level = "LOW"
        else:
            conf_level = "BELOW_THRESHOLD"

        return {
            "date":             target_date.strftime("%Y-%m-%d"),
            "ticker":           ticker,
            "signal":           signal,
            "up_probability":   round(up_prob * 100, 2),
            "down_probability": round(down_prob * 100, 2),
            "expected_return_pct": round(pct_ret, 4),
            "price_at_signal":  round(price, 2) if price else None,
            "confidence_level": conf_level,
            "opt_threshold":    round(opt_thr, 3),
        }

    except Exception as e:
        print(f"  ⚠ {ticker} @ {target_date}: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("X-INVEST — Rolling 30-Day Predictions Export")
    print("=" * 60)

    payload      = load_model()
    trading_days = get_last_trading_days(30)
    print(f"📅 Period: {trading_days[0]} → {trading_days[-1]} ({len(trading_days)} days)")
    print(f"📈 Tickers: {', '.join(TICKERS)}\n")

    all_rows = []

    for ticker in TICKERS:
        print(f"🔄 Processing {ticker}...")

        # نحمّل features مرة واحدة لكل شركة (كفاءة)
        feat_df = get_features(ticker, period="2y")
        if feat_df.empty:
            print(f"  ⚠ لا بيانات لـ {ticker}, skipping.")
            continue

        feat_df.index = pd.to_datetime(feat_df.index)
        feat_df.index = feat_df.index.tz_localize(None)  # إزالة timezone

        ticker_rows = []
        for day in trading_days:
            row = predict_for_date(ticker, day, feat_df, payload)
            if row:
                ticker_rows.append(row)

        n_buy  = sum(1 for r in ticker_rows if r["signal"] == "BUY")
        n_hold = sum(1 for r in ticker_rows if r["signal"] == "HOLD")
        n_sell = sum(1 for r in ticker_rows if r["signal"] == "SELL")
        print(f"  ✓ {len(ticker_rows)} days | BUY:{n_buy} HOLD:{n_hold} SELL:{n_sell}")

        all_rows.extend(ticker_rows)

    if not all_rows:
        print("\n❌ لا توجد بيانات للتصدير.")
        return

    # ── بناء الـ DataFrame وترتيبه ──────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ── ترتيب الأعمدة ────────────────────────────────────────────────────────
    cols = [
        "date", "ticker", "signal", "confidence_level",
        "up_probability", "down_probability",
        "expected_return_pct", "price_at_signal", "opt_threshold",
    ]
    df = df[cols]

    # ── تصدير CSV ────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"✅ CSV saved → {OUTPUT_CSV}")
    print(f"   Rows: {len(df)} | Tickers: {df['ticker'].nunique()} | Days: {df['date'].nunique()}")
    print(f"\n📊 Signal Distribution:")
    dist = df.groupby(["ticker", "signal"]).size().unstack(fill_value=0)
    print(dist.to_string())
    print(f"\n📊 Sample (last 5 rows):")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df.tail(5).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()