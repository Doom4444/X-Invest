import os, sys, argparse, datetime, time, traceback
import numpy as np
import pandas as pd
import yfinance as yf

# ── Path setup ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
for p in [ROOT_DIR, SCRIPT_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

try:
    from predict import _load_model
    from Train   import get_features
    IMPORTS_OK = True
except ImportError:
    try:
        pred_dir = os.path.join(ROOT_DIR, "prediction")
        if pred_dir not in sys.path: sys.path.insert(0, pred_dir)
        from predict import _load_model
        from Train   import get_features
        IMPORTS_OK = True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────
FORECAST_DAYS = 30
N_SIM         = 3000
DEFAULT_TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','NVDA',
    'META','TSLA','BRK-B','UNH','JNJ',
    'V','PG','JPM','HD','MA',
    'CVX','MRK','PEP','KO','ABBV',
]
COMPANY_NAMES = {
    'AAPL':'Apple Inc.','MSFT':'Microsoft Corp.','GOOGL':'Alphabet Inc.',
    'AMZN':'Amazon.com Inc.','NVDA':'NVIDIA Corp.','META':'Meta Platforms',
    'TSLA':'Tesla Inc.','BRK-B':'Berkshire Hathaway','UNH':'UnitedHealth Group',
    'JNJ':'Johnson & Johnson','V':'Visa Inc.','PG':'Procter & Gamble',
    'JPM':'JPMorgan Chase','HD':'Home Depot','MA':'Mastercard Inc.',
    'CVX':'Chevron Corp.','MRK':'Merck & Co.','PEP':'PepsiCo Inc.',
    'KO':'Coca-Cola Co.','ABBV':'AbbVie Inc.',
}
SECTORS = {
    'AAPL':'Technology','MSFT':'Technology','GOOGL':'Technology',
    'NVDA':'Technology','META':'Technology','AMZN':'Consumer',
    'TSLA':'Consumer','HD':'Consumer','UNH':'Healthcare',
    'JNJ':'Healthcare','MRK':'Healthcare','ABBV':'Healthcare',
    'JPM':'Financial','V':'Financial','MA':'Financial',
    'BRK-B':'Financial','PG':'Staples','PEP':'Staples',
    'KO':'Staples','CVX':'Energy',
}

def _trading_dates(n: int) -> list[datetime.date]:
    dates, d, offset = [], datetime.date.today(), 1
    while len(dates) < n:
        day = d + datetime.timedelta(days=offset)
        if day.weekday() < 5:
            dates.append(day)
        offset += 1
    return dates

def _monte_carlo_paths(start_price, hist_vol, daily_drift, n_days=FORECAST_DAYS, n_sim=N_SIM):
    rng          = np.random.default_rng(seed=42)
    Z            = rng.standard_normal((n_sim, n_days))
    jumps_occur  = rng.poisson(0.05, (n_sim, n_days))
    jump_sizes   = rng.normal(0.0, 0.03, (n_sim, n_days))
    daily_log    = (daily_drift - 0.5 * hist_vol**2) + hist_vol * Z + jumps_occur * jump_sizes
    paths        = start_price * np.exp(np.cumsum(daily_log, axis=1))
    return paths

def predict_30d(ticker: str, payload: dict) -> list[dict]:
    # ── نموذج السهم ──────────────────────────────────────────────
    tm_map = payload.get("ticker_models", {})
    if ticker in tm_map:
        tm        = tm_map[ticker]
        clf       = tm["classifier"]
        reg       = tm["regressor"]
        feat_list = tm["feature_names"]
        opt_thr   = tm.get("optimal_clf_threshold", 0.55)
    elif "classifier" in payload:
        clf       = payload["classifier"]
        reg       = payload["regressor"]
        feat_list = payload["feature_names"]
        opt_thr   = payload.get("optimal_clf_threshold", 0.55)
    else:
        raise ValueError(f"No model for {ticker}")

    
    feat_df = get_features(ticker, period="1y")
    if feat_df.empty: raise ValueError(f"Empty features for {ticker}")
    latest = feat_df.iloc[[-1]].copy()
    row_data = {c: float(latest[c].values[0]) if c in latest.columns else 0.0 for c in feat_list}
    X = pd.DataFrame([row_data]).values 

    hist = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = [c[0] for c in hist.columns]
    if hist.empty: raise ValueError(f"No price data for {ticker}")
    today_price = float(hist["Close"].iloc[-1])

    # Volatility
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    hist_vol = float(log_ret.tail(60).std()) if len(log_ret) > 1 else 0.02

    
    try:
        raw = float(reg.predict(X)[0])
        daily_drift = raw / FORECAST_DAYS
    except Exception:
        daily_drift = 0.0

    
    paths = _monte_carlo_paths(
        start_price=today_price, hist_vol=hist_vol,
        daily_drift=daily_drift, n_days=FORECAST_DAYS + 1, n_sim=N_SIM
    )
    dates = _trading_dates(FORECAST_DAYS + 1)

   
    rows = []
    for i in range(FORECAST_DAYS):
        col_today = paths[:, i]       
        col_next  = paths[:, i + 1]  

       
        up_count = np.sum(col_next > col_today)
        up_proba_daily = up_count / N_SIM
        dn_proba_daily = 1.0 - up_proba_daily

        p50_today = round(float(np.median(col_today)), 2)
        p50_next  = round(float(np.median(col_next)), 2)
        exp_ret   = round((p50_next - p50_today) / p50_today * 100, 4) if p50_today else 0.0

        
        direction = "UP" if up_proba_daily >= 0.5 else "DOWN"
        
        if up_proba_daily >= opt_thr:
            signal = "BUY"
        elif dn_proba_daily >= opt_thr:
            signal = "SELL"
        else:
            signal = "HOLD"

        rows.append({
            "ticker":              ticker,
            "company_name":        COMPANY_NAMES.get(ticker, ticker),
            "sector":              SECTORS.get(ticker, "Unknown"),
            "forecast_date":       dates[i].strftime("%Y-%m-%d"),
            "current_price":       p50_today,
            "price_target":        p50_next,
            "expected_return_pct": exp_ret,
            "up_probability":      round(up_proba_daily, 4),
            "down_probability":    round(dn_proba_daily, 4),
            "opt_threshold":       round(opt_thr, 4),
            "signal":              signal,
            "direction":           direction,
        })
    return rows

def run_export(tickers: list[str], output_path: str, delay: float = 2.0):
    print("=" * 60)
    print("  X-INVEST — 30-Day Dynamic Prediction Export (Fixed)")
    print(f"  Tickers : {len(tickers)}  |  Days each : {FORECAST_DAYS}")
    print(f"  Output  : {output_path}")
    print("=" * 60)
    
    payload = _load_model()
    if payload is None:
        print("❌ Model not found. Run Train.py first.")
        sys.exit(1)

    all_rows, failed = [], []
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i:02d}/{len(tickers)}] {ticker:<8} ({COMPANY_NAMES.get(ticker,'')})")
        try:
            rows = predict_30d(ticker, payload)
            all_rows.extend(rows)
            r0, r29 = rows[0], rows[-1]
            icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(r0["signal"], "⚪")
            print(f"  {icon} Day1: ${r0['current_price']:.2f} ({r0['signal']}/{r0['direction']}) → Day30: ${r29['current_price']:.2f}")
        except Exception as e:
            print(f"  ❌ {e}")
            traceback.print_exc()
            failed.append(ticker)
        if i < len(tickers): time.sleep(delay)

    if not all_rows:
        print("\n❌ No data.")
        return None

    df = pd.DataFrame(all_rows)
    cols = ["ticker","company_name","sector","forecast_date",
            "current_price","price_target","expected_return_pct",
            "up_probability","down_probability","opt_threshold","signal","direction"]
    df = df[[c for c in cols if c in df.columns]]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}\n  ✅ Saved  : {output_path}\n  📊 Rows   : {len(df):,}")
    if failed: print(f"  ⚠  Failed : {failed}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--output", type=str,
        default=os.path.join(SCRIPT_DIR, "exports", f"predictions_30d_{datetime.date.today().strftime('%Y%m%d')}.csv"))
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    run_export(tickers=args.tickers, output_path=args.output, delay=args.delay)