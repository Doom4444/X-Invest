import os
import sys
import argparse
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from prediction.predict import predict_signal, _load_model, model_info
except ImportError:
    try:
        from prediction.predict import predict_signal, _load_model, model_info
    except ImportError as e:
        print(f"Error: Could not import predict.py: {e}")
        sys.exit(1)

DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'BRK-B', 'UNH',  'JNJ',
    'V',    'PG',   'JPM',   'HD',   'MA',
    'CVX',  'MRK',  'PEP',   'KO',   'ABBV',
]

OUTPUT_PATH = os.path.join(SCRIPT_DIR, "dashboard_data_v2.csv")

SIGNALS      = ["BUY", "HOLD", "SELL"]
SIGNAL_TRANS = {
    "BUY":  [0.55, 0.25, 0.20],
    "HOLD": [0.25, 0.40, 0.35],
    "SELL": [0.20, 0.25, 0.55],
}
DIR_MAP = {"BUY": "UP", "HOLD": "FLAT", "SELL": "DOWN"}


def _is_trading_day(d) -> bool:
    return d.weekday() < 5


def _next_n_trading_days(start, n: int) -> list:
    days, d = [], start
    while len(days) < n:
        if _is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def _simulate_daily_series(base_result: dict, trade_days: list, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)

    hist_vol      = base_result.get("hist_vol") or 0.013
    base_price    = base_result.get("current_price", 100) or 100
    base_ret      = base_result.get("expected_return", 0) or 0
    lookahead     = base_result.get("lookahead_days") or 5
    init_sig      = base_result.get("signal", "HOLD")
    if init_sig not in SIGNALS:
        init_sig = "HOLD"

    rows          = []
    current_price = base_price
    current_sig   = init_sig

    for day_idx, forecast_day in enumerate(trade_days):
        if day_idx > 0:
            daily_return  = rng.normal(loc=0.0, scale=hist_vol)
            current_price = round(current_price * (1 + daily_return), 4)
            current_sig   = rng.choice(SIGNALS, p=SIGNAL_TRANS[current_sig])

        direction       = DIR_MAP[current_sig]
        daily_ret_est   = base_ret / max(lookahead, 1)
        adj_daily_ret   = daily_ret_est * (1 + rng.normal(0, 0.05))
        expected_price  = round(current_price * (1 + adj_daily_ret / 100), 4)

        rows.append({
            "date":                forecast_day.strftime("%Y-%m-%d"),
            "ticker":              base_result.get("ticker", "?"),
            "direction":           direction,
            "signal":              current_sig,
            "current_price":       round(current_price, 4),
            "expected_price":      expected_price,
            "expected_return_pct": round(adj_daily_ret, 4),
        })

    return rows


def generate(tickers: list = None, n_days: int = 30, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    tickers    = tickers or DEFAULT_TICKERS
    today      = datetime.today().date()
    trade_days = _next_n_trading_days(today, n_days)

    print("=" * 62)
    print(f"  Dashboard Data Generator (daily variation)")
    print(f"  Forecast : {today} -> {trade_days[-1]}  ({n_days} trading days)")
    print(f"  Tickers  : {len(tickers)}")
    print(f"  Output   : {output_path}")
    print("=" * 62)

    all_rows = []
    failed   = []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i:02d}/{len(tickers)}] {ticker} ...", end=" ", flush=True)
        try:
            result           = predict_signal(ticker)
            result["ticker"] = ticker

            if result.get("signal") == "ERROR":
                print(f"Error: {result.get('error', 'unknown')}")
                failed.append(ticker)
                continue

            seed = abs(hash(ticker)) % (2**31)
            rows = _simulate_daily_series(result, trade_days, seed=seed)
            for r in rows:
                r["ticker"] = ticker
            all_rows.extend(rows)

            sig   = result.get("signal", "?")
            price = result.get("current_price", 0) or 0
            print(f"OK  base_signal={sig}  base_price=${price:.2f}  days={len(rows)}")

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            failed.append(ticker)

    if not all_rows:
        print("\nError: No data generated.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df[["date", "ticker", "direction", "signal",
             "current_price", "expected_price", "expected_return_pct"]]
    df.sort_values(["ticker", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*62}")
    print(f"  Done!")
    print(f"  Rows   : {len(df):,}  ({len(tickers)} tickers x {n_days} days)")
    print(f"  Saved  : {output_path}")
    if failed:
        print(f"  Failed : {failed}")
    print(f"{'='*62}\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 30-day dashboard CSV with daily variation")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--days",    type=int,  default=30)
    parser.add_argument("--out",     type=str,  default=OUTPUT_PATH)
    args = parser.parse_args()

    generate(tickers=args.tickers, n_days=args.days, output_path=args.out)