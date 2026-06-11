import sys
# Ensure standard streams use UTF-8 on Windows to prevent UnicodeEncodeError
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse
import warnings
import pickle
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ── Path Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
for p in [ROOT_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL_PATH = os.path.join(SCRIPT_DIR, "saved_models", "signal_model.pkl")

try:
    from predict import predict_signal, _load_model
    _PREDICT_AVAILABLE = True
except ImportError:
    _PREDICT_AVAILABLE = False
    print("⚠️  predict.py not found — using fallback signals only")

try:
    from Train import (get_features, get_features_until_date,
                       FEATURES, compute_adaptive_threshold)
    _TRAIN_AVAILABLE = True
except ImportError:
    _TRAIN_AVAILABLE = False

LOOKAHEAD = 5


# ── [v13] Expanding Window Signal Generator ───────────────────────────────────
class ExpandingWindowSignalGenerator:
   

    
    RECALC_EVERY_N_DAYS = 21   

    def __init__(self, payload: dict = None):
        self.payload       = payload
        self._feature_cache = {}   # cache: date → feat_df

    def _get_model_for_ticker(self, ticker: str) -> dict | None:
        if self.payload is None:
            return None

        ticker_models = self.payload.get("ticker_models", {})
        if ticker in ticker_models:
            tm = ticker_models[ticker]
            return {
                "classifier":            tm["classifier"],
                "classifier_rf":         tm.get("classifier_rf", tm["classifier"]),
                "classifier_xgb":        tm.get("classifier_xgb", tm["classifier"]),
                "feature_names":         tm["feature_names"],
                "optimal_clf_threshold": tm["optimal_clf_threshold"],
                "rf_threshold":          tm.get("rf_threshold", tm["optimal_clf_threshold"]),
                "xgb_threshold":         tm.get("xgb_threshold", tm["optimal_clf_threshold"]),
                "reg_r2":                tm.get("reg_r2", 0.0),
                "is_per_ticker":         True,
            }

        return {
            "classifier":            self.payload["classifier"],
            "classifier_rf":         self.payload.get("classifier_rf", self.payload["classifier"]),
            "classifier_xgb":        self.payload.get("classifier_xgb", self.payload["classifier"]),
            "feature_names":         self.payload["feature_names"],
            "optimal_clf_threshold": self.payload.get("optimal_clf_threshold", 0.55),
            "rf_threshold":          self.payload.get("rf_threshold", 0.55),
            "xgb_threshold":         self.payload.get("xgb_threshold", 0.55),
            "reg_r2":                self.payload.get("holdout_reg_r2", 0.0),
            "is_per_ticker":         False,
        }

    def _get_features_for_date(self, ticker: str, as_of_date: pd.Timestamp,
                                feat_names: list) -> np.ndarray | None:
        
        
        cache_key = (
            ticker,
            as_of_date.strftime("%Y-%m") 
        )

        if cache_key not in self._feature_cache:
           
            until_date = (as_of_date + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
            # لكن لا نتجاوز تاريخ اليوم الفعلي
            until_date = min(until_date, as_of_date.strftime("%Y-%m-%d"))

            if _TRAIN_AVAILABLE:
                feat_df = get_features_until_date(ticker, until_date=until_date)
            else:
                feat_df = pd.DataFrame()

            if feat_df.empty:
                self._feature_cache[cache_key] = None
            else:
                feat_df.index = pd.to_datetime(feat_df.index)
                for c in feat_names:
                    if c not in feat_df.columns:
                        feat_df[c] = 0.0
                self._feature_cache[cache_key] = feat_df

        feat_df = self._feature_cache[cache_key]
        if feat_df is None:
            return None

        # أحدث صف قبل أو في as_of_date
        avail = feat_df.index[feat_df.index <= as_of_date]
        if len(avail) == 0:
            return None

        row = feat_df.loc[[avail[-1]], feat_names].ffill().fillna(0.0)
        return row.values

    def generate_signals(self, df: pd.DataFrame,
                         ticker: str) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        signals["signal"]          = "HOLD"
        signals["confidence"]      = 0.5
        signals["expected_return"] = 0.0
        signals["ensemble_type"]   = "none"

        if self.payload is not None and _TRAIN_AVAILABLE:
            try:
                result = self._expanding_model_signals(df, signals, ticker)
                n_buy = (result["signal"] == "BUY").sum()
                if n_buy == 0:
                    print("  ⚠ Model produced 0 BUY signals → falling back to technical.")
                    return self._technical_signals(df, signals)
                return result
            except Exception as e:
                print(f"  ⚠ Model signals failed ({e}), using technical fallback.")
                import traceback; traceback.print_exc()

        return self._technical_signals(df, signals)

    def _expanding_model_signals(self, df: pd.DataFrame,
                                  signals: pd.DataFrame,
                                  ticker: str) -> pd.DataFrame:
        tm = self._get_model_for_ticker(ticker)
        if tm is None:
            raise ValueError("No model available")

        clf_rf    = tm.get("classifier_rf")
        clf_xgb   = tm.get("classifier_xgb")
        clf_main  = tm["classifier"]
        feat_names = tm["feature_names"]
        rf_thr    = float(tm.get("rf_threshold", tm["optimal_clf_threshold"]))
        xgb_thr   = float(tm.get("xgb_threshold", tm["optimal_clf_threshold"]))
        opt_thr   = float(tm["optimal_clf_threshold"])
        reg_r2    = float(tm.get("reg_r2", 0.0))
        is_per    = tm["is_per_ticker"]

        model_type = "per-ticker ✓" if is_per else "shared ~"
        print(f"  🤖 [v14 Expanding Window] {ticker} [{model_type}] | thr={opt_thr:.3f}")

        n_buy_consensus = 0
        n_buy_partial   = 0
        n_errors        = 0
        cache_hits      = 0

        for date in df.index:
            date_ts = pd.Timestamp(date)

            # [v13] الحصول على الـ features بدون look-ahead
            row_X = self._get_features_for_date(ticker, date_ts, feat_names)
            if row_X is None:
                n_errors += 1
                continue

            cache_hits += 1

            try:
                if clf_rf is not None and clf_xgb is not None:
                    rf_proba  = float(clf_rf.predict_proba(row_X)[0][1])
                    xgb_proba = float(clf_xgb.predict_proba(row_X)[0][1])
                    avg_proba = 0.45 * rf_proba + 0.55 * xgb_proba

                    rf_buy  = rf_proba  >= rf_thr
                    xgb_buy = xgb_proba >= xgb_thr
                    ens_buy = avg_proba  >= opt_thr

                    signals.loc[date, "confidence"] = avg_proba
                    signals.loc[date, "expected_return"] = float(
                        avg_proba - (1 - avg_proba)
                    )

                    
                    if ens_buy or (rf_buy or xgb_buy):
                        signals.loc[date, "signal"]        = "BUY"
                        signals.loc[date, "ensemble_type"] = "consensus"
                        n_buy_consensus += 1
                    elif rf_buy or xgb_buy:
                        signals.loc[date, "signal"]        = "HOLD"
                        signals.loc[date, "ensemble_type"] = "partial"
                        n_buy_partial += 1
                    elif avg_proba < (1 - opt_thr):
                        signals.loc[date, "signal"] = "SELL"

                else:
                    proba   = clf_main.predict_proba(row_X)[0]
                    up_prob = float(proba[1])
                    signals.loc[date, "confidence"] = up_prob
                    if up_prob >= opt_thr:
                        signals.loc[date, "signal"] = "BUY"
                        n_buy_consensus += 1
                    elif up_prob < (1 - opt_thr):
                        signals.loc[date, "signal"] = "SELL"

            except Exception:
                n_errors += 1
                continue

        n_buy   = n_buy_consensus + n_buy_partial
        buy_pct = n_buy / max(len(df), 1) * 100
        consensus_pct = n_buy_consensus / max(len(df), 1) * 100
        print(f"  ✓ {n_buy_consensus} Consensus BUY ({consensus_pct:.1f}%) + "
              f"{n_buy_partial} Partial | Total={buy_pct:.1f}% of days "
              f"| Errors={n_errors} CacheSize={len(self._feature_cache)}")
        if consensus_pct > 60:
            print(f"  ⚠️  [v14] Consensus BUY > 60% of days — النموذج يحتاج إعادة تدريب مع --cutoff")
        return signals

    def _technical_signals(self, df: pd.DataFrame,
                            signals: pd.DataFrame) -> pd.DataFrame:
        """Fallback: RSI + MACD + Trend + Volume."""
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        ema12     = close.ewm(span=12, adjust=False).mean()
        ema26     = close.ewm(span=26, adjust=False).mean()
        macd      = ema12 - ema26
        macd_sig  = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_sig

        sma20  = close.rolling(20).mean()
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        mom5   = close.pct_change(5)
        mom20  = close.pct_change(20)

        vol_avg   = vol.rolling(20).mean()
        vol_ratio = vol / vol_avg.replace(0, np.nan)

        for date in df.index:
            try:
                if pd.isna(rsi.loc[date]) or pd.isna(macd_hist.loc[date]):
                    continue

                r   = float(rsi.loc[date])
                mh  = float(macd_hist.loc[date])
                ms  = float(macd_sig.loc[date])
                vr  = float(vol_ratio.loc[date]) if not pd.isna(vol_ratio.loc[date]) else 1.0
                p   = float(close.loc[date])
                s20 = float(sma20.loc[date])  if not pd.isna(sma20.loc[date])  else p
                s50 = float(sma50.loc[date])  if not pd.isna(sma50.loc[date])  else p
                s200= float(sma200.loc[date]) if not pd.isna(sma200.loc[date]) else p
                m5  = float(mom5.loc[date])   if not pd.isna(mom5.loc[date])   else 0.0
                m20 = float(mom20.loc[date])  if not pd.isna(mom20.loc[date])  else 0.0

                buy_score = 0
                if 30 < r < 60:   buy_score += 1
                if r < 40:        buy_score += 1
                if mh > 0:        buy_score += 1
                if mh > ms * 0.1: buy_score += 1
                if p > s20:       buy_score += 1
                if p > s50:       buy_score += 1
                if p > s200:      buy_score += 1
                if m5 > 0:        buy_score += 1
                if m20 > 0:       buy_score += 1
                if vr > 1.2:      buy_score += 1

                sell_score = 0
                if r > 70:        sell_score += 2
                if r > 80:        sell_score += 2
                if mh < 0:        sell_score += 1
                if p < s20:       sell_score += 1
                if p < s50:       sell_score += 1
                if m5 < -0.02:    sell_score += 1
                if m20 < -0.03:   sell_score += 1

                total  = buy_score + sell_score
                conf_b = buy_score / total if total > 0 else 0.5

                if p > s200 and m20 > 0.05 and r < 70:  # Bull market confirmation
                 buy_score += 2

                signals.loc[date, "confidence"]      = conf_b
                signals.loc[date, "expected_return"] = m5

                if buy_score >= 5:
                    signals.loc[date, "signal"] = "BUY"
                elif sell_score >= 5:
                    signals.loc[date, "signal"] = "SELL"

            except Exception:
                continue

        n_buy = (signals["signal"] == "BUY").sum()
        print(f"  [Technical Fallback] BUY={n_buy}/{len(df)}")
        return signals


# ── Backtest Engine ───────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, ticker: str, start: str, end: str,
                 initial_capital: float = 10_000.0,
                 use_model: bool = True,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 max_hold_days: int = 8,
                 cooldown_days: int = 2,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10):
        self.ticker          = ticker
        self.start           = start
        self.end             = end
        self.initial_capital = initial_capital
        self.use_model       = use_model
        self.commission      = commission
        self.slippage        = slippage
        self.max_hold_days   = max_hold_days
        self.cooldown_days   = cooldown_days
        self.stop_loss_pct   = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.capital         = initial_capital
        self.position        = None
        self.trades          = []
        self.daily_equity    = []
        self.metrics         = {}
        self.last_exit_date  = None

    def _load_payload(self):
        if not _PREDICT_AVAILABLE:
            return None
        try:
            return _load_model()
        except Exception:
            return None

    def _validate_no_leakage(self, payload: dict) -> bool:
        """
        [v13] يتحقق من أن النموذج لم يتدرب على بيانات بعد backtest_start.
        يُصدر تحذيراً إذا وُجد احتمال leakage.
        """
        cutoff_date   = payload.get("cutoff_date")
        trained_at    = payload.get("trained_at", "")
        backtest_start = pd.Timestamp(self.start)

        if cutoff_date is None:
            print(f"\n  ⚠ [v14 LEAKAGE WARNING] النموذج تدرب بدون cutoff_date!")
            print(f"     الحل: أعد التدريب مع --cutoff {self.start}")
            print(f"     مثال: python Train.py --cutoff {self.start}")
            print(f"     النتائج الحالية قد تكون متفائلة بسبب data leakage.\n")
            return False

        cutoff_ts = pd.Timestamp(cutoff_date)
        if cutoff_ts > backtest_start:
            print(f"\n  ⚠ [v14 LEAKAGE WARNING] cutoff={cutoff_date} > backtest_start={self.start}!")
            print(f"     النموذج شاف بيانات من فترة الـ backtest أثناء التدريب.")
            print(f"     أعد التدريب مع --cutoff {self.start}\n")
            return False

        print(f"  ✅ [v14] No Leakage: cutoff={cutoff_date} ≤ backtest_start={self.start}")
        return True

    def _bear_market_filter(self, df: pd.DataFrame, date) -> bool:
        """
        [v14-FIX] فلتر Bear Market محسّن:
        - يبلوك الدخول لو السعر تحت SMA200 (downtrend هيكلي)
        - يبلوك لو الـ 60-day return أقل من -12%
        - كلا الشرطين معاً يعطي فلترة أدق
        """
        try:
            loc = df.index.get_loc(date)

            # شرط 1: السعر تحت SMA200
            if loc >= 200:
                sma200 = df["Close"].iloc[max(0, loc-200):loc].mean()
                price  = float(df["Close"].iloc[loc])
                below_sma200 = price < sma200 * 0.98   # هامش 2%
            else:
                below_sma200 = False

            # شرط 2: 60-day return سالب بشكل واضح
            if loc >= 60:
                window = df["Close"].iloc[max(0, loc-60):loc]
                ret60  = (window.iloc[-1] / window.iloc[0]) - 1
                bad_trend_60d = ret60 < -0.10
            else:
                bad_trend_60d = False

            # شرط 3: 20-day momentum سلبي جداً (crash filter)
            if loc >= 20:
                window20 = df["Close"].iloc[max(0, loc-20):loc]
                ret20    = (window20.iloc[-1] / window20.iloc[0]) - 1
                crash    = ret20 < -0.12
            else:
                crash = False

            # بلوك الدخول إذا: (تحت SMA200 + trend سلبي) أو (crash حاد)
            return (below_sma200 and bad_trend_60d) or crash
        except Exception:
            return False

    def _open_position(self, date, price: float, confidence: float,
                       expected_return: float):
        cost   = price * (1 + self.commission + self.slippage)
        shares = self.capital / cost
        self.position = {
            "entry":           date,
            "entry_price":     price,
            "cost_price":      cost,
            "shares":          shares,
            "confidence":      confidence,
            "expected_return": expected_return,
            "hold_days":       0,
        }
        self.capital = 0.0

    def _close_position(self, date, price: float, reason: str = "signal"):
        if self.position is None:
            return
        net_price = price * (1 - self.commission - self.slippage)
        pnl       = (net_price - self.position["cost_price"]) * self.position["shares"]
        pnl_pct   = (net_price / self.position["cost_price"]) - 1

        self.capital = net_price * self.position["shares"]
        self.trades.append({
            "entry":       self.position["entry"],
            "exit":        date,
            "entry_price": self.position["entry_price"],
            "exit_price":  price,
            "shares":      self.position["shares"],
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
            "hold_days":   self.position["hold_days"],
            "reason":      reason,
            "confidence":  self.position["confidence"],
        })
        self.last_exit_date = date
        self.position = None

    def run(self) -> dict:
        print(f"\n{'='*55}")
        print(f"Backtest: {self.ticker} | {self.start} → {self.end}")

        df = yf.download(self.ticker, start=self.start, end=self.end,
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No price data for {self.ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.sort_index(inplace=True)

        payload = self._load_payload() if self.use_model else None

        # [v13] التحقق من عدم الـ leakage
        if payload is not None:
            self._validate_no_leakage(payload)

        # [v13] استخدام ExpandingWindowSignalGenerator بدلاً من القديم
        sig_gen = ExpandingWindowSignalGenerator(payload=payload)
        signals = sig_gen.generate_signals(df, self.ticker)

        bh_start  = float(df["Close"].iloc[0])
        bh_end    = float(df["Close"].iloc[-1])
        bh_return = (bh_end / bh_start) - 1

        for date in df.index:
            price = float(df.loc[date, "Close"])

            equity = (self.capital if self.position is None
                      else price * self.position["shares"])
            self.daily_equity.append({"date": date, "equity": equity})

            if self.position is not None:
                self.position["hold_days"] += 1
                dd = (price / self.position["entry_price"]) - 1
                if dd <= -self.stop_loss_pct:
                    self._close_position(date, price, "stop_loss")
                    continue
                if dd >= self.take_profit_pct:
                    self._close_position(date, price, "take_profit")
                    continue
                if self.position["hold_days"] >= self.max_hold_days:
                    self._close_position(date, price, "max_hold")
                    continue
                sig = str(signals.loc[date, "signal"]) if date in signals.index else "HOLD"
                if sig == "SELL":
                    self._close_position(date, price, "sell_signal")
                continue

            if self.last_exit_date is not None:
                delta = (pd.Timestamp(date) - pd.Timestamp(self.last_exit_date)).days
                if delta < self.cooldown_days:
                    continue

            if self._bear_market_filter(df, date):
                continue

            sig  = str(signals.loc[date, "signal"])   if date in signals.index else "HOLD"
            conf = float(signals.loc[date, "confidence"]) if date in signals.index else 0.5
            er   = float(signals.loc[date, "expected_return"]) if date in signals.index else 0.0

            if sig == "BUY" and self.capital > 0:
                self._open_position(date, price, conf, er)

        if self.position is not None:
            last_price = float(df["Close"].iloc[-1])
            self._close_position(df.index[-1], last_price, "end_of_period")

        self.metrics = self._compute_metrics(df, bh_return)
        self._print_metrics()
        self._plot_results(df)

        return self.metrics

    def _compute_metrics(self, df: pd.DataFrame, bh_return: float) -> dict:
        if not self.trades:
            return {
                "ticker": self.ticker, "final_capital": self.initial_capital,
                "total_trades": 0, "total_return": 0, "ann_return": 0,
                "max_drawdown": 0, "max_dd": 0, "sharpe": 0, "sharpe_active": 0,
                "sortino": 0, "calmar": 0, "win_rate": 0, "profit_factor": 0,
                "precision_buy": 0, "avg_hold_days": 0, "expected_value": 0,
                "bh_return": round(bh_return, 6), "alpha": 0,
            }

        eq_df     = pd.DataFrame(self.daily_equity).set_index("date").sort_index()
        final_eq  = float(eq_df["equity"].iloc[-1])
        total_ret = (final_eq / self.initial_capital) - 1

        days    = (eq_df.index[-1] - eq_df.index[0]).days
        years   = max(days / 365.25, 0.01)
        ann_ret = (1 + total_ret) ** (1 / years) - 1

        dr       = eq_df["equity"].pct_change().dropna()
        rf_daily = 0.05 / 252

        sharpe = ((dr.mean() - rf_daily) / dr.std() * np.sqrt(252)
                  if dr.std() > 0 else 0.0)

        active_dates = set()
        for t in self.trades:
            idx  = eq_df.index
            mask = (idx >= t["entry"]) & (idx <= t["exit"])
            active_dates.update(idx[mask].tolist())
        dr_active     = dr[[d in active_dates for d in dr.index]]
        sharpe_active = ((dr_active.mean() - rf_daily) / dr_active.std() * np.sqrt(252)
                         if len(dr_active) > 1 and dr_active.std() > 0 else sharpe)

        downside = dr[dr < rf_daily]
        sortino  = ((dr.mean() - rf_daily) / downside.std() * np.sqrt(252)
                    if len(downside) > 1 and downside.std() > 0 else 0.0)

        rolling_max = eq_df["equity"].cummax()
        dd          = (eq_df["equity"] - rolling_max) / rolling_max
        max_dd      = float(dd.min())
        calmar      = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

        trades_df    = pd.DataFrame(self.trades)
        wins         = trades_df[trades_df["pnl"] > 0]
        losses       = trades_df[trades_df["pnl"] <= 0]
        win_rate     = len(wins) / len(trades_df)
        gross_profit = wins["pnl"].sum()        if len(wins)   > 0 else 0.0
        gross_loss   = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-9
        pf           = gross_profit / gross_loss

        avg_win      = wins["pnl_pct"].mean()   if len(wins)   > 0 else 0.0
        avg_loss_pct = losses["pnl_pct"].mean() if len(losses) > 0 else 0.0
        ev           = (win_rate * avg_win) + ((1 - win_rate) * avg_loss_pct)
        avg_hold     = trades_df["hold_days"].mean()

        bh_ann = (1 + bh_return) ** (1 / years) - 1
        alpha  = ann_ret - bh_ann

        return {
            "ticker":        self.ticker,
            "final_capital": round(final_eq, 2),
            "total_trades":  len(trades_df),
            "total_return":  round(total_ret, 6),
            "ann_return":    round(ann_ret, 6),
            "max_drawdown":  round(max_dd, 6),
            "max_dd":        round(max_dd, 6),
            "sharpe":        round(sharpe, 4),
            "sharpe_active": round(sharpe_active, 4),
            "sortino":       round(sortino, 4),
            "calmar":        round(calmar, 4),
            "win_rate":      round(win_rate, 4),
            "profit_factor": round(pf, 4),
            "precision_buy": round(win_rate, 4),
            "avg_hold_days": round(avg_hold, 1),
            "expected_value":round(ev * 100, 3),
            "bh_return":     round(bh_return, 6),
            "alpha":         round(alpha, 6),
        }

    def _print_metrics(self):
        m          = self.metrics
        ev_icon    = "✅" if m["expected_value"] > 0 else "❌"
        alpha_icon = "✅" if m.get("alpha", 0) > 0 else "❌"
        print(f"\n📊 Results {m['ticker']}:")
        print(f"  💵 Final Capital   : ${m['final_capital']:>12,.2f}")
        print(f"  📈 Total Return    : {m['total_return']:>10.2%}")
        print(f"  📅 Annualized      : {m['ann_return']:>10.2%}")
        print(f"  📊 Buy & Hold      : {m['bh_return']:>10.2%}")
        print(f"  {alpha_icon} Alpha vs B&H    : {m.get('alpha', 0):>+10.2%}")
        print(f"  📉 Max Drawdown    : {m['max_drawdown']:>10.2%}")
        print(f"  📐 Sharpe Ratio    : {m['sharpe']:>10.2f}  (all days)")
        print(f"  📐 Sharpe (active) : {m.get('sharpe_active', m['sharpe']):>10.2f}  (in-market only)")
        print(f"  📐 Sortino Ratio   : {m.get('sortino', 0):>10.2f}")
        print(f"  📐 Calmar Ratio    : {m.get('calmar', 0):>10.2f}")
        print(f"  🎯 Win Rate        : {m['win_rate']:>10.1%}")
        print(f"  💸 Profit Factor   : {m['profit_factor']:>10.2f}")
        print(f"  {ev_icon} Expected Value  : {m['expected_value']:>+10.3f}%/trade")
        print(f"  ⏱  Avg Hold Days   : {m['avg_hold_days']:>10.1f}d")
        print(f"  🔄 Total Trades    : {m['total_trades']:>10d}")

    def _plot_results(self, df: pd.DataFrame):
        if not self.daily_equity:
            return
        eq_df = pd.DataFrame(self.daily_equity).set_index("date").sort_index()

        initial_shares = self.initial_capital / df["Close"].iloc[0]
        bh_equity      = df["Close"] * initial_shares

        fig, axes = plt.subplots(3, 1, figsize=(13, 10),
                                 gridspec_kw={"height_ratios": [3, 1, 1]},
                                 sharex=True)

        axes[0].plot(eq_df.index, eq_df["equity"],
                     color="#2ca02c", linewidth=2, label="Strategy (v14 Expanding Window)")
        axes[0].plot(bh_equity.index, bh_equity.values,
                     color="#1f77b4", linewidth=1.5, ls="--", alpha=0.7,
                     label="Buy & Hold")
        axes[0].axhline(self.initial_capital, color="gray", ls=":", alpha=0.5,
                        label="Initial Capital")
        axes[0].set_title(
            f"{self.ticker} — Equity Curve (v14 No-Leakage Expanding Window)", fontsize=13)
        axes[0].set_ylabel("Capital ($)")
        axes[0].legend()

        rolling_max = eq_df["equity"].cummax()
        dd          = (eq_df["equity"] - rolling_max) / rolling_max
        axes[1].fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.6)
        axes[1].set_title("Drawdown (%)")
        axes[1].set_ylabel("DD")

        axes[2].plot(df.index, df["Close"], color="#1f77b4",
                     linewidth=1.2, label="Price")
        for t in self.trades:
            color_e = "green" if t.get("reason") not in ("stop_loss",) else "orange"
            axes[2].axvline(t["entry"], color=color_e, alpha=0.4, lw=0.8)
            axes[2].axvline(t["exit"],  color="red",   alpha=0.4, lw=0.8)
        axes[2].set_title(f"{self.ticker} Price + Trade Markers")
        axes[2].set_ylabel("Price ($)")

        plt.tight_layout()
        out = f"{self.ticker}_backtest_v14.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  📊 Chart saved → {out}")


# ── Multi-Ticker Runner ───────────────────────────────────────────────────────
def run_multi(tickers, start, end, capital, use_model=True):
    all_results = []
    for t in tickers:
        try:
            eng = BacktestEngine(t, start, end, capital, use_model=use_model)
            res = eng.run()
            all_results.append(res)
        except Exception as e:
            print(f"\n❌ Error for {t}: {e}")
            all_results.append({
                "ticker": t, "final_capital": capital,
                "total_return": 0, "ann_return": 0, "bh_return": 0,
                "max_drawdown": 0, "sharpe": 0, "sharpe_active": 0,
                "sortino": 0, "calmar": 0,
                "win_rate": 0, "profit_factor": 0,
                "precision_buy": 0, "total_trades": 0,
                "expected_value": 0, "avg_hold_days": 0, "alpha": 0,
            })

    print("\n" + "="*100)
    print("📋 Performance Comparison — All Tickers")
    print("="*100)
    summary = pd.DataFrame(all_results)
    cols = [
        "ticker", "total_return", "ann_return", "bh_return", "alpha",
        "max_drawdown", "sharpe", "sharpe_active", "sortino", "calmar",
        "win_rate", "profit_factor", "expected_value", "total_trades",
    ]
    available = [c for c in cols if c in summary.columns]
    fmt = {
        "total_return":   "{:.2%}".format,
        "ann_return":     "{:.2%}".format,
        "bh_return":      "{:.2%}".format,
        "alpha":          "{:+.2%}".format,
        "max_drawdown":   "{:.2%}".format,
        "win_rate":       "{:.1%}".format,
        "profit_factor":  "{:.2f}".format,
        "sharpe":         "{:.2f}".format,
        "sharpe_active":  "{:.2f}".format,
        "sortino":        "{:.2f}".format,
        "calmar":         "{:.2f}".format,
        "expected_value": "{:+.3f}%".format,
    }
    print(summary[available].to_string(index=False, formatters=fmt))
    print("="*100 + "\n")

    valid = summary[summary["total_trades"] > 0]
    if not valid.empty:
        print("📊 Aggregate Summary:")
        print(f"  Avg Annualized Return : {valid['ann_return'].mean():.2%}")
        print(f"  Avg Sharpe            : {valid['sharpe'].mean():.2f}")
        print(f"  Avg Win Rate          : {valid['win_rate'].mean():.1%}")
        print(f"  Avg Max Drawdown      : {valid['max_drawdown'].mean():.2%}")
        print(f"  Tickers with +Alpha   : {(valid['alpha'] > 0).sum()}/{len(valid)}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────
DEFAULT_BACKTEST_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'V', 'PG', 'JPM', 'HD', 'MA',
    'CVX', 'MRK', 'PEP', 'KO', 'ABBV',
]

def main():
    parser = argparse.ArgumentParser(description="X-INVEST Backtest v13 (No-Leakage)")
    parser.add_argument("--tickers",  nargs="+", default=None)
    parser.add_argument("--start",    default=None)
    parser.add_argument("--end",      default=None)
    parser.add_argument("--capital",  type=float, default=None)
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()

    if not args.tickers:
        print("💡 For simulation using (click Enter):")
        print("   ⚠️  [v14] تأكد أن النموذج تدرب قبل تاريخ البداية!")
        print("   مثال: python Train.py --cutoff 2022-01-01")
        raw = input(f"📈 Tickers [{','.join(DEFAULT_BACKTEST_TICKERS[:5])},...]: ").strip()
        if raw:
            args.tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        else:
            args.tickers = DEFAULT_BACKTEST_TICKERS

        args.start   = input("📅 Start (YYYY-MM-DD) [2023-01-01]: ").strip() or "2023-01-01"
        args.end     = input("📅 End   (YYYY-MM-DD) [2025-12-31]: ").strip() or "2025-12-31"
        cap_raw      = input("💰 Capital ($) [10000]: ").strip()
        args.capital = float(cap_raw) if cap_raw else 10_000.0

    run_multi(
        tickers   = args.tickers,
        start     = args.start   or "2023-01-01",
        end       = args.end     or "2025-12-31",
        capital   = args.capital or 10_000.0,
        use_model = not args.no_model,
    )

if __name__ == "__main__":
    main()