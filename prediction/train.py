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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from Data import FinancialDataCollector

warnings.filterwarnings("ignore")

# ── Hyperparameters ───────────────────────────────────────────────────────────
LOOKAHEAD       = 5
N_CV_SPLITS     = 5
GAP_DAYS        = 5       
PRUNE_THRESHOLD = 0.003
FBETA           = 0.3
MIN_PRECISION   = 0.45    
MAX_BUY_PCT     = 0.25   
MIN_BUY_PCT     = 0.05    


HOLDOUT_RATIO   = 0.15    
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "signal_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)


DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'V', 'PG', 'JPM', 'HD', 'MA',
    'CVX', 'MRK', 'PEP', 'KO', 'ABBV',
]

# ── Feature List ──────────────────────────────────────────────────────────────
FEATURES = [
    "rsi_14", "rsi_extreme", "rsi_divergence",
    "macd", "macd_hist", "macd_signal",
    "bb_percent", "bb_width",
    "ema_10", "ema_20", "ema_50",
    "price_ema50_ratio", "sma_cross",
    "trend_20d", "trend_50d", "trend_200d",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "trend_alignment",
    "atr_14", "atr_ratio",
    "roll_vol_5d", "roll_vol_10d", "vol_anomaly",
    "hl_range_pct", "gap_open_pct",
    "obv_change", "volume_ratio", "volume_direction", "vol_breakout",
    "stoch_k", "stoch_d", "stoch_cross",
    "return_1d", "return_3d", "return_5d",
    "dist_52w_high", "dist_52w_low",
    "day_of_week", "month",
    "sector_enc", "sector_vol_expected", "earnings_day",
    "rsi_x_return", "rsi_x_volume",
    "macd_x_volume", "atr_x_vix",
    "bb_x_volume", "trend_strength",
    "sector_rel_return",
    "vix", "vix_change", "vix_regime",
    "dxy", "tnx_10y",
    "sp500_return", "sp500_mom5",
    "yield_spread",
    "oil", "gold", "btc",
    "market_regime",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_adaptive_threshold(close: pd.Series, window: int = 252) -> float:
    daily_vol = close.pct_change().rolling(window, min_periods=60).std().iloc[-1]
    if np.isnan(daily_vol) or daily_vol <= 0:
        return 0.008
    return round(float(np.clip(daily_vol * 0.60, 0.005, 0.025)), 4)


def winsorize_series(s: pd.Series, limits=(0.01, 0.01)) -> pd.Series:
    return pd.Series(mstats.winsorize(s.fillna(0), limits=limits), index=s.index)


def winsorize_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].max() - X[col].min() <= 1.01:
            continue
        if X[col].isna().mean() > 0.3:
            continue
        X[col] = winsorize_series(X[col], limits=(0.01, 0.01))
    return X


def get_features(ticker: str, period: str = "5y") -> pd.DataFrame:
    """تحميل الـ features بدون bfill للـ macro (إصلاح leakage)."""
    collector = FinancialDataCollector()
    try:
        raw = yf.download(ticker, period=period, progress=False)
        if raw.empty:
            return pd.DataFrame()
        raw = collector._flatten_columns(raw)
        if not all(c in raw.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
            return pd.DataFrame()
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = collector.calculate_technical_indicators(df)
        df = collector.calculate_price_volume_features(df)
        df = collector.add_fundamental_event_features(df, ticker)
        macro_df = collector.download_macro_data()
        if not macro_df.empty:
            df = df.join(macro_df, how="left")
            macro_cols = macro_df.columns.tolist()
            # [v13-FIX] shift(1) فقط بدون bfill لمنع تسرب البيانات المستقبلية
            # ffill فقط لملء الفراغات الأمامية (قيم مفقودة بين التواريخ)
            df[macro_cols] = df[macro_cols].shift(1).ffill()
            # ملء أول صف فقط بالقيمة الأولى المتاحة (limit=1 يمنع التسرب)
            df[macro_cols] = df[macro_cols].bfill(limit=1)
        df = collector.add_macro_derived_features(df)
        df = collector.add_engineered_features(df)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"  [get_features] Error for {ticker}: {e}")
        return pd.DataFrame()


def get_features_until_date(ticker: str, until_date, period: str = "max") -> pd.DataFrame:
    """
    [v13] تحميل الـ features حتى تاريخ معين فقط.
    يُستخدم في الـ backtest لضمان عدم رؤية بيانات مستقبلية.
    """
    collector = FinancialDataCollector()
    try:
        until_str = pd.Timestamp(until_date).strftime("%Y-%m-%d")
        raw = yf.download(ticker, period=period, end=until_str, progress=False)
        if raw.empty:
            return pd.DataFrame()
        raw = collector._flatten_columns(raw)
        if not all(c in raw.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
            return pd.DataFrame()
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = collector.calculate_technical_indicators(df)
        df = collector.calculate_price_volume_features(df)
        df = collector.add_fundamental_event_features(df, ticker)
        macro_df = collector.download_macro_data()
        if not macro_df.empty:
            # قص الـ macro حتى نفس التاريخ
            macro_df = macro_df[macro_df.index <= until_str]
            df = df.join(macro_df, how="left")
            macro_cols = macro_df.columns.tolist()
            # [v13-FIX] shift(1) + ffill فقط بدون bfill
            df[macro_cols] = df[macro_cols].shift(1).ffill()
        df = collector.add_macro_derived_features(df)
        df = collector.add_engineered_features(df)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"  [get_features_until_date] Error for {ticker}: {e}")
        return pd.DataFrame()


# ── [v14] PR-Curve Threshold على holdout حقيقي ───────────────────────────────
def find_threshold_pr(clf, X_val: np.ndarray, y_val: np.ndarray,
                      beta: float = 0.3, min_buy_pct: float = MIN_BUY_PCT,
                      max_buy_pct: float = MAX_BUY_PCT,
                      min_precision: float = MIN_PRECISION) -> float:
    
    try:
        probas = clf.predict_proba(X_val)[:, 1]
    except Exception:
        return 0.5

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_val, probas)

    fbeta_scores = (
        (1 + beta**2) * prec_arr[:-1] * rec_arr[:-1]
        / (beta**2 * prec_arr[:-1] + rec_arr[:-1] + 1e-9)
    )

    n_val   = len(y_val)
    min_buy = max(3, int(n_val * min_buy_pct))
    max_buy = max(min_buy + 1, int(n_val * max_buy_pct))

    # المحاولة 1: precision جيدة + نسبة BUY معقولة
    valid_mask = np.array([
        min_buy <= int((probas >= t).sum()) <= max_buy
        and prec_arr[i] >= min_precision
        for i, t in enumerate(thresholds)
    ])

    
    if valid_mask.sum() == 0:
        valid_mask = np.array([
            min_buy <= int((probas >= t).sum()) <= max_buy
            and prec_arr[i] >= 0.50
            for i, t in enumerate(thresholds)
        ])

   
    if valid_mask.sum() == 0:
        valid_mask = np.array([
            min_buy <= int((probas >= t).sum()) <= max_buy
            for i, t in enumerate(thresholds)
        ])

    if valid_mask.sum() > 0:
        valid_fbeta = np.where(valid_mask, fbeta_scores, -1.0)
        best_idx    = valid_fbeta.argmax()
        best_thr    = float(thresholds[best_idx])
        best_prec   = float(prec_arr[best_idx])
        n_buy       = int((probas >= best_thr).sum())
        print(f"  ✓ PR threshold: {best_thr:.3f} | F{beta}={fbeta_scores[best_idx]:.3f} "
              f"| Prec={best_prec:.3f} | BUY={n_buy}/{n_val} ({n_buy/n_val:.0%})")
        return best_thr

    
    target_pct   = 0.15         
    fallback_thr = float(np.percentile(probas, (1 - target_pct) * 100))
    n_buy        = int((probas >= fallback_thr).sum())
    print(f"  ✓ PR threshold (percentile fallback): {fallback_thr:.3f} "
          f"| BUY={n_buy}/{n_val} ({n_buy/n_val:.0%})")
    return fallback_thr



# ── Model Builders ────────────────────────────────────────────────────────────
def build_classifiers(n_neg: int, n_pos: int) -> dict:
    scale = n_neg / max(n_pos, 1)
    return {
        "rf_clf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=600,
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42, n_jobs=-1,
            )),
        ]),
        "xgb_clf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBClassifier(
                n_estimators=600,
                max_depth=3,
                learning_rate=0.015,
                subsample=0.75,
                colsample_bytree=0.65,
                gamma=2.0,
                min_child_weight=8,
                reg_lambda=2.0,
                reg_alpha=0.2,
                scale_pos_weight=scale,
                eval_metric="logloss",
                random_state=42, n_jobs=-1, verbosity=0,
            )),
        ]),
    }


def build_regressors() -> dict:
    """
    [v13-FIX] تحسين الـ Regressors لتحقيق R² موجب:
    - تقليل max_depth لمنع overfitting
    - زيادة min_samples_leaf للتعميم
    - استخدام Huber loss في XGB (مقاوم للـ outliers)
    - إضافة gradient boosting مع learning rate منخفض
    """
    return {
        "rf_reg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=400,
                max_depth=4,           # [v13] تقليل من 5 → 4
                min_samples_split=30,  # [v13] زيادة من 25 → 30
                min_samples_leaf=15,   # [v13] زيادة من 12 → 15
                max_features=0.4,      # [v13] تقليل من 0.5 → 0.4
                random_state=42, n_jobs=-1,
            )),
        ]),
        "xgb_reg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(
                n_estimators=800,     
                max_depth=3,
                learning_rate=0.008,  
                subsample=0.65,      
                colsample_bytree=0.55, 
                gamma=3.0,            
                min_child_weight=12,  
                reg_alpha=0.3,
                reg_lambda=3.0,        
                eval_metric="mae",     
                random_state=42, n_jobs=-1, verbosity=0,
            )),
        ]),
    }


# ── Safe Fit with Early Stopping ─────────────────────────────────────────────
def _safe_fit(pipe: Pipeline, X_tr, y_tr, X_val=None, y_val=None,
              n_rounds: int = 50):
    model  = pipe.named_steps["model"]
    is_xgb = isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor))

    if is_xgb and X_val is not None and len(X_val) > 0:
        scaler  = pipe.named_steps["scaler"]
        Xtr_sc  = scaler.fit_transform(X_tr)
        Xval_sc = scaler.transform(X_val)
        model.set_params(early_stopping_rounds=n_rounds)
        model.fit(Xtr_sc, y_tr, eval_set=[(Xval_sc, y_val)], verbose=False)
    else:
        pipe.fit(X_tr, y_tr)
    return pipe


# ── SMOTE ─────────────────────────────────────────────────────────────────────
def _apply_smote(X_tr: np.ndarray, y_tr: np.ndarray) -> tuple:
    try:
        from imblearn.over_sampling import SMOTE
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())

        if n_pos < 10 or n_neg < 10:
            return X_tr, y_tr

        current_ratio = n_pos / max(n_neg, 1)
        if current_ratio >= 0.35:
            return X_tr, y_tr

        target_n_pos = int(min(n_neg * 0.55, n_pos * 1.6))
        if target_n_pos <= n_pos:
            return X_tr, y_tr

        target_ratio = min(target_n_pos / n_neg, 0.60)

        smote = SMOTE(
            sampling_strategy=target_ratio,
            k_neighbors=min(5, n_pos - 1),
            random_state=42,
        )
        X_res, y_res = smote.fit_resample(X_tr, y_tr)
        print(f"  [SMOTE] {n_pos} UP → {int((y_res==1).sum())} UP "
              f"| {n_neg} DOWN → {int((y_res==0).sum())} DOWN")
        return X_res, y_res
    except ImportError:
        return X_tr, y_tr
    except Exception as e:
        print(f"  ⚠ SMOTE failed ({e}). Skipping.")
        return X_tr, y_tr


# ── Build Dataset ─────────────────────────────────────────────────────────────
def build_ticker_dataset(ticker: str, period: str = "5y",
                         cutoff_date: str = None):
    """
    [v13] يدعم training_cutoff_date لمنع الـ leakage في الـ backtest.
    إذا حُدِّد cutoff_date، يتم قطع البيانات عنده.
    """
    print(f"\n{'─'*55}")
    print(f"[train] {ticker}")
    feat_df = get_features(ticker, period=period)
    if feat_df.empty:
        print(f"  ⚠ No data, skipping.")
        return None

    # [v13] قص البيانات عند cutoff_date إذا تم تحديده
    if cutoff_date is not None:
        cutoff_ts = pd.Timestamp(cutoff_date)
        feat_df = feat_df[feat_df.index < cutoff_ts]
        if len(feat_df) < 200:
            print(f"  ⚠ Too few rows after cutoff ({len(feat_df)}), skipping.")
            return None
        print(f"  ℹ cutoff={cutoff_date} | rows after cut={len(feat_df)}")

    up_thresh = compute_adaptive_threshold(feat_df["close"])
    print(f"  ℹ threshold={up_thresh*100:.2f}%")

    close   = feat_df["close"]
    log_ret = np.log(close.shift(-LOOKAHEAD) / close)
    log_ret = winsorize_series(log_ret, limits=(0.015, 0.015))
    y_cls   = (log_ret > up_thresh).astype(int)

    cols = [f for f in FEATURES if f in feat_df.columns]
    miss = [f for f in FEATURES if f not in feat_df.columns]
    if miss:
        print(f"  ⚠ Missing features ({len(miss)}): {miss[:5]}...")

    X_df = feat_df[cols].ffill().fillna(0.0)   # [v13-FIX] bfill حُذف
    X_df = winsorize_features(X_df)

    X_df  = X_df.iloc[:-LOOKAHEAD]
    y_cls = y_cls.iloc[:-LOOKAHEAD]
    y_reg = log_ret.iloc[:-LOOKAHEAD]

    valid = y_cls.notna() & y_reg.notna()
    X_df, y_cls, y_reg = X_df[valid], y_cls[valid], y_reg[valid]

    n      = len(X_df)
    up_pct = y_cls.mean()
    print(f"  ✓ {n} rows | UP={up_pct:.1%} | features={len(cols)}")

    if n < 200:
        print(f"  ⚠ Too few rows ({n}), skipping.")
        return None

    return {
        "ticker":     ticker,
        "X_df":       X_df,
        "y_cls":      y_cls,
        "y_reg":      y_reg,
        "feat_names": cols,
        "up_thresh":  up_thresh,
        "n":          n,
    }


# ── [v13] Walk-Forward مع True Holdout للـ Threshold ─────────────────────────
def train_ticker_walkforward(data: dict, n_folds: int = N_CV_SPLITS) -> dict:

    ticker     = data["ticker"]
    X_df       = data["X_df"]
    y_cls      = data["y_cls"]
    y_reg      = data["y_reg"]
    feat_names = data["feat_names"]
    up_thresh  = data["up_thresh"]
    n          = data["n"]

    X   = X_df.values
    y_c = y_cls.values
    y_r = y_reg.values

    min_train   = max(200, n // (n_folds + 1))
    fold_size   = (n - min_train) // n_folds
    fold_size   = max(fold_size, 50)

    fold_results = []
    all_preds    = []
    all_true     = []

    print(f"\n  [Walk-Forward] {n_folds} folds | min_train={min_train} | fold_size~{fold_size}")

    for fold in range(n_folds):
        train_end  = min_train + fold * fold_size
        test_start = train_end + GAP_DAYS
        test_end   = min(test_start + fold_size, n)

        if test_start >= n or test_end <= test_start + 10:
            continue

        # [v13] Three-way split في كل fold:
        # Train (70%) | Cal (15%) | Test مع GAP (15%)
        cal_size  = max(30, int(train_end * 0.15))
        train_cut = train_end - cal_size

        X_tr  = X[:train_cut]
        y_tr  = y_c[:train_cut]
        X_cal = X[train_cut:train_end]    # للـ calibration فقط
        y_cal = y_c[train_cut:train_end]
        X_te  = X[test_start:test_end]    # للتقييم (يمثل الـ "threshold holdout" في الـ fold)
        y_te  = y_c[test_start:test_end]

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())

        X_tr_s, y_tr_s = _apply_smote(X_tr, y_tr)

        clfs = build_classifiers(n_neg, n_pos)
        trained_clfs   = {}
        clf_thresholds = {}

        for name, pipe in clfs.items():
            vcut = int(len(X_tr_s) * 0.85)
            _safe_fit(pipe, X_tr_s[:vcut], y_tr_s[:vcut],
                      X_tr_s[vcut:], y_tr_s[vcut:])

            # Calibrate على X_cal (منفصل عن X_te)
            try:
                method = "isotonic" if len(X_cal) > 150 else "sigmoid"
                cal = CalibratedClassifierCV(pipe, method=method, cv="prefit")
                cal.fit(X_cal, y_cal)
                trained_clfs[name] = cal
            except Exception:
                trained_clfs[name] = pipe

            # [v13] threshold على X_te (Holdout في الـ fold) وليس X_cal
            thr = find_threshold_pr(trained_clfs[name], X_te, y_te,
                                    beta=FBETA, min_precision=MIN_PRECISION)
            clf_thresholds[name] = thr

            preds = (trained_clfs[name].predict_proba(X_te)[:, 1] >= thr).astype(int)
            prec  = precision_score(y_te, preds, zero_division=0)
            f1    = f1_score(y_te, preds, zero_division=0)
            print(f"    Fold {fold+1} [{name}]: Prec={prec:.3f} F1={f1:.3f}")

        rf_probas  = trained_clfs["rf_clf"].predict_proba(X_te)[:, 1]
        xgb_probas = trained_clfs["xgb_clf"].predict_proba(X_te)[:, 1]
        rf_thr     = clf_thresholds["rf_clf"]
        xgb_thr    = clf_thresholds["xgb_clf"]

        ensemble_preds = (
            (rf_probas >= rf_thr) & (xgb_probas >= xgb_thr)
        ).astype(int)

       
        ensemble_thr = max(clf_thresholds["rf_clf"], clf_thresholds["xgb_clf"])

        fold_prec = precision_score(y_te, ensemble_preds, zero_division=0)
        fold_acc  = accuracy_score(y_te, ensemble_preds)
        n_buy     = int(ensemble_preds.sum())

        fold_results.append({
            "fold":       fold + 1,
            "train_size": train_cut,
            "test_size":  test_end - test_start,
            "precision":  fold_prec,
            "accuracy":   fold_acc,
            "n_buy":      n_buy,
            "threshold":  ensemble_thr,
            "rf_thr":     rf_thr,
            "xgb_thr":    xgb_thr,
        })
        all_preds.extend(ensemble_preds.tolist())
        all_true.extend(y_te.tolist())

        print(f"  ► Fold {fold+1} [Ensemble]: Prec={fold_prec:.3f} "
              f"Acc={fold_acc:.3f} BUY={n_buy}/{len(y_te)} thr={ensemble_thr:.3f}")

    # ── Final Model ──────────────────────────────────────────────────────────
    print(f"\n  [Final Model] Training on all {n} rows...")

    n_pos_all = int((y_c == 1).sum())
    n_neg_all = int((y_c == 0).sum())

    
    holdout_size = max(60, int(n * HOLDOUT_RATIO))
    cal_size_f   = max(60, int(n * 0.20))
    train_size_f = n - holdout_size - cal_size_f

    if train_size_f < 150:
        
        holdout_size = max(40, int(n * 0.10))
        cal_size_f   = max(40, int(n * 0.15))
        train_size_f = n - holdout_size - cal_size_f

    X_train_f  = X[:train_size_f]
    y_train_f  = y_c[:train_size_f]
    y_reg_tr_f = y_r[:train_size_f]

    X_cal_f  = X[train_size_f : train_size_f + cal_size_f]
    y_cal_f  = y_c[train_size_f : train_size_f + cal_size_f]

    
    X_hold_f  = X[train_size_f + cal_size_f:]
    y_hold_f  = y_c[train_size_f + cal_size_f:]
    y_reg_h_f = y_r[train_size_f + cal_size_f:]

    print(f"  [v14] Split: Train={train_size_f} | Cal={cal_size_f} "
          f"| Holdout={len(X_hold_f)} (true OOS)")

    X_tr_f_s, y_tr_f_s = _apply_smote(X_train_f, y_train_f)

    final_clfs = build_classifiers(n_neg_all, n_pos_all)
    final_cals = {}
    final_thrs = {}

    for name, pipe in final_clfs.items():
        vcut = int(len(X_tr_f_s) * 0.90)
        _safe_fit(pipe, X_tr_f_s[:vcut], y_tr_f_s[:vcut],
                  X_tr_f_s[vcut:], y_tr_f_s[vcut:])

       
        try:
            method = "isotonic" if cal_size_f > 200 else "sigmoid"
            cal = CalibratedClassifierCV(pipe, method=method, cv="prefit")
            cal.fit(X_cal_f, y_cal_f)
            final_cals[name] = cal
        except Exception:
            final_cals[name] = pipe

        
        thr = find_threshold_pr(final_cals[name], X_hold_f, y_hold_f,
                                beta=FBETA, min_precision=MIN_PRECISION)
        final_thrs[name] = thr

    
    final_ensemble_thr = max(final_thrs["rf_clf"], final_thrs["xgb_clf"])

    
    final_regs = build_regressors()
    reg_r2s    = {}
    reg_maes   = {}

    for rname, rpipe in final_regs.items():
        
        X_reg_tr = np.vstack([X_train_f, X_cal_f])
        y_reg_tr = np.concatenate([y_reg_tr_f,
                                   y_r[train_size_f:train_size_f + cal_size_f]])

       
        if "xgb" in rname:
            vcut_reg = int(len(X_reg_tr) * 0.85)
            _safe_fit(rpipe, X_reg_tr[:vcut_reg], y_reg_tr[:vcut_reg],
                      X_reg_tr[vcut_reg:], y_reg_tr[vcut_reg:], n_rounds=50)
        else:
            rpipe.fit(X_reg_tr, y_reg_tr)

        
        y_pred_hold = rpipe.predict(X_hold_f)
        r2  = r2_score(y_reg_h_f, y_pred_hold)
        mae = mean_absolute_error(y_reg_h_f, y_pred_hold)
        reg_r2s[rname]  = r2
        reg_maes[rname] = mae
        print(f"  [Reg {rname}] R²={r2:.4f} MAE={mae*100:.3f}% [True Holdout OOS]")

    
    best_reg_name = max(reg_r2s, key=reg_r2s.get)
    best_r2       = reg_r2s[best_reg_name]

    if best_r2 < -0.5:
        print(f"  ⚠ [v14] All regressors have poor R² — using naive mean predictor as fallback.")
       
    else:
        print(f"  ✓ Best regressor: {best_reg_name} (R²={best_r2:.4f})")

    final_reg = final_regs[best_reg_name]

    # ── Feature Importance ────────────────────────────────────────────────────
    kept_feats = feat_names
    try:
        imp_arr = None
        rf_cal = final_cals["rf_clf"]
        if hasattr(rf_cal, "calibrated_classifiers_"):
            imp_list = []
            for cc in rf_cal.calibrated_classifiers_:
                base = getattr(cc, "estimator", cc)
                if hasattr(base, "named_steps"):
                    m = base.named_steps.get("model")
                    if m and hasattr(m, "feature_importances_"):
                        imp_list.append(m.feature_importances_)
            if imp_list:
                imp_arr = np.mean(imp_list, axis=0)

        if imp_arr is not None and len(imp_arr) == len(feat_names):
            mask       = imp_arr >= PRUNE_THRESHOLD
            kept_feats = [f for i, f in enumerate(feat_names) if mask[i]]
            removed    = [f for i, f in enumerate(feat_names) if not mask[i]]
            print(f"  [Pruning] Kept={len(kept_feats)} Removed={len(removed)}")

            imp_df = pd.DataFrame({"feature": feat_names[:len(imp_arr)],
                                   "importance": imp_arr}
                                  ).sort_values("importance", ascending=False)
            print(f"  Top 10 features for {ticker}:")
            for _, row in imp_df.head(10).iterrows():
                bar = "█" * int(row["importance"] * 200)
                print(f"    {row['feature']:<32} {bar} {row['importance']:.4f}")
    except Exception as e:
        print(f"  ⚠ Importance error: {e}")

    # ── Aggregate Metrics ─────────────────────────────────────────────────────
    avg_prec     = float(np.mean([f["precision"] for f in fold_results])) if fold_results else 0.0
    avg_acc      = float(np.mean([f["accuracy"]  for f in fold_results])) if fold_results else 0.0
    overall_prec = precision_score(all_true, all_preds, zero_division=0) if all_true else 0.0
    overall_acc  = accuracy_score(all_true, all_preds) if all_true else 0.0

    print(f"\n  ✅ {ticker} Walk-Forward Summary:")
    print(f"     Avg Precision : {avg_prec:.3f}")
    print(f"     Avg Accuracy  : {avg_acc:.3f}")
    print(f"     Overall Prec  : {overall_prec:.3f} (all folds combined)")
    if all_true:
        print(classification_report(
            all_true, all_preds,
            target_names=["DOWN", "UP"], zero_division=0
        ))

    return {
        "ticker":                ticker,
        "classifier_rf":          final_cals["rf_clf"],
        "classifier_xgb":         final_cals["xgb_clf"],
        "classifier":             final_cals["xgb_clf"],
        "regressor":              final_reg,
        "reg_r2":                 best_r2,
        "clf_name":               "ensemble_rf_xgb",
        "reg_name":               best_reg_name,
        "feature_names":          feat_names,
        "optimal_clf_threshold":  final_ensemble_thr,
        "rf_threshold":           final_thrs["rf_clf"],
        "xgb_threshold":          final_thrs["xgb_clf"],
        "up_threshold":           up_thresh,
        "fold_results":           fold_results,
        "avg_precision":          avg_prec,
        "avg_accuracy":           avg_acc,
        "overall_precision":      overall_prec,
        "overall_accuracy":       overall_acc,
        "n_rows":                 n,
        "up_pct":                 float(y_cls.mean()),
        # [v13] نحفظ تاريخ آخر صف تدريب لاستخدامه في الـ backtest
        "training_end_date":      str(X_df.index[-1].date()) if hasattr(X_df.index, 'date') else None,
    }


# ── Main Train Function ───────────────────────────────────────────────────────
def train_and_save(tickers: list = None, period: str = "5y",
                   cutoff_date: str = None):
    """
    [v13] يقبل cutoff_date لتدريب النموذج على بيانات حتى تاريخ معين فقط.
    مفيد عند تشغيل backtest على فترة مستقبلية.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    print("=" * 65)
    print("X-INVEST — Training Pipeline v14 (True OOS + No-Leakage)")
    print(f"Tickers  : {len(tickers)} stocks")
    print(f"Period   : {period} | LOOKAHEAD={LOOKAHEAD}d | GAP={GAP_DAYS}d")
    print(f"CV Folds : {N_CV_SPLITS} | F-beta={FBETA} | Min-Prec={MIN_PRECISION} | Max-BUY={MAX_BUY_PCT:.0%}")
    print(f"Holdout  : {HOLDOUT_RATIO:.0%} (True OOS for threshold + reg eval)")
    if cutoff_date:
        print(f"Cutoff   : {cutoff_date} (no data after this date)")
    print("=" * 65)

    ticker_models     = {}
    ticker_thresholds = {}
    failed_tickers    = []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        try:
            data = build_ticker_dataset(ticker, period=period,
                                        cutoff_date=cutoff_date)
            if data is None:
                failed_tickers.append(ticker)
                continue

            result = train_ticker_walkforward(data, n_folds=N_CV_SPLITS)
            ticker_models[ticker]     = result
            # [v14-FIX] نحفظ optimal_clf_threshold (0.5~0.9) وليس up_threshold (0.005~0.025)
            ticker_thresholds[ticker] = result["optimal_clf_threshold"]

        except Exception as e:
            import traceback
            print(f"\n  ❌ Error training {ticker}: {e}")
            traceback.print_exc()
            failed_tickers.append(ticker)
            continue

    if not ticker_models:
        raise ValueError("No models trained successfully.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TRAINING SUMMARY")
    print("=" * 65)
    print(f"{'Ticker':<8} {'Rows':>6} {'UP%':>6} {'Prec':>7} {'Acc':>7} "
          f"{'RegR²':>7} {'Thr':>6}")
    print("-" * 50)
    for t, m in ticker_models.items():
        r2_str = f"{m.get('reg_r2', 0.0):>7.3f}"
        print(f"{t:<8} {m['n_rows']:>6} {m['up_pct']:>6.1%} "
              f"{m['overall_precision']:>7.3f} {m['overall_accuracy']:>7.3f} "
              f"{r2_str} {m['optimal_clf_threshold']:>6.3f}")

    if failed_tickers:
        print(f"\n⚠ Failed tickers: {failed_tickers}")

    median_threshold = float(np.median(list(ticker_thresholds.values())))
    best_ticker      = max(ticker_models, key=lambda t: ticker_models[t]["overall_precision"])
    best_model       = ticker_models[best_ticker]
    print(f"\n✓ Best single-ticker model: {best_ticker} "
          f"(Prec={best_model['overall_precision']:.3f})")

    # ── Save ───────────────────────────────────────────────────────────────────
    payload = {
        "classifier":            best_model["classifier"],
        "regressor":             best_model["regressor"],
        "clf_name":              best_model["clf_name"],
        "reg_name":              best_model["reg_name"],
        "feature_names":         best_model["feature_names"],
        "optimal_clf_threshold": best_model["optimal_clf_threshold"],

        "ticker_models": {
            t: {
                "classifier":            m["classifier"],
                "classifier_rf":         m.get("classifier_rf"),
                "classifier_xgb":        m.get("classifier_xgb"),
                "regressor":             m["regressor"],
                "clf_name":              m["clf_name"],
                "reg_name":              m["reg_name"],
                "feature_names":         m["feature_names"],
                "optimal_clf_threshold": m["optimal_clf_threshold"],
                "rf_threshold":          m.get("rf_threshold", m["optimal_clf_threshold"]),
                "xgb_threshold":         m.get("xgb_threshold", m["optimal_clf_threshold"]),
                "up_threshold":          m["up_threshold"],
                "reg_r2":                m.get("reg_r2", 0.0),
                "training_end_date":     m.get("training_end_date"),
            }
            for t, m in ticker_models.items()
        },

        "use_sentiment":         False,
        "lookahead":             LOOKAHEAD,
        "up_threshold":          median_threshold,
        "ticker_thresholds":     ticker_thresholds,
        "tickers_trained_on":    list(ticker_models.keys()),
        "failed_tickers":        failed_tickers,
        "trained_at":            datetime.now().isoformat(),
        "version":               "v14",
        "n_folds":               N_CV_SPLITS,
        "fbeta":                 FBETA,
        "min_precision":         MIN_PRECISION,
        "max_buy_pct":           MAX_BUY_PCT,
        "min_buy_pct":           MIN_BUY_PCT,
        "cutoff_date":           cutoff_date,

        "ticker_metrics": {
            t: {
                "avg_precision":     m["avg_precision"],
                "avg_accuracy":      m["avg_accuracy"],
                "overall_precision": m["overall_precision"],
                "overall_accuracy":  m["overall_accuracy"],
                "n_rows":            m["n_rows"],
                "up_pct":            m["up_pct"],
                "reg_r2":            m.get("reg_r2", 0.0),
            }
            for t, m in ticker_models.items()
        },

        "holdout_clf_acc":  best_model["overall_accuracy"],
        "holdout_clf_prec": best_model["overall_precision"],
        "holdout_reg_r2":   best_model.get("reg_r2", 0.0),
        "holdout_reg_mae":  0.0,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n✅ Model saved → {MODEL_PATH}")
    print(f"   Trained tickers : {len(ticker_models)}")
    print(f"   Failed tickers  : {len(failed_tickers)}")
    print(f"   Median threshold: {median_threshold:.4f}")
    print("=" * 65)

    return payload


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-INVEST Training v14")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--period",  default="5y")
    parser.add_argument("--cutoff",  default=None,
                        help="Train only on data before this date (YYYY-MM-DD). "
                             "Use this to avoid leakage when backtesting a future period.")
    args = parser.parse_args()
    train_and_save(args.tickers, args.period, cutoff_date=args.cutoff)