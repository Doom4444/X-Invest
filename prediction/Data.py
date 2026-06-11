import os
import shutil
import threading
import schedule
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')


class FinancialDataCollector:
    def __init__(self, start_date='2010-01-01', end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'PG', 'JPM', 'HD', 'MA',
            'CVX', 'MRK', 'PEP', 'KO', 'ABBV',
        ]

        self.company_names = {
            'apple':                'AAPL',
            'microsoft':            'MSFT',
            'google':               'GOOGL',
            'googl':                'GOOGL',
            'alphabet':             'GOOGL',
            'amazon':               'AMZN',
            'nvidia':               'NVDA',
            'meta':                 'META',
            'facebook':             'META',
            'tesla':                'TSLA',
            'berkshire':            'BRK-B',
            'berkshire hathaway':   'BRK-B',
            'berkshire hathawy':    'BRK-B',
            'unitedhealth':         'UNH',
            'united health':        'UNH',
            'johnson':              'JNJ',
            'johnson & johnson':    'JNJ',
            'johnson and johnson':  'JNJ',
            'visa':                 'V',
            'procter':              'PG',
            'procter & gamble':     'PG',
            'procter and gamble':   'PG',
            'jpmorgan':             'JPM',
            'jp morgan':            'JPM',
            'home depot':           'HD',
            'mastercard':           'MA',
            'chevron':              'CVX',
            'merck':                'MRK',
            'pepsi':                'PEP',
            'pepsico':              'PEP',
            'coca cola':            'KO',
            'coca-cola':            'KO',
            'coke':                 'KO',
            'abbvie':               'ABBV',
            'aapl': 'AAPL', 'msft': 'MSFT', 'amzn': 'AMZN',
            'nvda': 'NVDA', 'tsla': 'TSLA', 'brk-b': 'BRK-B', 'unh': 'UNH',
            'jnj': 'JNJ', 'v': 'V', 'pg': 'PG', 'jpm': 'JPM', 'hd': 'HD',
            'ma': 'MA', 'cvx': 'CVX', 'mrk': 'MRK', 'pep': 'PEP', 'ko': 'KO', 'abbv': 'ABBV',
        }

        self.sector_map = {
            'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'NVDA': 'Tech', 'META': 'Tech',
            'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer',
            'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'MRK': 'Healthcare', 'ABBV': 'Healthcare',
            'JPM': 'Financial', 'V': 'Financial', 'MA': 'Financial', 'BRK-B': 'Financial',
            'PG': 'Staples', 'PEP': 'Staples', 'KO': 'Staples', 'CVX': 'Energy',
        }

        self.sector_volatility = {
            'Tech': 0.025, 'Consumer': 0.022, 'Healthcare': 0.015,
            'Financial': 0.018, 'Staples': 0.010, 'Energy': 0.020,
        }

        sectors_sorted = sorted(set(self.sector_map.values()))
        self.sector_enc_map = {s: i for i, s in enumerate(sectors_sorted)}

    def get_sector_volatility(self, ticker: str) -> float:
        sector = self.sector_map.get(ticker, 'Tech')
        return self.sector_volatility.get(sector, 0.020)

    def get_ticker(self, user_input: str) -> str | None:
        return self.company_names.get(user_input.strip().lower(), None)

    def get_company_data(self, user_input: str, combined_df: pd.DataFrame) -> pd.DataFrame | None:
        ticker = self.get_ticker(user_input)
        if ticker is None:
            print(f"'{user_input}' not found. Available companies:")
            print("   " + ", ".join(sorted(self.company_names.keys())))
            return None
        result = combined_df[combined_df['Ticker'] == ticker].copy()
        if result.empty:
            print(f"Ticker '{ticker}' not in dataset.")
            return None
        print(f"'{user_input}' -> {ticker} | {len(result):,} rows")
        return result

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df['Close']
        high   = df['High']
        low    = df['Low']
        volume = df['Volume']

        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI_14']      = 100 - (100 / (1 + rs))
        df['RSI_Extreme'] = ((df['RSI_14'] > 70) | (df['RSI_14'] < 30)).astype(int)

        price_change_5 = close.diff(5)
        rsi_change_5   = df['RSI_14'].diff(5)
        df['RSI_Divergence'] = (
            ((price_change_5 > 0) & (rsi_change_5 < 0)).astype(int) -
            ((price_change_5 < 0) & (rsi_change_5 > 0)).astype(int)
        )

        ema12          = close.ewm(span=12, adjust=False).mean()
        ema26          = close.ewm(span=26, adjust=False).mean()
        df['MACD']        = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

        sma20    = close.rolling(20).mean()
        std20    = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        df['BB_Percent'] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        df['BB_Width']   = (bb_upper - bb_lower) / sma20.replace(0, np.nan)

        df['EMA_10']            = close.ewm(span=10, adjust=False).mean()
        df['EMA_20']            = close.ewm(span=20, adjust=False).mean()
        df['EMA_50']            = close.ewm(span=50, adjust=False).mean()
        df['Price_EMA50_Ratio'] = close / df['EMA_50']
        df['SMA_Cross']         = (close.rolling(20).mean() > close.rolling(50).mean()).astype(float)

        tr             = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        df['ATR_14']   = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        df['ATR_Ratio'] = df['ATR_14'] / close

        raw_obv        = (np.sign(close.diff()).fillna(0) * volume).cumsum()
        df['OBV']        = raw_obv
        df['OBV_Change'] = raw_obv.pct_change(5).fillna(0).clip(-5, 5)

        low14         = low.rolling(14).min()
        high14        = high.rolling(14).max()
        df['Stoch_K'] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        cross         = pd.Series(0, index=close.index, dtype=int)
        cross[(df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1)) & (df['Stoch_K'] > df['Stoch_D'])] = 1
        cross[(df['Stoch_K'].shift(1) >= df['Stoch_D'].shift(1)) & (df['Stoch_K'] < df['Stoch_D'])] = -1
        df['Stoch_Cross'] = cross

        return df

    def calculate_price_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df['Close']
        volume = df['Volume']

        df['Return_1d']  = close.pct_change(1)
        df['Return_3d']  = close.pct_change(3)
        df['Return_5d']  = close.pct_change(5)
        df['Trend_20d']  = close.pct_change(20)
        df['Trend_50d']  = close.pct_change(50)
        df['Trend_200d'] = close.pct_change(200)

        sma20  = close.rolling(20).mean()
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        df['Price_vs_SMA20']  = (close - sma20)  / sma20.replace(0, np.nan)
        df['Price_vs_SMA50']  = (close - sma50)  / sma50.replace(0, np.nan)
        df['Price_vs_SMA200'] = (close - sma200) / sma200.replace(0, np.nan)

        bullish = (sma20 > sma50) & (sma50 > sma200) & (close > sma200)
        bearish = (sma20 < sma50) & (sma50 < sma200) & (close < sma200)
        df['Trend_Alignment'] = np.select([bullish, bearish], [1.0, -1.0], default=0.0)

        vol_avg_20             = volume.rolling(20).mean()
        df['Volume_Ratio']     = volume / vol_avg_20.replace(0, np.nan)
        df['Volume_Change']    = volume.pct_change()
        df['Volume_Direction'] = ((df['Volume_Ratio'] > 1.5).astype(int)) * np.sign(df['Return_1d'])

        df['HL_Range_Pct'] = (df['High'] - df['Low']) / close * 100
        df['Gap_Open_Pct'] = (df['Open'] - close.shift(1)) / close.shift(1) * 100

        log_ret            = np.log(close / close.shift(1))
        df['Roll_Vol_5d']  = log_ret.rolling(5).std()
        df['Roll_Vol_10d'] = log_ret.rolling(10).std()
        df['Vol_Anomaly']  = df['Roll_Vol_5d'] / df['Roll_Vol_10d'].replace(0, np.nan)

        df['High_52w']      = df['High'].rolling(252).max()
        df['Low_52w']       = df['Low'].rolling(252).min()
        df['Dist_52w_High'] = (close - df['High_52w']) / df['High_52w'] * 100
        df['Dist_52w_Low']  = (close - df['Low_52w'])  / df['Low_52w']  * 100
        df.drop(columns=['High_52w', 'Low_52w'], inplace=True)

        return df

    def add_fundamental_event_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df['Day_of_Week']          = df.index.dayofweek
        df['Month']                = df.index.month
        sector                     = self.sector_map.get(ticker, 'Unknown')
        df['Sector']               = sector
        df['Sector_Enc']           = self.sector_enc_map.get(sector, -1)
        df['Sector_Vol_Expected']  = self.get_sector_volatility(ticker)

        df['Earnings_Day']     = 0
        df['Days_to_Earnings'] = np.nan
        try:
            cal = yf.Ticker(ticker).get_earnings_dates(limit=40)
            if cal is not None and not cal.empty:
                earn_dates = pd.to_datetime(cal.index).normalize()
                idx_dates  = pd.to_datetime(df.index).normalize()

                df['Earnings_Day'] = idx_dates.isin(earn_dates).astype(int)

                def _days_until(d):
                    d      = pd.Timestamp(d).normalize()
                    future = earn_dates[earn_dates >= d]
                    return int((future.min() - d).days) if len(future) else np.nan

                df['Days_to_Earnings'] = [_days_until(d) for d in idx_dates]
        except Exception:
            pass
        return df

    def download_macro_data(self) -> pd.DataFrame:
        print("\nDownloading Macro Data (yfinance) ...")

        macro_symbols = {
            'VIX':     '^VIX',
            'DXY':     'DX-Y.NYB',
            'TNX_10Y': '^TNX',
            'SP500':   '^GSPC',
            'Oil':     'CL=F',
            'Gold':    'GC=F',
            'BTC':     'BTC-USD',
        }

        frames = {}
        for name, sym in macro_symbols.items():
            try:
                raw = yf.download(sym, start=self.start_date, end=self.end_date,
                                  progress=False, auto_adjust=True)
                if raw.empty:
                    print(f"   Warning: {name} ({sym}) — no data")
                    continue
                raw = self._flatten_columns(raw)
                col = 'Close' if 'Close' in raw.columns else raw.columns[0]
                frames[name] = raw[[col]].rename(columns={col: name})
            except Exception as e:
                print(f"   Warning: {name} failed: {e}")

        if not frames:
            print("   Error: No macro data downloaded.")
            return pd.DataFrame()

        macro = pd.concat(frames.values(), axis=1)
        macro.index = pd.to_datetime(macro.index)

        if 'SP500' in macro.columns:
            macro['SP500_Return'] = macro['SP500'].pct_change()

        if 'TNX_10Y' in macro.columns:
            macro['Yield_Spread'] = macro['TNX_10Y'] - macro['TNX_10Y'].rolling(60).mean()

        macro = macro.ffill().bfill()
        macro.index.name = 'Date'

        print(f"   Macro ready — shape: {macro.shape}")
        return macro

    @staticmethod
    def _rolling_normalise(s: pd.Series, window: int = 252) -> pd.Series:
        lo = s.rolling(window, min_periods=20).min()
        hi = s.rolling(window, min_periods=20).max()
        return (s - lo) / (hi - lo).replace(0, np.nan)

    def add_macro_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'VIX' in df.columns:
            df['VIX_Change'] = df['VIX'].pct_change(1).clip(-0.5, 0.5)
        else:
            df['VIX_Change'] = 0.0

        if 'SP500' in df.columns:
            if 'SP500_Return' not in df.columns:
                df['SP500_Return'] = df['SP500'].pct_change(1)
            df['SP500_Mom5'] = df['SP500'].pct_change(5)
        else:
            if 'SP500_Return' not in df.columns:
                df['SP500_Return'] = 0.0
            df['SP500_Mom5'] = 0.0

        if 'Yield_Spread' not in df.columns:
            if 'TNX_10Y' in df.columns:
                df['Yield_Spread'] = df['TNX_10Y'] - df['TNX_10Y'].shift(63)
            else:
                df['Yield_Spread'] = 0.0

        return df

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['RSI_x_Return']  = df['RSI_14']    * df['Return_1d']
        df['RSI_x_Volume']  = df['RSI_14']    * df['Volume_Ratio']
        df['MACD_x_Volume'] = df['MACD_Hist'] * df['Volume_Direction']

        if 'VIX' in df.columns:
            df['ATR_x_VIX']  = df['ATR_Ratio'] * df['VIX']
            df['VIX_Regime'] = pd.cut(df['VIX'], bins=[0, 15, 20, 30, 100],
                                      labels=[0, 1, 2, 3]).astype(float)
        else:
            df['ATR_x_VIX']  = np.nan
            df['VIX_Regime'] = np.nan

        df['BB_x_Volume']    = df['BB_Percent'] * df['Volume_Ratio']
        df['Trend_Strength'] = df['Price_EMA50_Ratio'] * df['SMA_Cross']

        rsi_n  = self._rolling_normalise(df['RSI_14'])
        macd_n = self._rolling_normalise(df['MACD_Hist'])
        stch_n = self._rolling_normalise(df['Stoch_K'])
        df['Momentum_Score'] = (rsi_n + macd_n + stch_n) / 3

        df['Vol_Breakout'] = (
            (df['Volume_Ratio'] > 2.0) & (df['BB_Width'] > 0.10)
        ).astype(int)

        df['Sector_Rel_Return'] = np.nan

        if 'SP500' in df.columns:
            sp_sma50  = df['SP500'].rolling(50).mean()
            sp_sma200 = df['SP500'].rolling(200).mean()
            df['Market_Regime'] = np.where(
                df['SP500'] > sp_sma200, 1.0,
                np.where(df['SP500'] < sp_sma50, -1.0, 0.0)
            )
        else:
            df['Market_Regime'] = 0.0

        return df

    @staticmethod
    def add_labels(df: pd.DataFrame,
                   clf_threshold: float = 0.005,
                   signal_threshold: float = 0.02) -> pd.DataFrame:
        next_ret = df['Close'].pct_change(1).shift(-1)
        fwd_5d   = df['Close'].pct_change(5).shift(-5)

        df['Classification_Label'] = (next_ret > clf_threshold).astype(int)
        df['Regression_Label']     = next_ret

        df['Signal'] = np.select(
            [fwd_5d >  signal_threshold,
             fwd_5d < -signal_threshold],
            ['bullish', 'bearish'],
            default='neutral',
        )
        df.loc[fwd_5d.isna(), 'Signal'] = np.nan
        return df

    def collect_all_data(self) -> dict:
        print("=" * 60)
        print("Financial Data Collection  2010 -> 2025")
        print("=" * 60)

        macro_df   = self.download_macro_data()
        all_stocks = {}

        for i, ticker in enumerate(self.tickers, 1):
            print(f"\n[{i:02d}/20]  {ticker} ...", end=" ")
            try:
                raw = yf.download(ticker, start=self.start_date,
                                  end=self.end_date, progress=False)
                if raw.empty:
                    print("Warning: No data — skipped")
                    continue

                raw = self._flatten_columns(raw)
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(c in raw.columns for c in required):
                    print("Warning: Missing columns — skipped")
                    continue

                df = raw[required].copy()
                df.index = pd.to_datetime(df.index)

                df = self.calculate_technical_indicators(df)
                df = self.calculate_price_volume_features(df)
                df = self.add_fundamental_event_features(df, ticker)
                df = self.add_engineered_features(df)
                df = self.add_labels(df)

                df['Ticker'] = ticker

                if not macro_df.empty:
                    df = df.join(macro_df, how='left')
                    macro_cols = macro_df.columns.tolist()
                    df[macro_cols] = df[macro_cols].ffill()

                all_stocks[ticker] = df
                print(f"OK  rows={len(df)}  cols={len(df.columns)}")
                time.sleep(0.3)

            except Exception as e:
                print(f"Error: {e}")
                continue

        print(f"\nDone — {len(all_stocks)}/20 tickers collected.")
        return all_stocks

    def create_combined_dataset(self, all_stocks: dict) -> pd.DataFrame:
        print("\nCombining all tickers ...")

        if not all_stocks:
            raise ValueError("all_stocks is empty — check internet/tickers.")

        valid = {t: df for t, df in all_stocks.items()
                 if df is not None and not df.empty}
        if not valid:
            raise ValueError("All DataFrames are empty.")

        combined = pd.concat(valid.values(), ignore_index=False)
        combined.sort_index(inplace=True)
        combined.index.name = 'Date'

        if 'Return_1d' in combined.columns and 'Sector' in combined.columns:
            sector_avg = (
                combined.groupby([combined.index, 'Sector'])['Return_1d']
                .transform('mean')
            )
            combined['Sector_Rel_Return'] = combined['Return_1d'] - sector_avg

        combined.dropna(subset=['Close', 'RSI_14', 'EMA_50'], inplace=True)

        combined.drop(
            columns=[c for c in ['Open', 'High', 'Low', 'Volume']
                     if c in combined.columns],
            inplace=True,
        )

        print(f"Combined shape    : {combined.shape}")
        print(f"Date range        : {combined.index.min().date()} -> {combined.index.max().date()}")
        print(f"Tickers           : {combined['Ticker'].nunique()}")
        print(f"Features          : {len([c for c in combined.columns if c not in ['Ticker', 'Sector', 'Signal']])}")
        nan_pct = combined.isna().mean().mean()
        print(f"Overall NaN %     : {nan_pct:.1%}")

        return combined

    @staticmethod
    def save_to_csv(combined: pd.DataFrame,
                    filename: str = 'financial_data_2010_2025.csv',
                    lowercase_cols: bool = True) -> None:
        out = combined.copy()
        if lowercase_cols:
            out.columns = [c.lower() for c in out.columns]
        out.to_csv(filename)
        print(f"\nSaved -> {filename}  ({len(out):,} rows)")

    def export_processed_csv(self, output_path="data/processed_features.csv"):
        print("Collecting and processing data...")
        stocks_dict = self.collect_all_data()
        if not stocks_dict:
            print("Error: No data collected.")
            return
        combined = self.create_combined_dataset(stocks_dict)
        self.save_to_csv(combined, output_path)


class DailyUpdater:

    def __init__(self,
                 csv_path: str = 'financial_data_2010_2025.csv',
                 collector: 'FinancialDataCollector | None' = None,
                 indicator_lookback_days: int = 300):
        self.csv_path                = csv_path
        self.collector               = collector or FinancialDataCollector()
        self.indicator_lookback_days = indicator_lookback_days

    def _safe_save(self, df: pd.DataFrame) -> None:
        tmp_path = self.csv_path + '.tmp'
        retries  = 5
        for attempt in range(1, retries + 1):
            try:
                df.to_csv(tmp_path)
                shutil.move(tmp_path, self.csv_path)
                return
            except PermissionError:
                if attempt == retries:
                    raise
                print(f"   Warning: File is open — waiting 10 seconds (attempt {attempt}/{retries}) ...")
                time.sleep(10)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    @staticmethod
    def _is_us_trading_day(date: pd.Timestamp) -> bool:
        if date.weekday() >= 5:
            return False
        from pandas.tseries.holiday import USFederalHolidayCalendar
        holidays = USFederalHolidayCalendar().holidays(
            start=str(date.year), end=str(date.year))
        return date not in holidays

    @staticmethod
    def _last_trading_day() -> pd.Timestamp:
        d = pd.Timestamp(datetime.today().date()) - pd.Timedelta(days=1)
        while not DailyUpdater._is_us_trading_day(d):
            d -= pd.Timedelta(days=1)
        return d

    @staticmethod
    def _next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
        d = date + pd.Timedelta(days=1)
        while not DailyUpdater._is_us_trading_day(d):
            d += pd.Timedelta(days=1)
        return d

    def _get_last_date_in_csv(self) -> 'pd.Timestamp | None':
        if not os.path.exists(self.csv_path):
            return None
        idx = pd.read_csv(self.csv_path, usecols=[0], index_col=0)
        idx.index = pd.to_datetime(idx.index)
        return idx.index.max()

    def _build_ticker_df(self,
                         ticker: str,
                         fetch_start: str,
                         fetch_end: str,
                         macro_df: pd.DataFrame) -> 'pd.DataFrame | None':
        try:
            raw = yf.download(ticker, start=fetch_start, end=fetch_end,
                              progress=False, auto_adjust=True)
        except Exception as e:
            print(f"      Warning: yfinance download error [{ticker}]: {e}")
            return None

        if raw is None or raw.empty:
            print(f"      Warning: [{ticker}] yfinance returned empty data")
            return None

        raw = self.collector._flatten_columns(raw)

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing  = [c for c in required if c not in raw.columns]
        if missing:
            print(f"      Warning: [{ticker}] missing columns: {missing}")
            return None

        df = raw[required].copy()
        df.index = pd.to_datetime(df.index)

        try:
            df = self.collector.calculate_technical_indicators(df)
        except Exception as e:
            print(f"      Warning: [{ticker}] technical_indicators: {e}")
            return None

        if df is None or df.empty:
            return None

        try:
            df = self.collector.calculate_price_volume_features(df)
        except Exception as e:
            print(f"      Warning: [{ticker}] price_volume: {e}")
            return None

        try:
            df = self.collector.add_fundamental_event_features(df, ticker)
        except Exception as e:
            print(f"      Warning: [{ticker}] fundamental_event: {e}")
            return None

        try:
            df = self.collector.add_engineered_features(df)
        except Exception as e:
            print(f"      Warning: [{ticker}] engineered_features: {e}")
            return None

        try:
            df = self.collector.add_labels(df)
        except Exception as e:
            print(f"      Warning: [{ticker}] add_labels: {e}")
            return None

        df['Ticker'] = ticker

        if macro_df is not None and not macro_df.empty:
            df = df.join(macro_df, how='left')
            df[macro_df.columns] = df[macro_df.columns].ffill()

        before = len(df)
        df.dropna(subset=['Close', 'RSI_14', 'EMA_50'], inplace=True)
        after  = len(df)

        if df.empty:
            print(f"      Warning: [{ticker}] all rows dropped by dropna (was {before} rows)")
            return None

        if after < before:
            print(f"      Info: [{ticker}] dropna: {before} -> {after} rows")

        return df

    def update(self) -> None:
        target_date = self._last_trading_day()
        print(f"   Target date: {target_date.date()}")

        if not os.path.exists(self.csv_path):
            print(f"   '{self.csv_path}' not found — running full download ...")
            stocks   = self.collector.collect_all_data()
            combined = self.collector.create_combined_dataset(stocks)
            self.collector.save_to_csv(combined, self.csv_path)
            return

        last_date  = self._get_last_date_in_csv()
        start_date = self._next_trading_day(last_date)

        print(f"   Last date in CSV : {last_date.date()}")
        print(f"   First new day    : {start_date.date()}")

        if start_date > target_date:
            print("   Data is up to date — nothing to do.")
            return

        fetch_start = (last_date - pd.Timedelta(days=self.indicator_lookback_days)).strftime('%Y-%m-%d')
        fetch_end   = target_date.strftime('%Y-%m-%d')

        print(f"   Download range : {fetch_start} -> {fetch_end}")
        print(f"   (lookback {self.indicator_lookback_days} days for indicator accuracy)")

        orig_start, orig_end      = self.collector.start_date, self.collector.end_date
        self.collector.start_date = fetch_start
        self.collector.end_date   = fetch_end
        try:
            macro_df = self.collector.download_macro_data()
        except Exception as e:
            print(f"   Warning: macro download failed ({e}) — continuing without it")
            macro_df = pd.DataFrame()
        finally:
            self.collector.start_date = orig_start
            self.collector.end_date   = orig_end

        print(f"\n   Building features for each ticker ...")
        new_frames = {}
        failed     = []

        for ticker in self.collector.tickers:
            df = self._build_ticker_df(ticker, fetch_start, fetch_end, macro_df)
            if df is not None:
                new_frames[ticker] = df
                print(f"      OK [{ticker}]  {len(df)} rows")
            else:
                failed.append(ticker)

        print(f"\n   Succeeded: {len(new_frames)}/{len(self.collector.tickers)} tickers")
        if failed:
            print(f"   Failed   : {failed}")

        if not new_frames:
            print("   Error: No data — check internet connection and try again.")
            return

        new_combined = pd.concat(new_frames.values(), ignore_index=False)
        new_combined.sort_index(inplace=True)

        if 'Return_1d' in new_combined.columns and 'Sector' in new_combined.columns:
            sector_avg = (
                new_combined.groupby([new_combined.index, 'Sector'])['Return_1d']
                .transform('mean')
            )
            new_combined['Sector_Rel_Return'] = new_combined['Return_1d'] - sector_avg

        new_combined.drop(
            columns=[c for c in ['Open', 'High', 'Low', 'Volume']
                     if c in new_combined.columns],
            inplace=True)

        new_combined.index = pd.to_datetime(new_combined.index)
        truly_new = new_combined[new_combined.index > last_date].copy()

        if truly_new.empty:
            print("   Done — no new rows after last date.")
            return

        print(f"\n   {len(truly_new):,} new rows (after lookback filter)")

        truly_new.columns = [c.lower() for c in truly_new.columns]

        print(f"   Reading '{self.csv_path}' ...")
        old_df         = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        old_df.columns = [c.lower() for c in old_df.columns]

        merged = pd.concat([old_df, truly_new])
        merged = merged[~merged.index.duplicated(keep='last')]
        merged.sort_index(inplace=True)

        self._safe_save(merged)

        added = len(merged) - len(old_df)
        print(f"\n   Saved: {self.csv_path}")
        print(f"   Before: {len(old_df):,} rows -> After: {len(merged):,} rows (+{added:,})")
        print(f"   Range : {merged.index.min().date()} -> {merged.index.max().date()}")

    def run_scheduler(self, interval_minutes: int = 10) -> None:
        import time as _time

        def _job():
            now = datetime.now()
            print(f"\n{'='*55}")
            print(f"Auto update  [{now:%Y-%m-%d %H:%M:%S}]")
            print(f"{'='*55}")
            try:
                self.update()
            except Exception as e:
                print(f"   Error during update: {e}")

        print("Running first update now ...")
        _job()

        schedule.every(interval_minutes).minutes.do(_job)
        print(f"\nScheduler running — every {interval_minutes} minutes")
        print("Press Ctrl+C to stop\n")

        while True:
            schedule.run_pending()
            _time.sleep(30)


class BackgroundUpdater:
    """
    Usage:
        from Data import BackgroundUpdater
        bg = BackgroundUpdater(csv_path='path/to/file.csv')
        bg.start()
    """

    def __init__(self,
                 csv_path: str = 'financial_data_2010_2025.csv',
                 interval_minutes: int = 30):
        SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = csv_path if os.path.isabs(csv_path) else \
                        os.path.join(SCRIPT_DIR, csv_path)
        self.interval  = interval_minutes
        self.collector = FinancialDataCollector()
        self.updater   = DailyUpdater(csv_path=self.csv_path, collector=self.collector)
        self._thread   = None
        self._stop     = threading.Event()

    def _loop(self):
        self._run_once()
        while not self._stop.wait(self.interval * 60):
            self._run_once()

    def _run_once(self):
        now = datetime.now()
        print(f"\n{'='*55}")
        print(f"[BackgroundUpdater] {now:%Y-%m-%d %H:%M:%S}")
        print(f"{'='*55}")
        try:
            self.updater.update()
        except Exception as e:
            print(f"   Error: {e}")

    def start(self):
        if self._thread and self._thread.is_alive():
            print("Warning: BackgroundUpdater is already running")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="DataUpdater")
        self._thread.start()
        print(f"BackgroundUpdater started — updating every {self.interval} minutes")

    def stop(self):
        self._stop.set()
        print("BackgroundUpdater stopped")


if __name__ == '__main__':
    CSV_NAME         = 'financial_data_2010_2025.csv'
    INTERVAL_MINUTES = 30

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(SCRIPT_DIR, CSV_NAME),
        os.path.join(SCRIPT_DIR, '..', CSV_NAME),
        os.path.join(os.getcwd(), CSV_NAME),
    ]
    CSV_FILE = None
    for _c in _candidates:
        _c = os.path.normpath(_c)
        if os.path.exists(_c):
            CSV_FILE = _c
            print(f"Found file at: {CSV_FILE}")
            break
    if CSV_FILE is None:
        CSV_FILE = os.path.normpath(_candidates[0])
        print(f"File not found — will be created at: {CSV_FILE}")

    collector = FinancialDataCollector(start_date='2010-01-01')
    updater   = DailyUpdater(csv_path=CSV_FILE, collector=collector)

    sep = "=" * 55
    print(sep)
    print(f"Auto Updater started — every {INTERVAL_MINUTES} minutes")
    print(f"File: {CSV_FILE}")
    print(sep)

    cycle = 0
    while True:
        cycle += 1
        dash = "-" * 50
        print(f"\n{dash}")
        print(f"Cycle #{cycle}  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print(dash)
        try:
            updater.update()
        except Exception as e:
            print(f"Error: {e}")

        print(f"\nNext update in {INTERVAL_MINUTES} minutes ...\n")
        time.sleep(INTERVAL_MINUTES * 60)