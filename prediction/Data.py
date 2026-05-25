import os
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
            'apple': 'AAPL', 'microsoft': 'MSFT', 'googl': 'GOOGL',
            'alphabet': 'GOOGL', 'amazon': 'AMZN', 'nvidia': 'NVDA', 
            'meta': 'META', 'facebook': 'META', 'tesla': 'TSLA',
            'berkshire': 'BRK-B', 'berkshire hathaway': 'BRK-B',
            'unitedhealth': 'UNH', 'united health': 'UNH',
            'johnson': 'JNJ', 'johnson & johnson': 'JNJ', 'johnson and johnson': 'JNJ',
            'visa': 'V', 'procter': 'PG', 'procter & gamble': 'PG', 'procter and gamble': 'PG',
            'jpmorgan': 'JPM', 'jp morgan': 'JPM', 'home depot': 'HD', 'mastercard': 'MA',
            'chevron': 'CVX', 'merck': 'MRK', 'pepsi': 'PEP', 'pepsico': 'PEP',
            'coca cola': 'KO', 'coca-cola': 'KO', 'coke': 'KO', 'abbvie': 'ABBV',
            'aapl': 'AAPL', 'msft': 'MSFT', 'amzn': 'AMZN',
            'nvda': 'NVDA', 'tsla': 'TSLA', 'brk-b': 'BRK-B', 'unh': 'UNH',
            'jnj': 'JNJ', 'v': 'V', 'pg': 'PG', 'jpm': 'JPM', 'hd': 'HD',
            'ma': 'MA', 'cvx': 'CVX', 'mrk': 'MRK', 'pep': 'PEP', 'ko': 'KO', 'abbv': 'ABBV'
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

    def get_ticker(self, user_input: str) -> str | None:
        return self.company_names.get(user_input.strip().lower(), None)

    def get_company_data(self, user_input: str, combined_df: pd.DataFrame) -> pd.DataFrame | None:
        ticker = self.get_ticker(user_input)
        if ticker is None: return None
        result = combined_df[combined_df['Ticker'] == ticker].copy()
        return result if not result.empty else None

    def get_sector_volatility(self, ticker: str) -> float:
        sector = self.sector_map.get(ticker, 'Tech')
        return self.sector_volatility.get(sector, 0.020)

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI_14'] = 100 - (100 / (1 + rs))

        price_change_5 = close.diff(5)
        rsi_change_5 = df['RSI_14'].diff(5)
        bullish_div = ((price_change_5 < 0) & (rsi_change_5 > 0)).astype(int)
        bearish_div = ((price_change_5 > 0) & (rsi_change_5 < 0)).astype(int)
        df['RSI_Divergence'] = bullish_div - bearish_div

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        df['BB_Percent'] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        df['BB_Width'] = (bb_upper - bb_lower) / sma20.replace(0, np.nan)

        df['EMA_10'] = close.ewm(span=10, adjust=False).mean()
        df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
        df['Price_EMA50_Ratio'] = close / df['EMA_50']
        df['SMA_Cross'] = (close.rolling(20).mean() > close.rolling(50).mean()).astype(float)

        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        df['ATR_14'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        df['ATR_Ratio'] = df['ATR_14'] / close

        raw_obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
        df['OBV_Change'] = raw_obv.pct_change(5).fillna(0).clip(-5, 5)

        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        df['Stoch_K'] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        cross = pd.Series(0, index=close.index, dtype=int)
        cross[(df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1)) & (df['Stoch_K'] > df['Stoch_D'])] = 1
        cross[(df['Stoch_K'].shift(1) >= df['Stoch_D'].shift(1)) & (df['Stoch_K'] < df['Stoch_D'])] = -1
        df['Stoch_Cross'] = cross

        return df

    def calculate_price_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['Close']
        volume = df['Volume']

        df['Return_1d'] = close.pct_change(1)
        df['Return_3d'] = close.pct_change(3)
        df['Return_5d'] = close.pct_change(5)
        df['Trend_20d'] = close.pct_change(20)
        df['Trend_50d'] = close.pct_change(50)
        df['Trend_200d'] = close.pct_change(200)

        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        df['Price_vs_SMA20'] = (close - sma20) / sma20.replace(0, np.nan)
        df['Price_vs_SMA50'] = (close - sma50) / sma50.replace(0, np.nan)
        df['Price_vs_SMA200'] = (close - sma200) / sma200.replace(0, np.nan)

        bullish = (sma20 > sma50) & (sma50 > sma200) & (close > sma200)
        bearish = (sma20 < sma50) & (sma50 < sma200) & (close < sma200)
        df['Trend_Alignment'] = np.select([bullish, bearish], [1.0, -1.0], default=0.0)

        vol_avg_20 = volume.rolling(20).mean()
        df['Volume_Ratio'] = volume / vol_avg_20.replace(0, np.nan)
        df['Volume_Change'] = volume.pct_change()
        df['Volume_Direction'] = ((df['Volume_Ratio'] > 1.5).astype(int)) * np.sign(df['Return_1d'])

        df['HL_Range_Pct'] = (df['High'] - df['Low']) / close * 100
        df['Gap_Open_Pct'] = (df['Open'] - close.shift(1)) / close.shift(1) * 100

        log_ret = np.log(close / close.shift(1))
        df['Roll_Vol_5d'] = log_ret.rolling(5).std()
        df['Roll_Vol_10d'] = log_ret.rolling(10).std()
        df['Vol_Anomaly'] = df['Roll_Vol_5d'] / df['Roll_Vol_10d'].replace(0, np.nan)

        df['High_52w'] = df['High'].rolling(252).max()
        df['Low_52w'] = df['Low'].rolling(252).min()
        df['Dist_52w_High'] = (close - df['High_52w']) / df['High_52w'] * 100
        df['Dist_52w_Low'] = (close - df['Low_52w']) / df['Low_52w'] * 100
        df.drop(columns=['High_52w', 'Low_52w'], inplace=True)

        return df

    def add_fundamental_event_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        sector = self.sector_map.get(ticker, 'Unknown')
        df['Sector'] = sector
        df['Sector_Enc'] = self.sector_enc_map.get(sector, -1)
        df['Sector_Vol_Expected'] = self.get_sector_volatility(ticker)

        df['Earnings_Day'] = 0
        try:
            cal = yf.Ticker(ticker).get_earnings_dates(limit=40)
            if cal is not None and not cal.empty:
                earn_dates = pd.to_datetime(cal.index).normalize()
                idx_dates = pd.to_datetime(df.index).normalize()
                df['Earnings_Day'] = idx_dates.isin(earn_dates).astype(int)
        except Exception: pass
        return df

    def download_macro_data(self) -> pd.DataFrame:
        macro_symbols = {
            'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 'TNX_10Y': '^TNX',
            'SP500': '^GSPC', 'Oil': 'CL=F', 'Gold': 'GC=F', 'BTC': 'BTC-USD', 'TNX_2Y': '^IRX'
        }
        frames = {}
        for name, sym in macro_symbols.items():
            try:
                raw = yf.download(sym, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                if raw.empty: continue
                raw = self._flatten_columns(raw)
                col = 'Close' if 'Close' in raw.columns else raw.columns[0]
                frames[name] = raw[[col]].rename(columns={col: name})
            except Exception: pass

        if not frames: return pd.DataFrame()
        macro = pd.concat(frames.values(), axis=1)
        macro.index = pd.to_datetime(macro.index)
        macro = macro.ffill().bfill()
        macro.index.name = 'Date'
        return macro

    def add_macro_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'VIX' in df.columns: df['VIX_Change'] = df['VIX'].pct_change(1).clip(-0.5, 0.5)
        else: df['VIX_Change'] = 0.0
        if 'SP500' in df.columns:
            df['SP500_Return'] = df['SP500'].pct_change(1)
            df['SP500_Mom5'] = df['SP500'].pct_change(5)
        else:
            df['SP500_Return'] = df['SP500_Mom5'] = 0.0
        if 'TNX_10Y' in df.columns and 'TNX_2Y' in df.columns:
            df['Yield_Spread'] = df['TNX_10Y'] - (df['TNX_2Y'] / 100 * 365 / 360)
        elif 'TNX_10Y' in df.columns:
            df['Yield_Spread'] = df['TNX_10Y'] - df['TNX_10Y'].shift(63)
        else: df['Yield_Spread'] = 0.0
        return df

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['RSI_x_Return'] = df['RSI_14'] * df['Return_1d']
        df['RSI_x_Volume'] = df['RSI_14'] * df['Volume_Ratio']
        df['MACD_x_Volume'] = df['MACD_Hist'] * df['Volume_Direction']

        if 'VIX' in df.columns:
            df['ATR_x_VIX'] = df['ATR_Ratio'] * df['VIX']
            df['VIX_Regime'] = pd.cut(df['VIX'], bins=[0, 15, 20, 30, 100], labels=[0, 1, 2, 3]).astype(float)
        else:
            df['ATR_x_VIX'] = np.nan
            df['VIX_Regime'] = np.nan

        df['BB_x_Volume'] = df['BB_Percent'] * df['Volume_Ratio']
        df['Trend_Strength'] = df['Price_EMA50_Ratio'] * df['SMA_Cross']
        df['Vol_Breakout'] = ((df['Volume_Ratio'] > 2.0) & (df['BB_Width'] > 0.10)).astype(int)
        df['Sector_Rel_Return'] = np.nan
        df['RSI_Extreme'] = ((df['RSI_14'] > 70) | (df['RSI_14'] < 30)).astype(int)

        if 'SP500' in df.columns:
            sp_sma50 = df['SP500'].rolling(50).mean()
            sp_sma200 = df['SP500'].rolling(200).mean()
            df['Market_Regime'] = np.where(df['SP500'] > sp_sma200, 1.0, np.where(df['SP500'] < sp_sma50, -1.0, 0.0))
        else: df['Market_Regime'] = 0.0
        return df

    def collect_all_data(self) -> dict:
        macro_df = self.download_macro_data()
        all_stocks = {}
        for i, ticker in enumerate(self.tickers, 1):
            try:
                raw = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if raw.empty: continue
                raw = self._flatten_columns(raw)
                if not all(c in raw.columns for c in ['Open', 'High', 'Low', 'Close', 'Volume']): continue

                df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.index = pd.to_datetime(df.index)
                df = self.calculate_technical_indicators(df)
                df = self.calculate_price_volume_features(df)
                df = self.add_fundamental_event_features(df, ticker)

                if not macro_df.empty:
                    df = df.join(macro_df, how='left')
                    macro_cols = macro_df.columns.tolist()
                   
                    df[macro_cols] = df[macro_cols].shift(1).ffill().bfill()

                df = self.add_macro_derived_features(df)
                df = self.add_engineered_features(df)
                df['Ticker'] = ticker

                all_stocks[ticker] = df
                time.sleep(0.2)
            except Exception as e:
                print(f"[collect] Error for {ticker}: {e}")
        return all_stocks

    def export_processed_csv(self, output_path="data/processed_features.csv"):
        print("📦 Collecting and processing data...")
        stocks_dict = self.collect_all_data()
        if not stocks_dict:
            print("❌ No data collected.")
            return

        
        df_all = pd.concat(stocks_dict.values())
        
        
        df_all.index.name = 'Date'
        df_all.reset_index(inplace=True)
        
        
        df_all.sort_values(["Ticker", "Date"], inplace=True)
        df_all.reset_index(drop=True, inplace=True)

       
        save_dir = os.path.dirname(output_path)
        if save_dir: 
            os.makedirs(save_dir, exist_ok=True)

        # الحفظ
        df_all.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f" {output_path}")
        print(f"columns num : {len(df_all):,} {len(df_all.columns)}")

if __name__ == "__main__":
    collector = FinancialDataCollector(start_date="2015-01-01")
    collector.export_processed_csv("data/processed_features.csv")