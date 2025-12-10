# src/indicator_calculator.py
import pandas as pd
import numpy as np
import logging
import pandas_ta as ta

from config.settings import Indicators

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    def __init__(self):
        self.rsi_period = Indicators.RSI_PERIOD
        self.adx_period = Indicators.ADX_PERIOD
        self.atr_period = Indicators.ATR_PERIOD
        self.bbands_period = Indicators.BBANDS_PERIOD
        self.donchian_period = Indicators.DONCHIAN_PERIOD
        logger.debug("IndicatorCalculator initialized with market regime logic.")

    def add_regime_filter_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates indicators needed for market regime identification."""
        if 'ADX' not in df.columns:
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)
            if adx_df is not None and not adx_df.empty:
                adx_col_name = f'ADX_{self.adx_period}'
                if adx_col_name in adx_df.columns:
                    df['ADX'] = adx_df[adx_col_name]
                else:
                    df['ADX'] = np.nan
        
        if 'ATR' not in df.columns:
             df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        
        return df

    def identify_regime(self, row: pd.Series) -> str:
        """Identifies the market regime for a single data row."""
        adx = row.get('ADX', 0)
        if pd.isna(adx):
            return "CHAOS"
            
        if adx > 25:
            return "TRENDING"
        elif adx < 20:
            return "RANGING"
        else:
            return "CHAOS"

    def get_trending_features(self, df: pd.DataFrame, daily_data_c: pd.DataFrame) -> pd.DataFrame:
        """Calculates features specifically for trending market strategies."""
        df['donchian_lower'] = ta.donchian(df['high'], df['low'], length=self.donchian_period).iloc[:, 0]
        df['donchian_upper'] = ta.donchian(df['high'], df['low'], length=self.donchian_period).iloc[:, 1]
        
        daily_sma_short = ta.sma(daily_data_c['close'], length=20)
        daily_sma_long = ta.sma(daily_data_c['close'], length=50)
        daily_sma_short.name = "daily_sma_20"
        daily_sma_long.name = "daily_sma_50"

        df = pd.merge(df, daily_sma_short, left_index=True, right_index=True, how='left').ffill()
        df = pd.merge(df, daily_sma_long, left_index=True, right_index=True, how='left').ffill()

        df['dist_from_daily_sma_norm'] = (df['close'] - df['daily_sma_50']) / df['ATR']
        df['daily_sma_slope'] = (df['daily_sma_20'] - df['daily_sma_20'].shift(5)) / df['ATR']
        
        df['body_size_norm'] = (df['close'] - df['open']).abs() / df['ATR']
        df['volatility_6h'] = df['close'].pct_change().rolling(window=6).std() * 100
        return df

    def get_ranging_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates features specifically for mean-reversion (ranging) market strategies."""
        df['RSI'] = ta.rsi(df['close'], length=self.rsi_period)
        
        bbands_df = ta.bbands(df['close'], length=self.bbands_period)
        df['BB_lower'] = bbands_df.iloc[:, 0]
        df['BB_mid'] = bbands_df.iloc[:, 1]
        df['BB_upper'] = bbands_df.iloc[:, 2]
        
        df['BB_width_normalized'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
        df['price_bb_pos'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df['price_bb_pos'] = df['price_bb_pos'].clip(0, 1)

        df['upper_wick_norm'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['ATR']
        df['lower_wick_norm'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['ATR']
        return df

    def add_indicators(self, data: pd.DataFrame, daily_data: pd.DataFrame = None) -> pd.DataFrame:
        """Main method to add all indicators and identify market regimes."""
        if data.empty:
            logger.warning("Input dataframe is empty. No operations performed.")
            return data

        df = data.copy()
        logger.info(f"Starting feature calculation and regime identification for {len(df)} rows...")
        df.columns = [col.lower() for col in df.columns]
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required OHLC columns in the dataframe.")
        
        df.dropna(subset=required_cols, inplace=True)
        if df.empty:
            logger.error("Dataframe became empty after initial cleaning.")
            return df

        # 1. Add base indicators for regime filter and feature engineering
        df = self.add_regime_filter_indicators(df)

        # 2. Add features for both specialist models
        daily_data_c = None
        if daily_data is not None and not daily_data.empty:
            daily_data_c = daily_data.copy()
            daily_data_c.columns = [col.lower() for col in daily_data_c.columns]

        df = self.get_trending_features(df, daily_data_c)
        df = self.get_ranging_features(df)
        
        # 3. Add time features (useful for both)
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek

        # 4. Identify regime for each row
        df['regime'] = df.apply(self.identify_regime, axis=1)
        logger.info(f"Regime identification complete. Value counts:\n{df['regime'].value_counts(normalize=True)}")

        # 5. Cleanup and finalize
        required_env_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'ATR', 'regime']
        feature_cols = self.get_feature_columns('TRENDING') + self.get_feature_columns('RANGING')
        all_feature_cols = sorted(list(set(feature_cols))) # Get unique feature columns

        final_df = df[required_env_cols + all_feature_cols].copy()
        
        initial_rows = len(final_df)
        
        # --- [اصلاح کلیدی] مرحله ۱: حذف NaNs فقط بر اساس اندیکاتورها ---
        # این کار اطمینان می‌دهد که ردیف‌های ناقص از محاسبات حذف شده‌اند.
        final_df.dropna(subset=all_feature_cols, inplace=True)
        rows_dropped_warmup = initial_rows - len(final_df)
        if rows_dropped_warmup > 0:
            logger.info(f"{rows_dropped_warmup} rows dropped due to indicator warm-up.")

        if final_df.empty:
            logger.critical("Dataframe is empty after indicator warm-up. Cannot proceed.")
            return final_df

        # --- [اصلاح کلیدی] مرحله ۲: شیفت دادن سیگنال رژیم برای جلوگیری از Look-ahead Bias ---
        # رژیم در ردیف [t] اکنون رژیم شناسایی شده در [t-1] است.
        final_df['regime'] = final_df['regime'].shift(1)

        # --- [اصلاح کلیدی] مرحله ۳: حذف NaN ایجاد شده توسط شیفت ---
        # این کار اولین ردیف (که اکنون سیگنال رژیم NaN دارد) را حذف می‌کند.
        initial_rows_after_warmup = len(final_df)
        final_df.dropna(subset=['regime'], inplace=True)
        rows_dropped_shift = initial_rows_after_warmup - len(final_df)
        if rows_dropped_shift > 0:
            logger.info(f"{rows_dropped_shift} row(s) dropped due to regime signal shift (Look-ahead fix).")
        
        if final_df.empty:
            logger.critical("Dataframe is empty after all calculations.")
        else:
            logger.info(f"Feature calculation successful. Final data points: {len(final_df)}")
        
        final_df[all_feature_cols] = final_df[all_feature_cols].astype('float64')
        return final_df

    def get_feature_columns(self, regime_type: str) -> list[str]:
        """Returns the list of feature columns for a given market regime."""
        trending_features = [
            'dist_from_daily_sma_norm', 'daily_sma_slope', 'body_size_norm', 'volatility_6h',
            'hour_of_day', 'day_of_week'
        ]
        ranging_features = [
            'RSI', 'BB_width_normalized', 'price_bb_pos', 'upper_wick_norm', 'lower_wick_norm',
            'hour_of_day', 'day_of_week'
        ]
        if regime_type == 'TRENDING':
            return trending_features
        elif regime_type == 'RANGING':
            return ranging_features
        else:
            return []