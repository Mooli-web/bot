# src/env_wrapper.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces 
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging

from trading_env_rust import TradingEnv as RustEnv
from config.settings import Agent

logger = logging.getLogger(__name__)

class RustEnvWrapper(gym.Env):
    # --- [اصلاح کلیدی] تغییر امضای تابع __init__ برای حذف وابستگی به indicator_calculator ---
    def __init__(self, df: pd.DataFrame, feature_columns: List[str], agent_hyperparams: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # --- [اصلاح کلیدی] ستون‌های ویژگی مستقیماً از پارامتر ورودی گرفته می‌شوند ---
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'ATR'] + feature_columns
        for col in required_cols:
            if col not in df.columns: raise ValueError(f"Required column '{col}' not found in dataframe for environment setup.")

        open_np = df['open'].to_numpy(dtype=np.float32)
        high_np = df['high'].to_numpy(dtype=np.float32)
        low_np = df['low'].to_numpy(dtype=np.float32)
        close_np = df['close'].to_numpy(dtype=np.float32)
        volume_np = df['tick_volume'].to_numpy(dtype=np.float32)
        feature_data_np = df[feature_columns].to_numpy(dtype=np.float32)
        unscaled_atr_np = df['ATR'].to_numpy(dtype=np.float32)

        self._rust_env = RustEnv(
            open_py=open_np, high_py=high_np, low_py=low_np, close_py=close_np, volume_py=volume_np,
            feature_data_py=feature_data_np, unscaled_atr_py=unscaled_atr_np,
            agent_hyperparams=agent_hyperparams or {}, price_window_size=Agent.PRICE_WINDOW_SIZE
        )
        self.action_space = self._rust_env.action_space
        self.observation_space = spaces.Dict({
            "candles": spaces.Box(low=-np.inf, high=np.inf, shape=(self._rust_env.price_window_size, 5), dtype=np.float32),
            "features": spaces.Box(low=-np.inf, high=np.inf, shape=(self._rust_env.num_features,), dtype=np.float32),
        })

    @property
    def trade_history(self):
        """Provides direct access to the trade_history list from the underlying Rust environment."""
        return self._rust_env.trade_history
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, start_step: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        return self._rust_env.reset(start_step=start_step)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        return self._rust_env.step(int(action))
        
    def get_final_stats(self) -> Dict[str, Any]:
        return self._rust_env.get_final_stats()

    def get_attr(self, attr_name, indices=None):
        """Required by SB3. Handles requests for attributes from the underlying Rust env."""
        if attr_name == 'current_step':
            return [self._rust_env.current_step]
        return [getattr(self, attr_name)]