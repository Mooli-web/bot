# src/evaluation.py
import sys
import os
import logging
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Dict, Any, Optional, List # <-- [اصلاح] List اضافه شد
# import multiprocessing as mp # <-- حذف شد
import redis
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.env_wrapper import RustEnvWrapper
# from src.indicator_calculator import IndicatorCalculator # <-- [اصلاح] حذف شد

logger = logging.getLogger(__name__)

def evaluate_signals(
    model: BaseAlgorithm,
    eval_data: pd.DataFrame,
    feature_columns: List[str], # <-- [اصلاح] جایگزین شد
    agent_hyperparams: Dict[str, Any],
    agent_id: str = 'N/A',
    start_step: Optional[int] = None,
    redis_client: Optional[redis.Redis] = None, # <-- [اصلاح] جایگزینی صف با Redis
    eval_run_idx: int = 0,
    total_eval_runs: int = 1
) -> Dict[str, Any]:
    if eval_data.empty:
        logger.warning(f"Agent {agent_id}: Evaluation dataframe is empty. Skipping.")
        return {}

    try:
        # [اصلاح] feature_columns اکنون مستقیماً از پارامترهای تابع می‌آید
        env = RustEnvWrapper(
            df=eval_data, 
            feature_columns=feature_columns, # <-- [اصلاح] پاکسازی شد
            agent_hyperparams=agent_hyperparams
        )
    except Exception as e:
        logger.error(f"Agent {agent_id}: Failed to create Rust env for eval. Error: {e}", exc_info=True)
        return {}

    obs, _ = env.reset(start_step=start_step)
    done = False
    lstm_states = None
    step_count = 0
    
    while not done:
        try:
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            if redis_client and step_count % 50 == 0: # <-- [اصلاح] بررسی کلاینت Redis
                progress_pct = (step_count / len(eval_data)) * 100
                update_data = {
                    'id': agent_id,
                    'status': 'Evaluating',
                    'eval_step': f"{eval_run_idx}/{total_eval_runs} ({progress_pct:.0f}%)"
                }
                # --- [اصلاح کلیدی] استفاده از publish به جای put ---
                redis_client.publish("trading_bot_monitor", json.dumps(update_data))

        except Exception as e:
            logger.error(f"Agent {agent_id}: Error during evaluation step. Halting. Error: {e}", exc_info=True)
            break

    return env.get_final_stats()
