# src/callbacks.py
import logging
# import multiprocessing as mp # <-- حذف شد
import redis
import json
from stable_baselines3.common.callbacks import BaseCallback
from config.settings import Training
from src.metrics_calculator import calculate_metrics

logger = logging.getLogger(__name__)

class TradingMetricsCallback(BaseCallback):
    def __init__(self, agent_id: str, redis_client: redis.Redis, verbose: int = 0): # <-- [اصلاح] جایگزینی صف
        super().__init__(verbose)
        self.agent_id = agent_id
        self.redis_client = redis_client # <-- [اصلاح]
        self.log_freq = 50

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        all_pnl_history = self.training_env.get_attr('trade_history')[0]
        pnl_history = [log.pnl_pips for log in all_pnl_history]

        if not pnl_history:
            return True

        train_metrics = calculate_metrics(pnl_history, all_pnl_history)

        update_data = {
            'id': self.agent_id,
            'status': 'Training',
            'step': f"{self.num_timesteps}/{Training.TOTAL_TIMESTEPS_PER_AGENT}",
            'train_metrics': train_metrics
        }
        
        # --- [اصلاح کلیدی] استفاده از publish و حذف try-except غیرضروری ---
        if self.redis_client:
            try:
                self.redis_client.publish("trading_bot_monitor", json.dumps(update_data))
            except redis.exceptions.RedisError as e:
                # اگر ارسال ناموفق بود، لاگ کن اما ادامه بده
                logger.warning(f"Could not publish training update for {self.agent_id}: {e}")
            except (ConnectionResetError, BrokenPipeError):
                # اگر صف بسته شده باشد، به کار ادامه بده
                pass

        return True