# main.py
import os
import argparse
import pandas as pd
import logging
import json
import joblib
from sb3_contrib import RecurrentPPO
# import multiprocessing as mp # <-- حذف شد
import numpy as np
from rich.logging import RichHandler
from sklearn.preprocessing import StandardScaler    
import threading

from src.indicator_calculator import IndicatorCalculator
from src.multi_objective_trainer import MultiObjectiveTrainer
from src.evaluation import evaluate_signals
from config.settings import DataSplit, Training, Evolutionary
from src.live_monitor import LiveMonitor
from src.metrics_calculator import calculate_metrics
from src.env_wrapper import RustEnvWrapper

def setup_logging():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_path = "trading_pipeline.log"
    
    rich_handler = RichHandler(rich_tracebacks=True, show_path=False)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler, file_handler]
    )
    
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("joblib").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging is set up. All output will be saved to '{log_file_path}'")

logger = logging.getLogger(__name__)

def load_raw_data(file_path: str, timeframe: str) -> pd.DataFrame:
    logger.info(f"--- 1. Loading Raw {timeframe} Data ---")
    if not os.path.exists(file_path):
        logger.critical(f"Data file not found. Path: '{file_path}'")
        raise FileNotFoundError(f"{timeframe} data file is missing.")
    data = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    logger.info(f"Loaded {len(data)} rows of {timeframe} data.")
    return data
    
def run_training(h1_df_raw: pd.DataFrame, daily_df_raw: pd.DataFrame, args: argparse.Namespace):
    logger.info("--- 2. Preparing Data for Training and Final Test ---")
    
    n_samples = len(h1_df_raw)
    test_set_size = int(n_samples * DataSplit.FINAL_TEST_RATIO)
    train_val_df_raw = h1_df_raw.iloc[:-test_set_size]
    final_test_df_raw = h1_df_raw.iloc[-test_set_size:]
    logger.info(f"Final Hold-out raw data separation is confirmed: {len(final_test_df_raw)} rows.")

    indicator_calculator = IndicatorCalculator()
    
    logger.info("Processing full training data with regime identification...")
    train_val_df = indicator_calculator.add_indicators(train_val_df_raw, daily_df_raw)
    
    all_features = sorted(list(set(indicator_calculator.get_feature_columns('TRENDING') + indicator_calculator.get_feature_columns('RANGING'))))
    scaler = StandardScaler()
    train_val_df.loc[:, all_features] = scaler.fit_transform(train_val_df[all_features])
    joblib.dump(scaler, Training.SCALER_PATH)
    logger.info(f"Scaler fitted and saved to '{Training.SCALER_PATH}'.")

    try:
        with open(args.hyperparams_file, 'r', encoding='utf-8') as f:
            base_hyperparams = {'param_ranges': json.load(f).get('rl_params', {})}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.critical(f"Could not load hyperparameters. Error: {e}")
        return

    pareto_fronts = {}
    
    # --- [اصلاح کلیدی] حذف mp.Manager و mp.Queue ---
    with LiveMonitor() as monitor: # <-- [اصلاح] صف حذف شد
        monitor.total_gens = Evolutionary.NUM_GENERATIONS
        
        # --- [اصلاح کلیدی] راه‌اندازی ترد مانیتور به روش صحیح ---
        monitor_thread = threading.Thread(target=monitor.process_updates, daemon=True)
        monitor_thread.start()
        logger.info("Live monitor thread started.")

        for regime in ["TRENDING", "RANGING"]:
            # The first generation will be started inside the trainer now
            trainer = MultiObjectiveTrainer(
                full_train_df=train_val_df,
                indicator_calculator=indicator_calculator,
                base_hyperparams=base_hyperparams,
                monitor=monitor,
                resume=args.resume,
                regime_type=regime
            )
            pareto_fronts[regime] = trainer.run()

        # --- [اصلاح کلیدی] ارسال سیگنال توقف و انتظار برای پایان ترد ---
        logger.info("Training finished. Stopping monitor thread.")
        # update_queue.put("STOP") # <-- [اصلاح] حذف شد
        monitor_thread.join(timeout=5) # ترد به دلیل __exit__ مانیتور متوقف خواهد شد

    run_final_evaluation(final_test_df_raw, daily_df_raw, pareto_fronts, indicator_calculator)

def run_final_evaluation(test_df_raw: pd.DataFrame, daily_df_raw: pd.DataFrame, pareto_fronts, indicator_calculator: IndicatorCalculator):
    logger.info(f"\n{'='*25} FINAL EVALUATION ON HOLD-OUT TEST SET (Regime-Switching) {'='*25}")
    
    if test_df_raw.empty or not pareto_fronts.get("TRENDING") or not pareto_fronts.get("RANGING"):
        logger.warning("Skipping final test: Missing data or specialist models.")
        return
        
    scaler = joblib.load(Training.SCALER_PATH)
    logger.info("Processing hold-out test data for final evaluation...")
    test_df = indicator_calculator.add_indicators(test_df_raw, daily_df_raw)
    all_features = sorted(list(set(indicator_calculator.get_feature_columns('TRENDING') + indicator_calculator.get_feature_columns('RANGING'))))
    test_df.loc[:, all_features] = scaler.transform(test_df[all_features])

    best_trending_sol = max(pareto_fronts["TRENDING"], key=lambda x: x['metrics']['calmar'])
    best_ranging_sol = max(pareto_fronts["RANGING"], key=lambda x: x['metrics']['calmar'])

    trending_model = RecurrentPPO.load(best_trending_sol['model_path'])
    ranging_model = RecurrentPPO.load(best_ranging_sol['model_path'])
    logger.info(f"Loaded best TRENDING model: {best_trending_sol['id']} (Calmar: {best_trending_sol['metrics']['calmar']:.2f})")
    logger.info(f"Loaded best RANGING model: {best_ranging_sol['id']} (Calmar: {best_ranging_sol['metrics']['calmar']:.2f})")

    # --- [اصلاح کلیدی] ایجاد Wrapper با ارسال لیست ستون‌ها به جای شیء calculator ---
    # The environment for the final test needs all possible features in its observation space.
    env = RustEnvWrapper(
        df=test_df,
        feature_columns=all_features,
        agent_hyperparams=best_trending_sol['hyperparams']['rl_params']
    )
    obs, _ = env.reset()
    done = False
    trending_lstm_states, ranging_lstm_states = None, None
    
    logger.info("Starting regime-switching evaluation loop...")
    while not done:
        current_step_index = env.get_attr('current_step')[0]
        # Ensure we don't go out of bounds
        if current_step_index >= len(test_df): break
        current_regime = test_df.iloc[current_step_index]['regime']
        
        action = 0 # Default action: HOLD/FLAT

        if current_regime == "TRENDING":
            trending_features = indicator_calculator.get_feature_columns("TRENDING")
            feature_indices = [all_features.index(f) for f in trending_features]
            obs_trending = {"candles": obs["candles"], "features": obs["features"][feature_indices]}
            action, trending_lstm_states = trending_model.predict(obs_trending, state=trending_lstm_states, deterministic=True)
        
        elif current_regime == "RANGING":
            ranging_features = indicator_calculator.get_feature_columns("RANGING")
            feature_indices = [all_features.index(f) for f in ranging_features]
            obs_ranging = {"candles": obs["candles"], "features": obs["features"][feature_indices]}
            action, ranging_lstm_states = ranging_model.predict(obs_ranging, state=ranging_lstm_states, deterministic=True)
        
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    final_stats = env.get_final_stats()
    if final_stats and final_stats.get('pnl_history'):
        logger.info(f"--- Final Combined Hold-out Test Results ---")
        final_metrics = calculate_metrics(final_stats['pnl_history'], final_stats['trade_history'])
        for key, value in final_metrics.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")
    else:
        logger.warning("Combined evaluation produced no trades or stats on the test set.")
    
    logger.info("\nFinal hold-out evaluation for the combined system is complete.")

def main(args):
    # ... (No changes)
    setup_logging()
    os.makedirs(Training.MODEL_SAVE_PATH, exist_ok=True)
    try:
        data_path = os.path.join('data', 'raw')
        h1_df_raw = load_raw_data(os.path.join(data_path, args.data_file), "H1")
        daily_df_raw = load_raw_data(os.path.join(data_path, args.daily_data_file), "Daily")
        run_training(h1_df_raw, daily_df_raw, args)
    except FileNotFoundError as e: logger.critical(f"Execution stopped: {e}")
    except Exception: logger.critical("An unexpected error occurred.", exc_info=True)

if __name__ == '__main__':
    # [اصلاح] mp.set_start_method دیگر ضروری نیست مگر اینکه joblib به آن نیاز داشته باشد
    # try: mp.set_start_method('spawn', force=True)
    # except RuntimeError: pass
    parser = argparse.ArgumentParser(description="Multi-Objective Evolutionary Trading Bot Pipeline")
    parser.add_argument('--data-file', type=str, default='EURUSD_H1.csv', help='Name of the H1 data file in data/raw/.')
    parser.add_argument('--daily-data-file', type=str, default='EURUSD_D1.csv', help='Name of the Daily data file in data/raw/.')
    parser.add_argument('--hyperparams-file', type=str, default='config/hyperparameters.json', help='The JSON file for hyperparameter ranges.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last saved state (e.g., trending_moga_state.pkl).')
    args = parser.parse_args()
    main(args)
    logging.getLogger(__name__).info("--- TRADING PIPELINE FINISHED ---")