# src/multi_objective_trainer.py
import os
import logging
import pickle
import shutil
import csv
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import threading
# from queue import Empty # <-- حذف شد
import redis
import json
import traceback

# --- [تست حافظه] واردات جدید ---
import psutil
from memory_profiler import profile as memory_profile
import io
# --- [پایان تست حافظه] ---

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

# --- [تغییر کلیدی] وارد کردن پارامترهای فیلترها ---
from config.settings import Evolutionary, Training, Agent
from src.moga_setup import TradingProblem, HyperparameterSampling, HyperparameterCrossover, HyperparameterMutation
from src.evaluation import evaluate_signals
from src.indicator_calculator import IndicatorCalculator
from src.env_wrapper import RustEnvWrapper
from sb3_contrib import RecurrentPPO
from src.callbacks import TradingMetricsCallback
from src.metrics_calculator import calculate_metrics

logger = logging.getLogger(__name__)

# --- [تست حافظه] تابع کمکی ---
def get_process_memory():
    """Returns the non-shared memory usage (RSS) of the current process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)
# --- [پایان تست حافظه] ---

def clean_for_json(obj):
    """به صورت بازگشتی انواع داده NumPy را به انواع استاندارد پایتون تبدیل می‌کند."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [clean_for_json(item) for item in obj.tolist()]
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    return obj

# --- Helper functions remain outside the class to avoid pickle issues ---
def _evaluate_population_static(
    population_hyperparams, full_train_df, gen_num, regime_type): # <-- [اصلاح] صف حذف شد
    
    regime_df = full_train_df[full_train_df['regime'] == regime_type].copy()
    if regime_df.empty:
        logger.warning(f"No data available for regime '{regime_type}'. Skipping evaluation.")
        failed_metrics = calculate_metrics([], [])
        failed_metrics['hyperparams'] = {}
        return [failed_metrics] * len(population_hyperparams), [None] * len(population_hyperparams)

    NUM_FOLDS = 4
    n_samples = len(regime_df)
    fold_size = n_samples // NUM_FOLDS
    indices = np.arange(n_samples)
    all_fold_results = []

    for i in range(NUM_FOLDS - 1):
        train_end_idx = (i + 1) * fold_size
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + fold_size
        train_indices = indices[:train_end_idx]
        val_indices = indices[val_start_idx:val_end_idx]

        if len(val_indices) == 0: continue

        train_df_fold = regime_df.iloc[train_indices]
        val_df_fold = regime_df.iloc[val_indices]

        # --- [اصلاح کلیدی] حذف کد اضافی و تکراری از اینجا ---

        job_results_fold = Parallel(n_jobs=Evolutionary.N_JOBS)(
            delayed(run_single_agent_evaluation)(
                f"g{gen_num}_{regime_type[:4]}_a{j}",
                hyperparams, train_df_fold, [val_df_fold],
                regime_type # <-- [اصلاح] صف حذف شد
            ) for j, hyperparams in enumerate(population_hyperparams)
        )
        all_fold_results.append(job_results_fold)

    # ... (بقیه تابع بدون تغییر) ...
    final_averaged_metrics = []
    num_agents = len(population_hyperparams)

    for agent_idx in range(num_agents):
        agent_metrics_across_folds = [fr[agent_idx][0] for fr in all_fold_results if fr[agent_idx] and fr[agent_idx][0]]
        
        if not agent_metrics_across_folds:
            avg_metrics = calculate_metrics([], [])
        else:
            avg_metrics = {
                'calmar': np.mean([m.get('calmar', 0) for m in agent_metrics_across_folds]),
                'drawdown': np.mean([m.get('drawdown', 0) for m in agent_metrics_across_folds]),
                'profit_factor': np.mean([m.get('profit_factor', 0) for m in agent_metrics_across_folds]),
                'sharpe': np.mean([m.get('sharpe', 0) for m in agent_metrics_across_folds]),
                'win_rate': np.mean([m.get('win_rate', 0) for m in agent_metrics_across_folds]),
                'total_trades': np.sum([m.get('total_trades', 0) for m in agent_metrics_across_folds]),
                'long_trades': np.sum([m.get('long_trades', 0) for m in agent_metrics_across_folds]),
                'short_trades': np.sum([m.get('short_trades', 0) for m in agent_metrics_across_folds]),
            }
        
        avg_metrics['hyperparams'] = population_hyperparams[agent_idx]
        final_averaged_metrics.append(avg_metrics)

    last_fold_model_paths = [res[1] for res in all_fold_results[-1]] if all_fold_results else [None] * num_agents
    return final_averaged_metrics, last_fold_model_paths

# --- [تست حافظه] اضافه کردن دکوراتور ---
@memory_profile(stream=io.StringIO())
def run_single_agent_evaluation(
    agent_id, hyperparams, train_df, validation_chunks,
    regime_type # <-- [اصلاح] صف حذف شد
):
    if not logger.hasHandlers():
        try:
            log_file_path = "trading_pipeline.log"
            # استفاده از حالت 'a' (append) تا فرآیندها روی لاگ هم ننویسند
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8') 
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False # جلوگیری از ارسال لاگ به ریشه
        except Exception:
            pass
        
    indicator_calculator = IndicatorCalculator()
    
    temp_model_dir = "temp_models"
    os.makedirs(temp_model_dir, exist_ok=True)
    temp_model_path = os.path.join(temp_model_dir, f"{agent_id}.zip")
    
    redis_client = None # <-- [اصلاح] تعریف اولیه
    try:
        # --- [اصلاح کلیدی] ساخت کلاینت Redis محلی برای این فرآیند ---
        redis_client = redis.Redis()

        # --- [تست حافظه] لاگ کردن حافظه اولیه ---
        initial_mem_mb = get_process_memory()
        df_mem_mb = train_df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"[{agent_id}] Process Started. Initial RSS: {initial_mem_mb:.2f} MB. Received train_df: {df_mem_mb:.2f} MB.")
        # --- [پایان تست حافظه] ---

        # --- [اصلاح کلیدی] استفاده از Redis publish ---
        redis_client.publish("trading_bot_monitor", json.dumps({'id': agent_id, 'status': 'Initializing'}))

        regime_feature_columns = indicator_calculator.get_feature_columns(regime_type)

        train_env = RustEnvWrapper(
            df=train_df,
            feature_columns=regime_feature_columns,
            agent_hyperparams=hyperparams['rl_params']
        )
        
        # --- [اصلاح کلیدی] پاس دادن کلاینت Redis به Callback ---
        training_callback = TradingMetricsCallback(agent_id=agent_id, redis_client=redis_client)
        
        rl_params = hyperparams['rl_params'].copy()
        for key in ['stop_loss_atr_multiplier', 'take_profit_atr_multiplier']:
            rl_params.pop(key, None)

        model = RecurrentPPO("MultiInputLstmPolicy", train_env, verbose=0, **rl_params)
        model.learn(total_timesteps=Training.TOTAL_TIMESTEPS_PER_AGENT, callback=training_callback)

        # --- [اصلاح کلیدی] استفاده از Redis publish ---
        redis_client.publish("trading_bot_monitor", json.dumps({'id': agent_id, 'status': 'Evaluating'}))

        # --- [اصلاح] حذف "مانکی‌پچ" (monkey-patch) ---
        # eval_indicator_calculator = IndicatorCalculator() # <-- [اصلاح] حذف شد
        # eval_indicator_calculator.get_feature_columns = lambda: indicator_calculator.get_feature_columns(regime_type) # <-- [اصلاح] حذف شد

        all_trade_stats = [
            evaluate_signals(
                model, chunk.copy(), 
                regime_feature_columns, # <-- [اصلاح] پاس دادن مستقیم لیست ستون‌ها
                hyperparams['rl_params'], 
                agent_id, None, redis_client, i + 1, len(validation_chunks) # <-- [اصلاح] پاس دادن کلاینت Redis
            ) for i, chunk in enumerate(validation_chunks)
        ]
        # --- [پایان اصلاح] ---

        pnl_history = [p for res in all_trade_stats for p in res.get('pnl_history', [])]
        trade_history = [t for res in all_trade_stats for t in res.get('trade_history', [])]
        final_metrics = calculate_metrics(pnl_history, trade_history)
        final_metrics['hyperparams'] = hyperparams
        model.save(temp_model_path)

        # --- [تست حافظه] دریافت گزارش و حافظه نهایی ---
        final_mem_mb = get_process_memory()
        mem_report = ""
        if 'stream' in run_single_agent_evaluation.__wrapped__.__globals__:
            stream = run_single_agent_evaluation.__wrapped__.__globals__['stream']
            mem_report = stream.getvalue()
            stream.close() # پاکسازی استریم
        
        logger.info(f"[{agent_id}] Evaluation Done. Final RSS: {final_mem_mb:.2f} MB.")

        update_data = {
            'id': agent_id, 
            'status': 'Done', 
            'final_metrics': final_metrics,
            'memory_profile': mem_report, # <-- [تست حافظه] ارسال گزارش
            'memory_rss_final': final_mem_mb # <-- [تست حافظه] ارسال حافظه نهایی
        }
        # --- [پایان تست حافظه] ---

        # --- [اصلاح کلیدی] پاکسازی دیکشنری قبل از ارسال ---
        cleaned_update_data = clean_for_json(update_data)
        
        # ارسال داده‌های پاکسازی شده
        redis_client.publish("trading_bot_monitor", json.dumps(cleaned_update_data))

        return (final_metrics, temp_model_path)

    # --- [تست حافظه] اضافه کردن مدیریت خطای حافظه ---
    except MemoryError as me:
        error_message = f"MEMORY ERROR. Process ran out of RAM. {traceback.format_exc()}"
        logger.critical(f"--- MEMORY FAILURE: Agent {agent_id} --- \n{error_message}")
        if redis_client:
            failed_update = {
                'id': agent_id,
                'status': 'Failed',
                'error': str(me),
                'traceback': error_message,
            }
            cleaned_failed_update = clean_for_json(failed_update)
            redis_client.publish("trading_bot_monitor", json.dumps(cleaned_failed_update))
    # --- [پایان تست حافظه] ---
    except Exception as e:
        error_message = traceback.format_exc()
        # لاگ کردن خطا در اینجا مشکلی ندارد
        logger.error(f"Error evaluating agent {agent_id} for regime {regime_type}:\n{error_message}")
        
        if redis_client:
            failed_update = {
                'id': agent_id,
                'status': 'Failed',
                'error': str(e),
                'traceback': error_message,
            }
            # --- [اصلاح کلیدی] پاکسازی پیام خطا (برای اطمینان) ---
            cleaned_failed_update = clean_for_json(failed_update)
            redis_client.publish("trading_bot_monitor", json.dumps(cleaned_failed_update))
            
        failed_metrics = calculate_metrics([], [])
        failed_metrics['hyperparams'] = hyperparams
        return (failed_metrics, None)
    finally:
        if redis_client:
            redis_client.close()


class PymooMonitorCallback(Callback):
    # ... (No changes in this class)
    def __init__(self, monitor, trainer, regime_type: str):
        super().__init__()
        self.monitor = monitor
        self.trainer = trainer
        self.regime_type = regime_type
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'monitor' in state: del state['monitor']
        if 'trainer' in state: del state['trainer']
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.monitor = None
        self.trainer = None
    def notify(self, algorithm):
        gen_num = algorithm.n_gen
        results = algorithm.problem.result_from_last_eval
        if results:
            if self.monitor: self.monitor.update_generation_summary(gen_num=gen_num, results=results)
            if self.trainer: self.trainer._log_generation_results_to_csv(gen_num, results, self.regime_type)

class MultiObjectiveTrainer:
    # ... (No changes in this class, providing for completeness)
    def __init__(self, full_train_df, indicator_calculator, base_hyperparams, monitor=None, resume=False, regime_type="TRENDING"):
        self.full_train_df = full_train_df
        self.indicator_calculator = indicator_calculator
        self.base_hyperparams = base_hyperparams
        self.monitor = monitor
        self.resume = resume
        self.regime_type = regime_type
        self.state_file_path = Training.STATE_FILE_PATH.replace('.pkl', f'_{self.regime_type.lower()}.pkl')
        self.results_csv_path = Training.RESULTS_CSV_PATH.replace('.csv', f'_{self.regime_type.lower()}.csv')
        self.models_path = os.path.join(Training.MODEL_SAVE_PATH, f"pareto_front_{self.regime_type.lower()}")
        self.models_from_last_eval = []
        if not self.resume: self._cleanup_files()

    def _log_generation_results_to_csv(self, gen_num: int, results: list, regime_type: str):
        if not results: return
        log_file = self.results_csv_path
        file_exists = os.path.isfile(log_file)
        first_result = next((r for r in results if r), None)
        if not first_result: return
        hyperparam_keys = list(first_result.get('hyperparams', {}).get('rl_params', {}).keys())
        metric_keys = list(calculate_metrics([],[]).keys())
        header = ['generation', 'agent_id', 'regime'] + metric_keys + hyperparam_keys
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists: writer.writeheader()
            for i, res in enumerate(results):
                if not res: continue
                row = {'generation': gen_num, 'agent_id': f"g{gen_num}_{regime_type[:4]}_a{i}", 'regime': regime_type}
                row.update({key: res.get(key) for key in metric_keys})
                hyperparams = res.get('hyperparams', {}).get('rl_params', {})
                row.update({key: hyperparams.get(key) for key in hyperparam_keys})
                writer.writerow(row)
        logger.info(f"Logged {len(results)} results for generation {gen_num} ({regime_type}) to '{log_file}'.")

    def _evaluation_wrapper(self, population_hyperparams): # <-- [اصلاح] صف حذف شد
        """
        A wrapper function that is called once per generation by the pymoo algorithm.
        It now sends the signal to initialize the monitor table for the new generation.
        """
        gen_num = self.monitor.gen_num if self.monitor else 0
        
        redis_client = None
        try:
            # --- [اصلاح کلیدی] ساخت کلاینت Redis محلی برای ارسال پیام شروع ---
            redis_client = redis.Redis()
            
            # --- [اصلاح کلیدی] ارسال پیام برای آماده‌سازی جدول مانیتور ---
            start_msg = {
                'type': 'start_generation',
                'pop_size': len(population_hyperparams),
                'regime_type': self.regime_type
            }
            redis_client.publish("trading_bot_monitor", json.dumps(start_msg))

            # فراخوانی تابع اصلی ارزیابی که فرآیندهای موازی را اجرا می‌کند
            metrics, model_paths = _evaluate_population_static(
                population_hyperparams, self.full_train_df,
                gen_num, self.regime_type # <-- [اصلاح] صف حذف شد
            )
            self.models_from_last_eval = model_paths
            
            # --- [اصلاح] بازگرداندن هر دو مقدار ---
            return metrics, model_paths
        finally:
            # --- [اصلاح کلیدی] بستن کلاینت Redis ---
            if redis_client:
                redis_client.close()
    
    def run(self):
        logger.info(f"--- Starting MOGA Training for REGIME: {self.regime_type} ---")
        
        # --- [اصلاح کلیدی] حذف closure ---
        
        # --- [اصلاح کلیدی] پاس دادن تابع واسط به Problem به جای صف ---
        problem = TradingProblem(
            self.base_hyperparams,
            self._evaluation_wrapper # <-- [اصلاح] تابع wrapper مستقیماً پاس داده شد
        )
        
        algorithm = None
        if self.resume and os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'rb') as f: algorithm = pickle.load(f)
                logger.info(f"Resuming {self.regime_type} training from Generation {algorithm.n_gen + 1}.")
                algorithm.problem = problem
                if self.monitor: self.monitor.gen_num = algorithm.n_gen + 1
            except Exception as e:
                logger.error(f"Could not load state file for {self.regime_type}. Starting fresh. Error: {e}")
                self.resume = False
        
        if algorithm is None:
            algorithm = NSGA2(
                pop_size=Evolutionary.POPULATION_SIZE,
                sampling=HyperparameterSampling(self.base_hyperparams),
                crossover=HyperparameterCrossover(prob=Evolutionary.CROSSOVER_PROB),
                mutation=HyperparameterMutation(self.base_hyperparams),
                eliminate_duplicates=True)

        termination = get_termination("n_gen", Evolutionary.NUM_GENERATIONS)
        callback = PymooMonitorCallback(self.monitor, self, self.regime_type)

        res = minimize(
            problem, algorithm, termination, seed=1,
            callback=callback, verbose=False, copy_algorithm=False
        )
        
        logger.info(f"Saving final algorithm state for {self.regime_type} to {self.state_file_path}")
        with open(self.state_file_path, 'wb') as f: pickle.dump(res.algorithm, f)

        logger.info(f"Optimization finished for {self.regime_type}.")
        pareto_front = self._save_pareto_front(res, problem)
        
        if hasattr(problem, 'result_from_last_eval') and problem.result_from_last_eval:
             self._log_generation_results_to_csv(res.algorithm.n_gen, problem.result_from_last_eval, self.regime_type)
        
        return pareto_front
    
    # --- [START] بازنویسی کامل تابع _save_pareto_front بر اساس دستورالعمل نهایی ---
    def _save_pareto_front(self, result, problem):
        if result.opt is None or len(result.opt) == 0:
            logger.warning(f"No optimal solutions found for {self.regime_type}.")
            return []

        pareto_solutions = []

        # --- [اصلاح کلیدی نهایی] ---
        # دریافت لیست کامل نتایج، مدل‌ها و هایپرپارامترها از آخرین نسل
        results_from_eval = problem.result_from_last_eval
        models_from_eval = problem.models_from_last_eval

        # دریافت هایپرپارامترهایی که مستقیماً در problem ذخیره کردیم
        hyperparams_from_eval = getattr(problem, 'hyperparams_from_last_eval', None)

        if not results_from_eval or not models_from_eval or not hyperparams_from_eval:
            logger.error(f"Could not find all required evaluation artifacts for {self.regime_type}. Skipping save.")
            return []

        # ایجاد یک دیکشنری جستجو (Lookup Dictionary) بر اساس هایپرپارامترها
        # ما هایپرپارامترها را به یک رشته JSON تبدیل می‌کنیم تا قابل هش (Hashable) باشند
        lookup_map = {}
        for i, h_dict in enumerate(hyperparams_from_eval):
            # اطمینان از مرتب‌سازی کلیدها برای مقایسه یکسان
            try:
                # --- [اصلاح ۱] پاکسازی هایپرپارامترها قبل از سریال‌سازی برای رفع هشدار ---
                cleaned_h_dict = clean_for_json(h_dict)
                h_str = json.dumps(cleaned_h_dict, sort_keys=True)
                lookup_map[h_str] = {
                    'metrics': results_from_eval[i],
                    'model_path': models_from_eval[i],
                    'original_index': i
                }
            except TypeError as e:
                logger.warning(f"Could not hash hyperparams for index {i} due to non-serializable data: {e}. Params: {h_dict}")


        # اکنون در میان راه‌حل‌های بهینه NSGA2 (result.opt) می‌گردیم
        for i, optimal_ind in enumerate(result.opt):
            # optimal_ind.X هایپرپارامترهای این فرد بهینه است
            # ما باید آن‌ها را به فرمت دیکشنری که در ابتدا ساخته بودیم، برگردانیم
            param_list = list(self.base_hyperparams['param_ranges'].keys())
            hyperparams_dict = {'rl_params': {key: optimal_ind.X[j] for j, key in enumerate(param_list)}}

            # --- [اصلاح ۲] پاکسازی هایپرپارامترها قبل از سریال‌سازی برای رفع خطای اصلی ---
            cleaned_hyperparams_dict = clean_for_json(hyperparams_dict)
            
            # تبدیل به همان رشته JSON برای جستجو
            hyperparams_str = json.dumps(cleaned_hyperparams_dict, sort_keys=True)

            # جستجوی مستقیم و مطمئن
            found_data = lookup_map.get(hyperparams_str)

            if found_data:
                metrics = found_data['metrics']
                temp_model_path = found_data['model_path']

                if temp_model_path is None or not os.path.exists(temp_model_path):
                    logger.warning(f"Found matching hyperparams for solution {i} but model path '{temp_model_path}' is missing. Skipping.")
                    continue

                agent_id = f"pareto_sol_{self.regime_type.lower()}_{i}"
                model_path = os.path.join(self.models_path, f"{agent_id}.zip")
                shutil.move(temp_model_path, model_path)

                solution = {
                    'id': agent_id, 
                    'hyperparams': metrics['hyperparams'], 
                    'metrics': metrics,
                    'model_path': model_path, 
                    'objectives': optimal_ind.F.tolist() # اهداف از NSGA2
                }
                pareto_solutions.append(solution)
            else:
                # این حالت نباید رخ دهد اگر منطق درست باشد
                logger.error(f"CRITICAL: Could not find matching hyperparams for optimal solution {i} (Objectives: {optimal_ind.F}). This model will not be saved. Hash key was: {hyperparams_str}")

        logger.info(f"Saved {len(pareto_solutions)} solutions for {self.regime_type} from the final Pareto front.")
        return pareto_solutions
    # --- [END] بازنویسی کامل ---

    def _cleanup_files(self):
        if os.path.exists(self.models_path): shutil.rmtree(self.models_path)
        os.makedirs(self.models_path, exist_ok=True)
        if os.path.exists(self.state_file_path): os.remove(self.state_file_path)
        if os.path.exists(self.results_csv_path): os.remove(self.results_csv_path)