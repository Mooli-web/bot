# src/moga_setup.py
import numpy as np
import json
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
# --- [تغییر کلیدی] وارد کردن پارامترهای فیلترها ---
from config.settings import Evolutionary

# Helper function to get random hyperparameter values
def get_random_value(param_info):
    p_type = param_info['type']
    if p_type == 'choice':
        val = np.random.choice(param_info['values'])
        
        # --- [اصلاح کلیدی] تبدیل صریح نوع داده NumPy به پایتون ---
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        return val # بازگرداندن مقدار در صورتی که از قبل استاندارد باشد

    p_range = param_info.get('range')
    use_log_scale = param_info.get('log', False)
    if use_log_scale:
        val = 10**np.random.uniform(np.log10(p_range[0]), np.log10(p_range[1]))
    else:
        val = np.random.uniform(p_range[0], p_range[1])
        
    if p_type == 'int':
        return int(round(val)) # این بخش از قبل درست بود
        
    # --- [اصلاح کلیدی] تبدیل صریح float ها به نوع استاندارد پایتون ---
    return float(round(val, 6))

# Pymoo Custom Sampling Operator
class HyperparameterSampling(Sampling):
    def __init__(self, base_hyperparams, initial_population=None):
        super().__init__()
        self.base_hyperparams = base_hyperparams
        self.param_list = list(self.base_hyperparams['param_ranges'].keys())
        self.initial_population = initial_population if initial_population is not None else []
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), None, dtype=object)
        num_initial = len(self.initial_population)
        if num_initial > 0:
            for i in range(min(num_initial, n_samples)):
                hyperparams = self.initial_population[i]['rl_params']
                for j, key in enumerate(self.param_list):
                    X[i, j] = hyperparams.get(key)
        for i in range(num_initial, n_samples):
            for j, key in enumerate(self.param_list):
                param_info = self.base_hyperparams['param_ranges'][key]
                X[i, j] = get_random_value(param_info)
        return X

# Pymoo Custom Crossover Operator
class HyperparameterCrossover(Crossover):
    def __init__(self, prob=0.9):
        super().__init__(2, 2, prob=prob)
    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            m = np.random.rand(n_var) < 0.5
            Y[0, k, m] = p1[m]
            Y[0, k, ~m] = p2[~m]
            Y[1, k, m] = p2[m]
            Y[1, k, ~m] = p1[~m]
        return Y

# Pymoo Custom Mutation Operator
class HyperparameterMutation(Mutation):
    def __init__(self, base_hyperparams):
        super().__init__()
        self.base_hyperparams = base_hyperparams
        self.param_list = list(self.base_hyperparams['param_ranges'].keys())
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            for j in range(problem.n_var):
                if np.random.rand() < 0.1:
                    param_key = self.param_list[j]
                    param_info = self.base_hyperparams['param_ranges'][param_key]
                    X[i, j] = get_random_value(param_info)
        return X

# Pymoo Problem Definition
class TradingProblem(Problem):
    def __init__(self, base_hyperparams, evaluate_population_func): # <-- [اصلاح] صف حذف شد
        self.param_list = list(base_hyperparams['param_ranges'].keys())
        self.evaluate_population_func = evaluate_population_func
        # self.update_queue = update_queue # <-- [اصلاح] حذف شد
        
        super().__init__(
            n_var=len(self.param_list),
            n_obj=2,
            n_ieq_constr=0,
        )
        self.result_from_last_eval = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # --- [اصلاح] پاکسازی کامل ارجاعات حذف شده ---
        state.pop('update_queue', None) 
        if 'evaluate_population_func' in state:
            del state['evaluate_population_func']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # --- [اصلاح] پاکسازی کامل ارجاعات حذف شده ---
        self.update_queue = None 
        self.evaluate_population_func = None

    def _evaluate(self, X, out, *args, **kwargs):
        population_hyperparams = []
        for i in range(len(X)):
            hyperparams = {'rl_params': {key: X[i, j] for j, key in enumerate(self.param_list)}}
            population_hyperparams.append(hyperparams)

        # --- [اصلاح کلیدی] فراخوانی تابع واسط ---
        # این تابع یک تاپل شامل نتایج و مسیر مدل‌ها را برمی‌گرداند
        results, model_paths = self.evaluate_population_func(population_hyperparams)
        
        # --- [اصلاح کلیدی] ذخیره نتایج در خود شیء Problem ---
        self.result_from_last_eval = results
        self.models_from_last_eval = model_paths

        objectives = np.zeros((len(X), self.n_obj))
        for i, res in enumerate(results):
            if res:
                calmar_ratio = res.get('calmar', 0.0)
                drawdown = res.get('drawdown', 10000.0)
                total_trades = res.get('total_trades', 0)
                long_trades = res.get('long_trades', 0)
                short_trades = res.get('short_trades', 0)
            else:
                calmar_ratio, drawdown, total_trades, long_trades, short_trades = 0.0, 10000.0, 0, 0, 0

            if total_trades < Evolutionary.MINIMUM_TOTAL_TRADES:
                objectives[i, 0] = 1000.0
                objectives[i, 1] = 50000.0
                continue

            safe_long = max(long_trades, 1)
            safe_short = max(short_trades, 1)
            balance_ratio = min(safe_long, safe_short) / max(safe_long, safe_short)
            
            if balance_ratio < Evolutionary.TRADE_BALANCE_RATIO_THRESHOLD:
                calmar_ratio *= balance_ratio

            objectives[i, 0] = -calmar_ratio if calmar_ratio > 0 else 100.0
            objectives[i, 1] = drawdown if drawdown > 1e-9 else 10000.0
        
        # --- [اصلاح نهایی] ذخیره هایپرپارامترهای این نسل برای مقایسه مطمئن ---
        self.hyperparams_from_last_eval = population_hyperparams

        out["F"] = objectives