# config/settings.py
import os

class Indicators:
    RSI_PERIOD = 14
    SMA_SHORT = 50
    SMA_LONG = 200
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    BBANDS_PERIOD = 20
    DONCHIAN_PERIOD = 20

class Agent:
    STOP_LOSS_ATR_MULTIPLIER = 3.0
    TAKE_PROFIT_ATR_MULTIPLIER = 2.5
    MAX_TRADE_DURATION = 48
    PRICE_WINDOW_SIZE = 24
    SLIPPAGE_ATR_FRACTION = 0.1
    COMMISSION_PIPS = 0.7
    SWAP_PIPS_PER_DAY = -0.2

class Training:
    MODEL_SAVE_PATH = "models/"
    SCALER_PATH = os.path.join(MODEL_SAVE_PATH, "scaler.joblib")
    # --- [اصلاح] مسیر پایه برای فایل‌های state و نتایج ---
    STATE_FILE_PATH = os.path.join(MODEL_SAVE_PATH, "moga_state.pkl") # Template, will be modified by trainer
    RESULTS_CSV_PATH = os.path.join(MODEL_SAVE_PATH, "evolution_log.csv") # Template, will be modified by trainer
    TOTAL_TIMESTEPS_PER_AGENT = 1500

class DataSplit:
    FINAL_TEST_RATIO = 0.15

class Evolutionary:
    POPULATION_SIZE = 4
    NUM_GENERATIONS = 3
    
    MUTATION_ETA = 20.0
    CROSSOVER_PROB = 0.9
    
    N_JOBS = -1
    
    MINIMUM_TOTAL_TRADES = 40
    TRADE_BALANCE_RATIO_THRESHOLD = 0.2