# src/metrics_calculator.py
import pandas as pd
import numpy as np

def calculate_metrics(pnl_history, trade_history):
    """
    Calculates a dictionary of performance metrics from PnL and trade history.
    """
    metrics = {
        'profit_factor': 0.0, 'sharpe': 0.0, 'drawdown': 10000.0,
        'total_pnl': 0.0, 'win_rate': 0.0, 'total_trades': 0,
        'avg_pnl_trade': 0.0, 'payoff_ratio': 0.0, 'avg_duration': 0.0,
        'long_trades': 0, 'short_trades': 0,
        'sortino': 0.0, 'calmar': 0.0,
        'avg_win_pips': 0.0, 'avg_loss_pips': 0.0,
    }
    if not pnl_history:
        return metrics

    pnl_series = pd.Series(pnl_history)
    metrics['total_trades'] = len(pnl_series)
    
    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss = abs(pnl_series[pnl_series < 0].sum())
    
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 1e-9 else 999.0
    
    pnl_std = pnl_series.std()
    metrics['sharpe'] = pnl_series.mean() / pnl_std if pnl_std > 1e-9 else 0.0

    cumulative_pnl = pnl_series.cumsum()
    peak = cumulative_pnl.cummax()
    metrics['drawdown'] = (peak - cumulative_pnl).max()

    metrics['total_pnl'] = pnl_series.sum()
    metrics['win_rate'] = (pnl_series > 0).mean() * 100 if not pnl_series.empty else 0.0
    metrics['avg_pnl_trade'] = pnl_series.mean()

    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0
    metrics['avg_win_pips'] = avg_win
    metrics['avg_loss_pips'] = avg_loss

    metrics['payoff_ratio'] = avg_win / avg_loss if avg_loss > 1e-9 else 999.0
    
    downside_returns = pnl_series[pnl_series < 0]
    downside_std = downside_returns.std()
    metrics['sortino'] = pnl_series.mean() / downside_std if downside_std > 1e-9 else 0.0
    
    # --- <<-- FIX: Handle both object and dict types for trade_history -->> ---
    if trade_history:
        # Check the type of the first element to decide access method
        is_dict = isinstance(trade_history[0], dict)
        
        if is_dict:
            total_hours = sum(t.get('duration_candles', 0) for t in trade_history)
            metrics['avg_duration'] = np.mean([t.get('duration_candles', 0) for t in trade_history])
            metrics['long_trades'] = sum(1 for t in trade_history if t.get('direction') == 'Long')
            metrics['short_trades'] = sum(1 for t in trade_history if t.get('direction') == 'Short')
        else: # Assumes it's an object with attributes
            total_hours = sum(t.duration_candles for t in trade_history)
            metrics['avg_duration'] = np.mean([t.duration_candles for t in trade_history])
            metrics['long_trades'] = sum(1 for t in trade_history if t.direction == 'Long')
            metrics['short_trades'] = sum(1 for t in trade_history if t.direction == 'Short')
    else:
        total_hours = len(pnl_series)

    annual_return = metrics['total_pnl'] * (252 * 24 / total_hours) if total_hours > 0 else 0
    metrics['calmar'] = annual_return / metrics['drawdown'] if metrics['drawdown'] > 1e-9 else 0.0
    
    return metrics