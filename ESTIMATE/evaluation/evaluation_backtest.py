# %%
import pandas as pd
import numpy as np
from collections import namedtuple
from evaluation import metrics
#%%
import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")
#%%
Metrics = namedtuple(
    "Metrics",
    [
        "Return",
        "SR",
        "MDD"
    ]
)
#%%
def evaluate(y_trues, y_preds):
    """
    Evaluate returns, Sharpe ratio, and MDD for a specific phase where both y_preds
    and ground truths are in DataFrame format.
    
    Args:
        y_preds (pd.DataFrame): Predicted values for all stocks in the phase (index: date, columns: tickers).
        y_trues (pd.DataFrame): Actual values for all stocks in the phase (index: date, columns: tickers).
        top_k (int): Number of top stocks to consider for evaluation.
        
    Returns:
        dict: Performance metrics including total return, Sharpe ratio, and MDD.
    """
    
    top_k = 20
    bt_long = 1.0
    
    y_true = pd.concat(
        [y_trues[key]['trend_return'] for key in y_trues.keys()], 
        axis=1
    )
    y_true.columns = y_trues.keys()

    # Ensure y_preds and y_trues have the same structure
    assert y_preds.shape == y_true.shape, "Shape mismatch between y_preds and y_trues"
    assert all(y_preds.columns == y_true.columns), "Column mismatch between y_preds and y_trues"
    
    sharpe_li = []
    cumulative_returns = [1.0] 
    
    # Iterate over each row (date)
    for date in y_preds.index:
        prediction_row = y_preds.loc[date].values # Predicted values for the date
        y_true_row = y_true.loc[date].values # Actual values for the date

        rank_pre = np.argsort(prediction_row) # Top-K ranking for y_preds
        top_k_indices = rank_pre[-top_k:] # Select indices of top-K y_preds

        # Backtesting for top-k
        real_ret_rat_top_k = np.sum(y_true_row[top_k_indices]) / top_k
        bt_long += real_ret_rat_top_k
        sharpe_li.append(real_ret_rat_top_k)
        cumulative_returns.append(bt_long)
    #%%
    """Total return"""
    Return = bt_long - 1
    #%%
    """Sharpe ratio"""
    SR = np.array(sharpe_li)
    SR = (np.mean(sharpe_li) / np.std(sharpe_li)) * 15.87 if len(sharpe_li) > 1 else None
    #%%
    """Maximum Drawdown (MDD)"""
    cumulative_returns = np.array(cumulative_returns)
    peak_value = np.maximum.accumulate(cumulative_returns)
    
    MDD = cumulative_returns - peak_value 
    MDD /= peak_value
    MDD = abs(np.min(MDD))
    #%%
    return Metrics(Return, SR, MDD)
# %%
