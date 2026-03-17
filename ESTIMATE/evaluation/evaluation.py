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
        "mae",
        "rmse",
        "ic", 
        "icir",
        "rank_ic", 
        "rank_icir",
        "long_prec", 
        "short_prec",
        "accuracy",
    ]
)
#%%
def evaluate(y_true, y_pred):
    
    print("\n1. Mean Absolute Error (MAE)...")
    mae = metrics.eval_mae(y_true, y_pred)
    
    print("\n2. Root Mean Squared Error (RMSE)...")
    rmse = metrics.eval_rmse(y_true, y_pred)
    
    print("\n3. Information Coefficient (IC) & ICIR...")
    ics = metrics.eval_ic(y_true, y_pred)
    ic = np.mean(ics)
    icir = ic / (np.std(ics) + 1e-8)
    
    print("\n4. Rank Information Coeeficient (RankIC) & RankICIR...")
    rank_ics = metrics.eval_rank_ic(y_true, y_pred)
    rank_ic = np.mean(rank_ics)
    rank_icir = rank_ic / (np.std(rank_ics) + 1e-8)
    
    print("\n5. Long & short precision@k...")
    long_k_precs = metrics.eval_long_prec(y_true, y_pred)
    long_k_prec = np.mean(long_k_precs)
    
    short_k_precs = metrics.eval_short_prec(y_true, y_pred)
    short_k_prec = np.mean(short_k_precs)
    
    print("\n6. Accuracy...")
    accuracy = metrics.eval_acc(y_true, y_pred)
    accuracy = np.mean(accuracy)
    
    return Metrics(
        mae, rmse, 
        ic, icir, rank_ic, rank_icir, 
        long_k_prec, short_k_prec, accuracy
    )