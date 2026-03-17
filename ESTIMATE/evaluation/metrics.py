#%%
import math
import numpy as np

from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
#%%
def eval_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
#%%
def eval_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#%%
def eval_rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))
#%%
def eval_acc(y_true, y_pred):

    accs = []
    for i in range(y_true.shape[0]):

        y_true_sign = y_true[i] > 0
        y_pred_sign = y_pred[i] > 0

        total = len(y_true[i])
        num_true = (y_true_sign == y_pred_sign).sum()

        if total > 0:
            acc = num_true / total
        else:
            acc = 0

        accs.append(acc)

    return np.asanyarray(accs)
#%%
def eval_long_acc(y_true, y_pred, confidence_threshold = 0):
    res = y_true[y_pred > confidence_threshold]
    total = len(res)
    num_true = len(res[res > 0])
    if total > 0:
        acc = num_true / total
    else:
        acc = 0

    return acc, num_true

def eval_short_acc(y_true, y_pred, confidence_threshold = 0):
    res = y_true[y_pred < confidence_threshold]
    total = len(res)
    num_true = len(res[res < 0])
    if total > 0:
        acc = num_true / total
    else:
        acc = 0

    return acc, num_true

def eval_ic(y_true, y_pred):
    ics = []
    for i in range(y_true.shape[0]):
        ics.append(stats.pearsonr(y_true[i], y_pred[i])[0])
    return np.asanyarray(ics)

def eval_rank_ic(y_true, y_pred):
    rank_ics = []
    for i in range(y_true.shape[0]):
        rank_ics.append(stats.spearmanr(y_true[i], y_pred[i])[0])
    return np.asanyarray(rank_ics)

def eval_long_prec(y_true, y_pred, K=10):
    long_precs = []
    for i in range(y_true.shape[0]):
        pred = y_pred[i]
        gt = y_true[i]
        top_k_largest = np.argpartition(pred, -K)[-K:]
        top_return = gt[top_k_largest] * pred[top_k_largest]
        prec = len(top_return[top_return > 0]) / len(top_return)
        long_precs.append(prec)
    return np.asanyarray(long_precs)

def eval_short_prec(y_true, y_pred, K=10):
    short_precs = []
    for i in range(y_true.shape[0]):
        pred = y_pred[i]
        gt = y_true[i]
        top_k_smallest = np.argpartition(pred, K)[:K]
        top_return = gt[top_k_smallest] * pred[top_k_smallest]
        prec = len(top_return[top_return > 0]) / len(top_return)
        short_precs.append(prec)
    return np.asanyarray(short_precs)