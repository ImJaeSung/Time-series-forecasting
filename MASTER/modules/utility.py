#%%
import random
from datetime import datetime

import numpy as np
import pandas as pd

import torch
#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed) 
#%%
def add_month(date, months=1):
    new_month = (date.month + months - 1) % 12 + 1
    new_year = date.year + (date.month + months - 1) // 12
    return date.replace(year=new_year, month=new_month)

#%%
def add_month_to_string_time(str_date, months=1):
    date = str(pd.to_datetime(str_date))
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    new_date = add_month(date, months=months)
    return new_date.strftime("%Y-%m-%d")