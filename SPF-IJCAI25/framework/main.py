#%%
import os
import yaml
import argparse
from datetime import datetime

import random
import numpy as np
import torch
#%%
from model import Models
from utils.common import seed_everything
from supervisor import get_tickers_sp500
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "spf" # put your WANDB project name
# entity = "xxx" # put your WANDB username

run = wandb.init(
    project=project,
    # entity=entity, 
    tags=["train"], # put tags of this python project
)
#%%
parser = argparse.ArgumentParser(description="SPF")
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--data_path', type=str, default='../data/sp500/')

parser.add_argument("--mode", type=str, default='train',
                    help="training or evaluation")

parser.add_argument("--d", type=int, default=32, 
                    help="d_model")
parser.add_argument("--nh", type=int, default=4, 
                    help="num_heads")
parser.add_argument("--dr", type=float, default=0.1, 
                    help="dropout")
parser.add_argument("--edges", type=int, default=64, 
                    help="num_edges")
parser.add_argument("--fusion", type=str, default='sum', 
                    choices=["sum", "cat", "adaptive"],
                    help="fusion mode")

parser.add_argument("--lr", type=float, default=1e-4, 
                    help="learning_rate")
args = parser.parse_args(args=[])
#%% 
def main():
    #%%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    #%%
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('./config/backtest_config.yaml', 'r') as f:
        backtest_config = yaml.safe_load(f)
    config['backtest'] = backtest_config

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        config['model']['cuda'] = True
    else:
        device = torch.device("cpu")
        config['model']['cuda'] = False
    #%%
    seed = args.seed
    mode = args.mode

    config['model']['d_model'] = args.d
    config['model']['num_heads'] = args.nh
    config['model']['dropout'] = args.dr
    config['model']['num_edges'] = args.edges
    config['model']['fusion_mode'] = args.fusion
    config['train']['learning_rate'] = args.lr

    seed_everything(seed)

    config['data']['data_path'] = args.data_path
    config['data']['symbols'] = get_tickers_sp500()
    current_time = datetime.now()
    run_name = current_time.strftime("%y%m%d-%H%M")
    config['run_name'] = run_name
    #%%
    model = Models(config=config, device=device)
    model.set_model_instanece(model)
    #%%
    if mode == 'train':
        model.fit()
    if key is not None:
        wandb.finish()
# %%
