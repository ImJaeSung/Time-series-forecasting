#%%
import os
import copy
import sys
import argparse
import importlib

import numpy as np

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.utility import (
    set_random_seed, 
    add_month, 
    add_month_to_string_time
)
#%%
import warnings
warnings.filterwarnings("ignore")
#%%
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "Estimate" # put your WANDB project name
# entity = "" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)
#%%
def get_args(debug):
    parser = argparse.ArgumentParser(description='Estimate-parameters')
    parser.add_argument('--model', type=str, default='Estimate')
    parser.add_argument('--market', type=str, default='SP500',
                        choices=['NASDAQ', 'NYSE', 'SP500'],
                        help='Market to train on (default: NASDAQ)')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed (default: based on current time)')
    
    parser.add_argument('--lookback_length', type=int, default=32, 
                        help='lookback window length')
    parser.add_argument('--dropout', type=int, default=0.4, 
                        help='dropout')
    
    parser.add_argument('--num_features', type=float, default=17, 
                        help='the number of features')
    parser.add_argument('--num_heads', type=float, default=4, 
                        help='the number of attention heads')
    parser.add_argument("--rnn_units", type=int, default=16, 
                        help="the number of RNN units")
    parser.add_argument('--hidden_dim', type=int, default=32, 
                        help='hidden dimension')
    parser.add_argument('--mlp_hidden', type=int, default=16, 
                        help='hidden dimension')

    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, 
                        help='weight decay')
    parser.add_argument('--steps', type=int, default=1, 
                        help='')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='batch size')
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
def main():    
    #%%
    config = vars(get_args(debug=False)) # default configuration
    config_module = importlib.import_module('datasets.config')
    importlib.reload(config_module)
    config = {**config, **config_module.get_config(config['market'])}
    #%%
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    wandb.config.update(config)
    #%%
    """Training per phase"""
    all_phase_results = []
    phase = 1
    flag = True
    #%%
    config['end_train'] = copy.copy(config['start_train'])
    config['end_test'] = copy.copy(config['start_backtest'])
    config['end_backtest'] = add_month_to_string_time(
        config['start_backtest'], config['backtest_month'])
    #%%
    while flag:
        #%%
        final_backtest = add_month_to_string_time(
            config['end_backtest'], 
            config['backtest_month']
        )
        if final_backtest > config['end_backtest_final']:
            flag = False # final backtesting duration
        
        print(f"Start (Phase {phase})...")
        #%%
        """dataset"""
        dataset_module = importlib.import_module('datasets.sp500_dataset')
        importlib.reload(dataset_module)
        StockDataset = dataset_module.StockDataset
        train_dataset = StockDataset(config, train=True)
        test_dataset = StockDataset(config, train=False)

        train_dataloader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(
            test_dataset, batch_size=config['batch_size'], shuffle=False)
        #%%
        """model"""
        model_module = importlib.import_module('modules.model')
        importlib.reload(model_module)
        model = getattr(model_module, "Estimate")(
            config, 
            train_dataset
        ).to(device)
        #%%
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        #%%
        """the number of parameters"""
        count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_params = count_parameters(model)
        print(f"Number of Parameters: {num_params / 1000:.1f}K")
        #%%
        """train"""
        train_module = importlib.import_module('modules.train')
        importlib.reload(train_module)
        best_params, pred_results = train_module.train_function(
            model,
            train_dataloader,
            test_dataloader,
            config,
            optimizer, 
            phase,
            device
        )
        #%%
        """save the model (each phase)"""
        base_name = f"{config['model']}_{config['market']}_Phase_{phase}_{config['lookback_length']}_{config['hidden_dim']}_{config['rnn_units']}"
        model_dir = f"./assets/models/{base_name}/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_name = f"{base_name}_{config['seed']}"
        torch.save(best_params, f"./{model_dir}/{model_name}.pth")
        #%%
        model.load_state_dict(best_params)
        artifact = wandb.Artifact(
            "_".join(model_name.split("_")[:-1]), 
            type='model',
            metadata=config)
        artifact.add_file(f"./{model_dir}/{model_name}.pth")
        artifact.add_file('./main.py')
        artifact.add_file(f'./datasets/sp500_dataset.py')
        artifact.add_file('./modules/train.py')
        artifact.add_file('./modules/model.py')
        wandb.log_artifact(artifact)
        #%%
        """backtesting"""
        y_trues, y_preds = model.backtesting(config, train_dataset, device)
        #%%
        evaluate_module = importlib.import_module('evaluation.evaluation_backtest')
        importlib.reload(evaluate_module)
        backtest_results = evaluate_module.evaluate(y_trues, y_preds)
        for x, y in backtest_results._asdict().items():
            print(f"{x} (Phase {phase}): {y:.3f}")
            wandb.log({f"{x} (Phase {phase})": y})
        #%%
        backtest_results = pd.DataFrame([backtest_results], index=[f"Phase_{phase}"]).T
        results = pd.concat([pred_results, backtest_results], axis=0)
        all_phase_results.append(results)
        #%%
        config['start_train'] = add_month_to_string_time(config['start_train'], config['backtest_month'])
        config['end_train'] = add_month_to_string_time(config['end_train'], config['backtest_month'])
        config['start_test'] = add_month_to_string_time(config['start_test'], config['backtest_month'])
        config['end_test'] = add_month_to_string_time(config['end_test'], config['backtest_month'])
        config['start_backtest'] = add_month_to_string_time(config['start_backtest'], config['backtest_month'])
        config['end_backtest'] = add_month_to_string_time(config['end_backtest'], config['backtest_month'])

        phase += 1
    #%%
    all_phase_results = pd.concat(all_phase_results, axis=1)  
    final_results = pd.concat(
        [
            all_phase_results, 
            all_phase_results.mean(axis=1).to_frame(name='Mean')
        ], axis=1
    ) 
    wandb.log(
        {
            "Test Result": wandb.Table(dataframe=final_results.reset_index())
        }
    )
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%%
if __name__ == "__main__":
    main()
# %%
