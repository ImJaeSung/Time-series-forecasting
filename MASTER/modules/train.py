#%%
import os
import copy

import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch import nn
import torch.nn.functional as F

from evaluation.evaluation import evaluate
#%%
def train_function(model, train_dataloader, test_dataloader, config, optimizer, phase, device):
    #%%    
    best_acc = -9999
    #%%
    for epoch in range(config['epochs']):
        #%%
        logs = {
            f'loss (Phase {phase})': [],
            f'valid_loss (Phase {phase})': [],
        }
        model.train()
        #%%
        for _, (X_batch, y_batch) in tqdm(enumerate(train_dataloader), desc='(Train) inner loop...'):
            #%%
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # (B, S, T, P), (B, S)
            y_batch = y_batch.T
            prediction = model(X_batch.squeeze(0)) # (S, 1)
            #%%
            assert prediction.shape == y_batch.shape
            
            optimizer.zero_grad()
            
            loss_ = []
            
            """MSE loss"""
            loss = nn.MSELoss()(prediction, y_batch)
            loss_.append((f'loss (Phase {phase})', loss))
            #%%
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.detach()
            
            optimizer.step()
            
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
        #%%
        model.eval()
        
        y_trues = []
        y_preds = []
        #%%
        with torch.no_grad():
            #%%
            for _, (X_valid, y_valid) in tqdm(enumerate(test_dataloader), desc='(Eval) inner loop...'):
                #%%
                X_valid, y_valid = X_valid.to(device), y_valid.to(device) # (B, S, T, P), (B, S)
                y_valid = y_valid.T
                prediction = model(X_valid.squeeze(0)) # (S, 1)
                
                y_trues.append(y_valid.T.cpu().numpy())
                y_preds.append(prediction.reshape(1, -1).cpu().numpy())
                
                assert y_valid.shape == prediction.shape
                
                valid_loss_ = []
                #%%
                """validation loss"""
                valid_mse = nn.MSELoss()(prediction, y_valid)
                valid_loss_.append((f'valid_loss (Phase {phase})', valid_mse))
                
                for x, y in valid_loss_:
                    logs[x] = logs.get(x) + [y.item()]
        #%%
        """Training results"""
        print_input = f"Epoch [{epoch+1:03d}/{config['epochs']}]"
        print_input += "".join(
            [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
        )
        print(print_input)
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        #%%
        """valid performance""" 
        y_true = np.concatenate(y_trues, axis=0)
        y_pred = np.concatenate(y_preds, axis=0)
        
        results_ = evaluate(y_true, y_pred)
        
        test_acc = results_._asdict().get('accuracy')
        if test_acc > best_acc:
            best_acc = test_acc
            best_params = copy.deepcopy(model.state_dict())

            for x, y in results_._asdict().items():
                print(f"{x} (Phase {phase}): {y:.3f}")
                wandb.log({f"{x} (Phase {phase})": y})
                
            results = pd.DataFrame([results_], index=[f"Phase_{phase}"]).T
    #%% 
    return best_params, results
# %%
