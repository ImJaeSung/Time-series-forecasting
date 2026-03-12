#%%
import os
import copy

import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch import nn
import torch.nn.functional as F

from evaluation.evaluation import evaluate
#%%
def train_function(model, train_dataloader, test_dataloader, config, optimizer, phase, device): 
    
    best_acc = -9999
    for epoch in range(config['epochs']):
        #%%
        logs = {
            f'loss (Phase {phase})': [],
            f'huber (Phase {phase})': [], 
            f'corr (Phase {phase})': [],
            f'valid_loss (Phase {phase})': [],
            f'valid_huber (Phase {phase})': [], 
            f'valid_corr (Phase {phase})': [],
        }
        model.train()
        #%%
        for _, (X_batch, y_batch) in tqdm(enumerate(train_dataloader), desc='(Train) inner loop...'):
            #%%
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # (B, S, T, P), (B, S)
            prediction, clustering_info = model(X_batch.squeeze(0)) # (S, 1)
            prediction = prediction.reshape(1, -1)
        
            assert prediction.shape == y_batch.shape
            
            optimizer.zero_grad()
            
            loss_ = []
            
            """1. huber loss"""
            huber = nn.HuberLoss(delta=1.0)(prediction, y_batch)
            loss_.append((f'huber (Phase {phase})', huber))
            
            """2. correlation-aware loss"""
            pred_normalized = (prediction - torch.mean(prediction)) 
            pred_normalized /= torch.std(prediction) + 1e-8
            y_batch_normalized = (y_batch - torch.mean(y_batch)) 
            y_batch_normalized /= torch.std(y_batch) + 1e-8
            
            corr = torch.mean(pred_normalized * y_batch_normalized)
            loss_.append((f'corr (Phase {phase})', 1 - corr))
            #%%
            loss = (1 - config['corr_weight']) * huber + config['corr_weight'] * (1 - corr)
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
                prediction, _ = model(X_valid.squeeze(0)) # (S, 1)
                prediction = prediction.reshape(1, -1)
                
                y_trues.append(y_valid.cpu().numpy())
                y_preds.append(prediction.cpu().numpy())
                
                assert y_valid.shape == prediction.shape
                
                valid_loss_ = []
                #%%
                """validation loss"""
                valid_huber = nn.HuberLoss(delta=1.0)(prediction, y_valid)
                valid_loss_.append((f'valid_huber (Phase {phase})', valid_huber))
            
                pred_normalized = (prediction - torch.mean(prediction)) 
                pred_normalized /= torch.std(prediction) + 1e-8
                y_valid_normalized = (y_valid - torch.mean(y_valid)) 
                y_valid_normalized /= torch.std(y_valid) + 1e-8
                
                valid_corr = torch.mean(pred_normalized * y_valid_normalized)
                valid_loss_.append((f'valid_corr (Phase {phase})', (1 - valid_corr)))

                valid_loss = (1 - config['corr_weight']) * valid_huber + config['corr_weight'] * (1 - valid_corr)
                valid_loss_.append((f'valid_loss (Phase {phase})', valid_loss))

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
        y_true = np.concatenate(y_trues, axis=1)
        y_pred = np.concatenate(y_preds, axis=1)
        
        results = evaluate(y_true, y_pred)
        
        test_acc = results._asdict().get('accuracy')
        if test_acc > best_acc:
            best_acc = test_acc
            best_params = copy.deepcopy(model.state_dict())

        for x, y in results._asdict().items():
            print(f"{x} (Phase {phase}): {y:.3f}")
            wandb.log({f"{x} (Phase {phase})": y})
    #%% 
    return best_params
# %%
