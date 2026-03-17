#%%
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layer import *
#%%
class Model(nn.Module):
    def __init__(self, config, snapshot):
        super(Model, self).__init__()
        self.config = config
        self.snapshot = snapshot
        
        if self.snapshot is not None:
            self.num_snapshot = len(self.snapshot)
            self.params_of_snapshot = nn.Parameter(torch.Tensor(self.num_snapshot))
            nn.init.uniform_(self.params_of_snapshot, 0, 0.99)

        self.feature_embedding = nn.Linear(config['num_features'], config['d_model'])
        self.positional_encoding = PositionalEncoding(config['d_model'])
        self.individual_attn = IntraStockAttention(config['d_model'], config['num_heads'], config['dropout'])
        self.intensity_func = IntensityFunction(config['d_model'])
        self.mixing_layer = nn.Sequential(
            nn.Linear(config['d_model'] * 2, config['d_model']),
            nn.ReLU(),
            nn.LayerNorm(config['d_model'])
        )
        self.temporal_attn = TemporalAttention(config['d_model'], config['dropout'])
        self.dropout = nn.Dropout(config['dropout'])

        self.whconv1 = WaveletHypergraphConvolution(config['d_model'], config['num_stocks'], K1=2, K2=2, approx=False)
        self.whconv2 = WaveletHypergraphConvolution(config['d_model'], config['num_stocks'], K1=2, K2=2, approx=False)

        self.dynamic_graph_layer = DynamicGraphLearning(d_model=config['d_model'], num_edges=config['num_edges'])

        if config['fusion_mode'] == "sum":
            self.beta = nn.Parameter(torch.tensor([0.33,0.33,0.33]))

        elif config['fusion_mode'] == "adaptive":
            self.repre_selection = RepresentationSelection(d_model=config['d_model'])

        self.mlp = MLP(config['d_model'], fusion_mode=config['fusion_mode'])

    def forward(self, x):
        """
        inputs shape: (S, T, F)
        S: the number of stocks
        T: length of sequence
        F: the number of features
        D: dimension of embeddings
        """
        # Feature Embedding and Positional Encoding
        x = self.feature_embedding(x) # (S, T, D)
        x = self.positional_encoding(x) # (S, T, D)

        # Individual Stock Attention
        x = self.individual_attn(x) # (S, T, D)
        intensity = self.intensity_func(x) # (S, T, D)
        x = torch.cat([x, intensity], dim=-1) # (S, T, 2D)
        x = self.mixing_layer(x) # (S, T, D)

        # Temporal Attention
        x, _ = self.temporal_attn(x) # (S, D)

        # Wavelet Hypergraph Convolution
        repre = []
        for snapshot_idx in range(self.num_snapshot):
            hyper_repre_1 = F.leaky_relu(self.whconv1(x, self.snapshot, snapshot_idx), 0.1) # (S, D)
            hyper_repre_1 = self.dropout(hyper_repre_1)
            hyper_repre_2 = F.leaky_relu(self.whconv2(hyper_repre_1, self.snapshot, snapshot_idx), 0.1) # (S, D)
            repre.append(hyper_repre_2)

        hyper_repre = torch.zeros_like(hyper_repre_2)
        for snapshot_idx in range(self.num_snapshot):
            hyper_repre += repre[snapshot_idx] * self.params_of_snapshot[snapshot_idx] # (S, D)

        # Dynamic Graph Representation
        dynamic_repre, _ = self.dynamic_graph_layer(x) # (S, D)

        if self.config['fusion_mode'] == "sum": # Final Model

            weights = torch.softmax(self.beta, dim=0)
            fused_representation = (
                weights[0] * x +
                weights[1] * dynamic_repre +
                weights[2] * hyper_repre
            ) # (S, D)
        
        elif self.config['fusion_mode'] == "cat":

            fused_representation = torch.cat([x, dynamic_repre, hyper_repre], dim=-1) # (S, 3D)

        elif self.config['fusion_mode'] == "adaptive":
            
            fused_representation = self.repre_selection(x, dynamic_repre, hyper_repre) # (S, D)

        # Final Prediction
        x = self.mlp(fused_representation) # (S, D or 3D) -> (S)

        return x # (S)
    
    def backtesting(self, config, train_dataset, device):
        X_backtest, y_backtest = train_dataset.gen_backtest_data(
            config, config['start_backtest'], config['end_backtest'])
        X_backtest = torch.tensor(X_backtest, dtype=torch.float32)
        X_backtest = X_backtest.to(device)
        #%%
        y_preds = np.zeros(
            (
                len(X_backtest), 
                len(train_dataset.sp500_tickers())
            )
        ) # (T, N)
        #%%
        self.eval()
        with torch.no_grad():
            for i in range(0, len(X_backtest), config['batch_size']):
                st = i
                ed = i + config['batch_size']
                
                input = X_backtest[st:ed].squeeze(0).to(device)
                y_pred = self(input)
                y_pred = y_pred.cpu().detach().numpy()
                y_preds[st:ed] = y_pred.reshape(1, -1)

            y_preds = pd.DataFrame(
                y_preds, 
                columns=y_backtest.keys(), 
                index=y_backtest[
                    list(y_backtest.keys())[0]
                ].index
            )
        
        return y_backtest, y_preds