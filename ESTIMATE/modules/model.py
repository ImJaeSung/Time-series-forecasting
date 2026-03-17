#%%
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

from modules.layers import get_subsequent_mask, TemporalAttention
from modules.layers import DLSTM
from modules.layers import HWNNLayer
#%%
class Estimate(nn.Module):
    def __init__(
        self, 
        config,
        train_dataset, 
    ):
        super(Estimate, self).__init__()
        #%%
        self.config = config        
        self.train_dataset = train_dataset
        self.hyper_snapshot_num = len(train_dataset.hypergraphsnapshot)
        self.par = torch.nn.Parameter(torch.Tensor(self.hyper_snapshot_num))
        torch.nn.init.uniform_(self.par, 0, 0.99)
        #%%
        self.embedding = nn.Linear(config['num_features'], config['hidden_dim'])        
        self.lstm1 = DLSTM(
            config['hidden_dim'] * config['stock_num'], 
            config['rnn_units'] * config['stock_num'] * 2, 
            config['stock_num']
        ) 
        self.ln_1 = nn.LayerNorm(config['rnn_units'] * config['stock_num'] * 2) 
        
        self.lstm2 = DLSTM(
            config['rnn_units'] * config['stock_num'] * 2, 
            config['rnn_units'] * config['stock_num'],
            config['stock_num']
        ) 
        
        self.temp_attn = TemporalAttention(
            config['num_heads'], 
            config['rnn_units'] * config['stock_num'],
            d_k=8,
            d_v=8,
            dropout=config['dropout']
        )         
        self.dropout = nn.Dropout(config['dropout']) 
        
        self.hatt1 = HypergraphConv(
            config['rnn_units'] * config['lookback_length'], 
            config['rnn_units'] * config['lookback_length'],
            use_attention=False,
            heads=4, 
            concat=False, 
            negative_slope=0.1,
            dropout=config['dropout'],
            bias=True
        ) 
        self.hatt2 = HypergraphConv(
            config['rnn_units'] * config['lookback_length'],
            config['rnn_units'] * config['lookback_length'],
            use_attention=False,
            heads=1,
            concat=False,
            negative_slope=0.1, 
            dropout=config['dropout'],
            bias=True
        ) 
        
        self.convolution_1 = HWNNLayer(
            config['rnn_units'] * config['lookback_length'],
            config['rnn_units'] * config['lookback_length'],
            config['stock_num'],
            K1=3,
            K2=3,
            approx=False,
            data=train_dataset
        ) 

        self.convolution_2 = HWNNLayer(
            config['rnn_units'] * config['lookback_length'],
            config['rnn_units'] * config['lookback_length'],
            config['stock_num'],
            K1=3,
            K2=3,
            approx=False,
            data=train_dataset
        ) 

        self.mlp_1 = nn.Linear(config['rnn_units'] * 2, config['mlp_hidden']) 
        self.mlp_2 = nn.Linear(config['mlp_hidden'], 1) 
        self.mlp_3 = nn.Linear(config['lookback_length'], 1) 
        
    #%%
    def forward(self, inputs):        
        # X_batch.shape # (B, S, T, P)
        inputs = self.embedding(inputs) # (B, S, T, H)
        inputs = inputs.permute(0, 2, 1, 3) # (B, T, S, H)
        inputs = torch.reshape(
            inputs, (
                -1 , 
                self.config['lookback_length'], 
                self.config['hidden_dim'] * self.config['stock_num']
            )
        ) # (B, T, H*S)
        #%%
        slf_attn_mask = get_subsequent_mask(inputs).bool() # (B, T, T)              

        output, _ = self.lstm1(inputs) # (B, T, H*S*2)
        output = self.ln_1(output) # (B, T, H*S*2)
        #%%
        enc_output, _ = self.lstm2(output) # (B, T, H*S)
        enc_output, _ = self.temp_attn(
            enc_output,
            enc_output, 
            enc_output,
            mask=slf_attn_mask.bool()
        ) # (B, T, H*S)
        #%%
        enc_output = torch.reshape(
            enc_output, (
                -1,
                self.config['lookback_length'], 
                self.config['stock_num'], 
                self.config['rnn_units']
            )
        ) # (B, T, H, S)
        enc_output = self.dropout(enc_output)  
        enc_output = enc_output.permute(0, 2, 1, 3) # (B, S, T, H1)
        #%%
        outputs = []
        for i in range(enc_output.shape[0]):
            x = enc_output[i].reshape(
                self.config['stock_num'], 
                self.config['lookback_length'] *self.config['rnn_units']
            )
            channel_feature = []
            for snap_index in range(self.hyper_snapshot_num):
                deep_features_1 = F.leaky_relu(
                    self.convolution_1(
                        x,
                        snap_index,
                        self.train_dataset
                    ), 0.1
                )
                deep_features_1 = self.dropout(deep_features_1)
                
                deep_features_2 = self.convolution_2(
                    deep_features_1,
                    snap_index,
                    self.train_dataset
                )
                deep_features_2 = F.leaky_relu(deep_features_2, 0.1)
                channel_feature.append(deep_features_2)
                
            deep_features_3 = torch.zeros_like(channel_feature[0])
            for ind in range(self.hyper_snapshot_num):
                deep_features_3 = deep_features_3 + self.par[ind] * channel_feature[ind]
            outputs.append(deep_features_3)
        #%%   
        outputs = torch.stack(outputs) # (B, S, T*H1)
        hyper_output = outputs.reshape(
            -1, 
            self.config['stock_num'], 
            self.config['lookback_length'], 
            self.config['rnn_units']
        ) # (B, S, T, H1)
        #%%
        enc_output = torch.cat((enc_output, hyper_output), dim=-1) # (B, S, T, 2*H1)
        
        output = self.mlp_1(enc_output) # (B, S, T, 2*H1)        
        output = F.relu(output)
        output = self.mlp_2(output).squeeze(-1) # (B, S, T)             
        
        return output
    #%%
    def backtesting(self, config, train_dataset, device):
        #%%
        X_backtest, y_backtest = train_dataset.gen_backtest_data(
            config, config['start_backtest'], config['end_backtest'])
        X_backtest = torch.tensor(X_backtest, dtype=torch.float32)
        X_backtest = X_backtest.to(device)
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
                
                input = X_backtest[st:ed].to(device)
                y_pred = self(input)[:,:,-1]
                y_pred = y_pred.cpu().detach().numpy()
                y_preds[st:ed] = y_pred

            y_preds = pd.DataFrame(
                y_preds, 
                columns=y_backtest.keys(), 
                index=y_backtest[
                    list(y_backtest.keys())[0]
                ].index
            )
        
        return y_backtest, y_preds