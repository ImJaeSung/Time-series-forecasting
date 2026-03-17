"""
Reference:
[1] https://github.com/SJTU-DMTai/MASTER/blob/master/master.py
"""
#%%
import torch
from torch import nn

from modules.layers import (
    Gate,
    PositionalEncoding, 
    TAttention, 
    SAttention, 
    TemporalAttention
)
#%%
class MASTER(nn.Module):
    def __init__(
        self, 
        config
    ):
        super(MASTER, self).__init__()
        #%%
        """market"""
        self.gate_input_start_index = 17
        self.gate_input_end_index = 38
        d_gate_input = (self.gate_input_end_index - self.gate_input_start_index) # F'
        self.feature_gate = Gate(
            d_gate_input, 
            config['fea_num'],
            beta=config['beta']
        )
        #%%
        """feature"""
        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(config['fea_num'], config['d_model']),
            PositionalEncoding(config['d_model']),
            
            # intra-stock aggregation
            TAttention(
                d_model=config['d_model'], 
                nhead=config['num_heads'], 
                dropout=config['dropout']
            ),
            
            # inter-stock aggregation
            SAttention(
                d_model=config['d_model'], 
                nhead=config['num_heads'], 
                dropout=config['dropout']
            ),
            TemporalAttention(d_model=config['d_model']),
            
            # decoder
            nn.Linear(config['d_model'], 1)
        )
    #%%
    def forward(self, x):
        #%%
        # x = X_batch.squeeze(0) # (S, T, P)
        src = x[:, :, :self.gate_input_start_index] # (S, T, P1) only indicator
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index] # (S, T, P2) only market
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
       
        output = self.layers(src)
        
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