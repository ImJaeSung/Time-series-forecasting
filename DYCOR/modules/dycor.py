#%%
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from modules.encoding import StockEncoding
from modules.clustering import DynamicStockClustering
from modules.intra_corr import IntraStockCorrelation
from modules.aggregation import InterClusterAggregator
#%%
class DYCOR(nn.Module):
    """
    Dynamic Correlation model for stock trend prediction.
    This is the main model that integrates all components:
    1. Stock Encoding - encodes historical features of stocks
    2. Dynamic Stock Clustering - clusters stocks into market segments and subsegments
    3. Intra-Stock Correlation - models relationships between stocks within subcluster
    4. Inter-Cluster Aggregation - combines multiple views of each stock
    5. Prediction - forecasts stock return ratios
    """
    def __init__(self, config):
        
        # stock_num,           # Number of stocks in the market
        # lookback_length,     # Length of historical time window
        # fea_num,             # Number of features per stock
        # hidden_dim,          # Dimension of hidden representations
        # min_var_ratio,       # Minimum variance ratio for PCA clustering
        # n_subclusters,       # Number of subclusters per market segment
        # temperature,         # Temperature for softmax in clustering
        # dropout              # Dropout probability
 
        super().__init__()
        
        self.stock_encoding = StockEncoding(
            config,
            time_steps=config['lookback_length'],
            channels=config['fea_num']
        )
        self.clustering = DynamicStockClustering(
            embedding_dim=config['hidden_dim'],
            min_var_ratio=config['min_var_ratio'],
            temperature=config['temperature'],
            n_subclusters=config['n_subclusters']
        )
        self.intra_corr = IntraStockCorrelation(
            hidden_dim=config['hidden_dim'],
            n_subclusters=config['n_subclusters'],
            dropout=config['dropout_prob']
        )
        self.aggregation = InterClusterAggregator(
            hidden_dim=config['hidden_dim'],
            n_subclusters=config['n_subclusters']
        )
        self.prediction = nn.Sequential(
            nn.LayerNorm(config['hidden_dim']),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.GELU(),
            nn.Dropout(config['dropout_prob']),
            nn.Linear(config['hidden_dim'], 1)
        )

    def forward(self, batch_x):
        """
        Forward pass of the DYCOR model.
        
        Args:
            batch_x: Batch of stock features [N, fea_num, lookback_length]
            
        Returns:
            pred: Predicted return ratios [N, 1]
            clustering_info: Dictionary containing clustering information for analysis
        """

        stock_embs = self.stock_encoding(batch_x)
        soft_cluster_weights, latent_subsegment_assignments, current_n_clusters, market_stock_similarities, market_reps = self.clustering(stock_embs)
        interval_outputs = self.intra_corr(stock_embs, soft_cluster_weights, latent_subsegment_assignments, current_n_clusters)
        final_stock_reps = self.aggregation(stock_embs, soft_cluster_weights, latent_subsegment_assignments, interval_outputs, current_n_clusters)
        pred = self.prediction(final_stock_reps)

        clustering_info = {
            'market_stock_similarities': market_stock_similarities,
            'market_reps': market_reps,
            'soft_cluster_weights': soft_cluster_weights,
            'latent_subsegment_assignments': latent_subsegment_assignments
        }

        return pred, clustering_info
    
    def backtesting(self, config, train_dataset, device):
        X_backtest, y_backtest = train_dataset.gen_backtest_data(
            config, config['start_backtest'], config['end_backtest_final'])
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
        for i in range(0, len(X_backtest), config['batch_size']):
            st = i
            ed = i + config['batch_size']
            
            input = X_backtest[st:ed].squeeze(0).to(device)
            y_pred, _ = self(input)
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
