#%%
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import datasets.indicator as indi
#%%
class StockDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.max_slow_period = max(
            [
                max(indi.values()) 
                for indi in config['indicators'].values() if indi.values()
            ]
        )
        #%%
        """Save the dataset"""
        dataset_dir = f"./assets/datasets/{config['market']}/"
        dataset_dir += "train/" if train else "test/"
        dataset_name = dataset_dir + f"{config['market']}_{config['start_train']}_{config['lookback_length']}" 
        #%%
        os.makedirs(dataset_dir, exist_ok=True)
        #%%
        try:
            data = np.load(dataset_name + ".npz")
            self.X = torch.tensor(data["X"], dtype=torch.float32)
            self.y = torch.tensor(data["y"], dtype=torch.float32)
            self.buy_prob_threshold = data["buy_prob_threshold"]
            self.sell_prob_threshold = data["sell_prob_threshold"]
        
        except Exception as e:
            print(e)
            X_train_full, y_train_full, X_test_full, y_test_full, buy_prob_threshold, sell_prob_threshold = self.split_train_test(config)
            
            self.X = X_train_full if train else X_test_full
            self.y = y_train_full[:, :, -1] if train else y_test_full[:, :, -1]
            self.buy_prob_threshold = buy_prob_threshold
            self.sell_prob_threshold = sell_prob_threshold
            
            np.savez(
                dataset_name, 
                X=self.X, 
                y=self.y, 
                buy_prob_threshold=buy_prob_threshold,
                sell_prob_threshold=sell_prob_threshold
            )
    #%%
    def split_train_test(self, config):
        train_data_storage = {}
        test_data_storage = {}
        buy_prob_threshold = []
        sell_prob_threshold = []
        #%%
        for idx, sym in tqdm(enumerate(self.sp500_tickers()), desc='Splitting dataset...'):
            #%%
            train_data = self.get_data_baseline(
                sym, 
                config['start_train'], 
                config['start_test'],
                config["lookback_length"] + self.max_slow_period, 
                config["n_step_ahead"]
            )
            test_data = self.get_data_baseline(
                sym, 
                config['start_test'], 
                config['start_backtest'], 
                config["lookback_length"] + self.max_slow_period, 
                config["n_step_ahead"]
            )
            #%%
            if idx == 0:
                max_train_trade_index = train_data.index 
                max_test_trade_index = test_data.index 
            #%%
            train_data = self.fill_missing_data(train_data, max_train_trade_index)
            test_data = self.fill_missing_data(test_data, max_test_trade_index)
            #%%
            train_data_storage[sym] = train_data
            test_data_storage[sym] = test_data
            #%%
            _, df_x_train, df_y_train = self.preprocess(train_data, config) # technical indicator & ratio
            _, df_x_test, df_y_test = self.preprocess(test_data, config)
            #%% 
            buy_prob_threshold.append(df_y_train.mean()) # buy = average of return ratio
            sell_prob_threshold.append(-df_y_train.mean()) # sell = average of negative return ratio
            #%%
            train_x = df_x_train.to_numpy()
            train_y = df_y_train.to_numpy()

            test_x = df_x_test.to_numpy()
            test_y = df_y_test.to_numpy()            
            #%%
            if idx == 0:
                X_train = np.zeros(
                    (
                        len(train_x) - config["lookback_length"]+1, 
                        len(self.sp500_tickers()),
                        config["lookback_length"], 
                        train_x.shape[-1]
                    )
                )
                y_train = np.zeros(
                    (
                        len(train_y) - config["lookback_length"]+1, 
                        len(self.sp500_tickers()), 
                        config["lookback_length"]
                    )
                )
                X_test = np.zeros(
                    (
                        len(test_x) - config["lookback_length"]+1,
                        len(self.sp500_tickers()), 
                        config["lookback_length"], 
                        test_x.shape[-1]
                    )
                )
                y_test = np.zeros(
                    (
                        len(test_y)-config["lookback_length"]+1, 
                        len(self.sp500_tickers()), 
                        config["lookback_length"]
                    )
                )
            #%%
            X_train[:,idx,:] = np.array(
                [
                    train_x[i: i + config["lookback_length"]] 
                    for i in range(len(train_x)-config["lookback_length"]+1)
                ]
            )
            y_train[:,idx,:] = np.array(
                [
                    train_y[i: i + config["lookback_length"]]
                    for i in range(len(train_y)-config["lookback_length"]+1)
                ]
            )
            X_test[:,idx,:] = np.array(
                [
                    test_x[i: i + config["lookback_length"]]
                    for i in range(len(test_x)-config["lookback_length"]+1)
                ]
            )
            y_test[:,idx,:] = np.array(
                [
                    test_y[i: i + config["lookback_length"]]
                    for i in range(len(test_y)-config["lookback_length"]+1)
                ]
            )
        #%%
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        return X_train, y_train, X_test, y_test, buy_prob_threshold, sell_prob_threshold
    #%%
    def get_data_baseline(self, symbol, start_date, end_date, lookback=0, lookforward=0):
        #%%
        df = np.load(
            "data/baseline_data_sp500.npy",
            allow_pickle=True
        ).item()[symbol]
        mask = (df.index >= start_date) & (df.index <= end_date) # duration
        data = df.loc[mask] # (#trainday, OHLCV)
        #%%
        if lookback:
            concat_df = df.loc[:data.index[0]].tail(lookback) # (lookback + slow, OHLCV)
            data = pd.concat(
                [data, concat_df], axis=0
            ).sort_index().drop_duplicates() 
        
        if lookforward:
            concat_df = df.loc[data.index[-1]:].head(lookforward + 1) # (lookahead, OHLCV)
            data = pd.concat(
                [data, concat_df], axis=0
            ).sort_index().drop_duplicates() 
        
        data = data.astype(float)
        
        return data
    #%%
    def fill_missing_data(self, df, max_trade_index):
        missing_points = np.setdiff1d(max_trade_index, df.index)
        df.fillna(np.NaN, inplace=True)
        df = df.interpolate(limit_direction = "both")
        
        if len(missing_points) > 0:
            
            for ind in missing_points:
                df.loc[ind] = np.nan
                
            df = df.sort_index(axis=0, ascending=True)
            for i in ["open", "high", "low", "close", "volume"]:
                df[f"{i}"] = df[f"{i}"].interpolate(limit_direction="both")
            
        return df
    #%%
    def sp500_tickers(self, start_date='2012-11-01', end_date='2022-01-01'):
        data = np.load('data/baseline_data_sp500.npy', allow_pickle=True).item()

        valid_tickers = []
        for t in data.keys():
            ticker = data[t]
            filtered = ticker[
                (ticker.index >= start_date)&(ticker.index <=end_date)
            ]
            
            if filtered.isna().sum().sum() == 0:
                valid_tickers.append(t)
                
        return valid_tickers
    #%%
    def preprocess(self, source_df, config):
        df = source_df.copy()
        features = []

        if "ohlcv_ratio" in config['indicators']:
            period = config['indicators']["ohlcv_ratio"]["period"]
            df, features = indi.ohlcv_ratio(df, features, period)

        if "close_ratio" in config['indicators']:
            medium_period = config['indicators']["close_ratio"]["medium_period"]
            slow_period = config['indicators']["close_ratio"]["slow_period"]
            df, features = indi.close_ratio(df, features, [medium_period, slow_period])

        if "volume_ratio" in config['indicators']:
            medium_period = config['indicators']["volume_ratio"]["medium_period"]
            slow_period = config['indicators']["volume_ratio"]["slow_period"]
            df, features = indi.volume_ratio(df, features, [medium_period, slow_period])

        if "close_sma" in config['indicators']:
            medium_period = config['indicators']["close_sma"]["medium_period"]
            slow_period = config['indicators']["close_sma"]["slow_period"]
            df, features = indi.close_sma(df, features, [medium_period, slow_period])

        if "volume_sma" in config['indicators']:
            medium_period = config['indicators']["volume_sma"]["medium_period"]
            slow_period = config['indicators']["volume_sma"]["slow_period"]
            df, features = indi.volume_sma(df, features, [medium_period, slow_period])
        
        if "close_ema" in config['indicators']:
            medium_period = config['indicators']["close_ema"]["medium_period"]
            slow_period = config['indicators']["close_ema"]["slow_period"]
            df, features = indi.close_ema(df, features, [medium_period, slow_period])

        if "volume_ema" in config['indicators']:
            medium_period = config['indicators']["volume_ema"]["medium_period"]
            slow_period = config['indicators']["volume_ema"]["slow_period"]
            df, features = indi.volume_ema(df, features, [medium_period, slow_period])

        if "atr" in config['indicators']:
            medium_period = config['indicators']["atr"]["medium_period"]
            slow_period = config['indicators']["atr"]["slow_period"]
            df, features = indi.atr(df, features, [medium_period, slow_period])

        if "adx" in config['indicators']:
            medium_period = config['indicators']["adx"]["medium_period"]
            slow_period = config['indicators']["adx"]["slow_period"]
            df, features = indi.adx(df, features, [medium_period, slow_period])
        
        if "kdj" in config['indicators']:    
            medium_period = config['indicators']["kdj"]["medium_period"]
            slow_period = config['indicators']["kdj"]["slow_period"]
            df, features = indi.kdj(df, features, [medium_period, slow_period])

        if "rsi" in config['indicators']:  
            medium_period = config['indicators']["rsi"]["medium_period"]
            slow_period = config['indicators']["rsi"]["slow_period"]          
            df, features = indi.rsi(df, features, [medium_period, slow_period])

        if "macd" in config['indicators']:    
            short_period = config['indicators']["macd"]["short_period"]
            medium_period = config['indicators']["macd"]["medium_period"]
            slow_period = config['indicators']["macd"]["slow_period"]        
            df, features = indi.macd(df, features, short_period, medium_period, slow_period)
        
        if "mfi" in config['indicators']:
            medium_period = config['indicators']["mfi"]["medium_period"]
            slow_period = config['indicators']["mfi"]["slow_period"]
            df, features = indi.mfi(df, features, [medium_period, slow_period])

        if "bb" in config['indicators']:                
            df, features = indi.bb(df, features)

        if "arithmetic_returns" in config['indicators']:
            df, features = indi.arithmetic_returns(df, features)
        
        if "obv" in config['indicators']:
            medium_period = config['indicators']["rsi"]["medium_period"]
            slow_period = config['indicators']["rsi"]["slow_period"]
            df, features = indi.obv(df, features, [medium_period, slow_period])

        if "ichimoku" in config['indicators']:
            fast_period = config['indicators']["ichimoku"]["fast_period"]
            medium_period = config['indicators']["ichimoku"]["medium_period"]
            slow_period = config['indicators']["ichimoku"]["slow_period"]        
            df, features = indi.ichimoku(df, features, fast_period, medium_period, slow_period)
        
        if "k_line" in config['indicators']:
            df, features = indi.k_line(df, features)
        
        if "eight_trigrams" in config['indicators']:
            df, features = indi.eight_trigrams(df, features)

        if "trend_return" in config['indicators']:
            df, features = indi.trend_return(df, features, config['n_step_ahead'])

        df = df[self.max_slow_period:-config['n_step_ahead']]
        if not config['include_target']:
            features.remove(config['target_col'])
            
        df = df.interpolate(limit_direction="both")
        df = indi.remove_outliers(df, features, threshold=config['outlier_threshold'])
        df_y = df[config['target_col']]
        df_x = df.filter((features))

        return df, df_x, df_y
    #%%
    def gen_backtest_data(self, config, start_date, end_date):
        y_pred = {}
        X_backtest = []

        for idx, sym in tqdm(enumerate(self.sp500_tickers()), desc='Generating backtest data...'):
            data = self.get_data_baseline(
                sym,
                start_date,
                end_date, 
                config["lookback_length"] + self.max_slow_period + 5,
                config['n_step_ahead']
            )

            if idx == 0:
                max_data_trade_index = data.index
                
            data = self.fill_missing_data(data, max_data_trade_index)
        
            df, df_x, _ = self.preprocess(data, config)
            df = df.iloc[config["lookback_length"]:]
            df_x = df_x.to_numpy()
            
            y_pred[sym] = df[["open", "high", "low", "close", "volume", "trend_return"]]
            input_data = np.array(
                [
                    df_x[i: i + config["lookback_length"]] 
                    for i in range(len(df_x) - config["lookback_length"])
                ]
            )
            X_backtest.append(input_data)
            
        num_sample = X_backtest[0].shape[0]
        X_backtest = np.vstack((X_backtest))
        X_backtest = np.reshape(
            X_backtest, 
            (
                num_sample, 
                len(self.sp500_tickers()), 
                X_backtest.shape[1], 
                X_backtest.shape[2]
            )
        )
        
        return X_backtest, y_pred
    #%%
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
#%%
