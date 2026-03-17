#%%
import os
#%%
DATASET_CONFIGS = {
    'NASDAQ': {
        'stock_num': 1026,
        'n_subclusters': 4,
        'valid_index': 756,
        'test_index': 1008,
        'data_path': '../data/NASDAQ.pkl',
    },
    'NYSE': {
        'stock_num': 1737,
        'n_subclusters': 6,
        'valid_index': 756,
        'test_index': 1008,
        'data_path': '../data/NYSE.pkl',
    },
    'SP500': {
        'stock_num': 463,
        'n_subclusters': 4,
        'indicators': {
            'close_sma': {
                'medium_period': 10,
                'slow_period': 20,
            },
            'macd': {
                'short_period' : 9,
                'medium_period': 12,
                'slow_period': 26,
            },
            'mfi': {
                'medium_period': 10,
                'slow_period': 20,
            },
            'ohlcv_ratio': {
                'period': 5,
            },
            'rsi': {
                'medium_period': 10,
                'slow_period': 20,
            },
            'trend_return': {},
        },
        'max_slow_period': 20,
        'n_step_ahead': 5,
        'outlier_threshold': 20,
        'start_train': '2013-01-01',
        'start_test': '2015-01-01',
        'target_col': 'trend_return',
        'include_target': False,
        'res_path': './backtest_res/',
        'top_k': 20,
        'start_backtest': '2015-05-01',
        'backtest_month': 8,
        'end_backtest_final': '2022-01-01',
    }
}
#%%
def get_config(market_name): 
    config = {**DATASET_CONFIGS[market_name]}
    config['results_dir'] = f"../results/{market_name}"
    os.makedirs(config['results_dir'], exist_ok=True)
    
    return config