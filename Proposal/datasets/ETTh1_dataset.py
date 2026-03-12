#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from modules.TimeFeatures import time_features
#%%
class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        args,
        flag='train',
        scale=True, 
        freq='h',
    ):
        # info
        self.timeenc = 0 if args.embed != 'timeF' else 1
        self.features=args.features
        self.target=args.target
        self.scale = scale
        self.freq = freq
        self.add_noise = args.add_noise
        self.cycle = args.cycle
        
        # seasonal_patterns=args.seasonal_patterns
        # noise_amp=args.noise_amp
        # noise_freq_percentage=args.noise_freq_percentage
        # noise_seed=args.noise_seed
        # noise_type=args.noise_type
        # data_percentage=args.data_percentage
        # rank_ratio=args.rank_ratio
        # pca_dim=args.pca_dim
        # reinit=args.reinit
        # shift=args.shift
        # num_freqs=args.num_freqs
        # speedup_sklearn=args.speedup_sklearn
        # align_type=args.align_type
        # load_from_disk=args.load_from_disk
        
        # size [seq_len, label_len, pred_len]
        size=[args.seq_len, args.label_len, args.pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # init
        assert flag in ['train', 'test', 'val']
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]  

        # self.root_path = root_path
        # self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv('./data/ETTh1.csv')

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

