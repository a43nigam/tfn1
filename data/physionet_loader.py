# data/physionet_loader.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from . import resolve_data_path
from .split_utils import get_split_sizes, DEFAULT_SPLIT_FRAC

class PhysioNet2012Dataset(Dataset):
    """
    Dataset for the PhysioNet Challenge 2012 (Mortality Prediction).
    Handles irregularly sampled multivariate time series from ICU patients.
    """
    def __init__(self, data_dir: str, split: str = 'train', max_seq_len: int = 512):
        self.data_dir = resolve_data_path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        
        outcomes_path = os.path.join(self.data_dir, 'outcomes-a.csv')
        outcomes_df = pd.read_csv(outcomes_path)
        
        patient_dir = os.path.join(self.data_dir, 'set-a')
        all_patient_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('.txt')])
        
        # Create deterministic splits of patient files
        n_train, n_val, _ = get_split_sizes(len(all_patient_files))
        np.random.RandomState(42).shuffle(all_patient_files)
        
        if split == 'train':
            self.patient_files = all_patient_files[:n_train]
        elif split == 'val':
            self.patient_files = all_patient_files[n_train:n_train + n_val]
        else: # test
            self.patient_files = all_patient_files[n_train + n_val:]

        self.outcomes_map = outcomes_df.set_index('RecordID')['In-hospital_death'].to_dict()
        self.features = ['HR', 'SysABP', 'MeanABP', 'RespRate', 'Temp', 'WBC', 'pH', 'PaCO2']

    def __len__(self):
        return len(self.patient_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, 'set-a', self.patient_files[idx])
        df = pd.read_csv(file_path, index_col=False)
        df_pivot = df.pivot_table(index='Time', columns='Parameter', values='Value')
        df_pivot = df_pivot.reindex(columns=self.features)

        df_pivot.ffill(inplace=True)
        df_pivot.bfill(inplace=True)
        df_pivot.fillna(0, inplace=True)
        
        values = torch.tensor(df_pivot.values, dtype=torch.float32)
        time_in_minutes = df_pivot.index.str.split(':').str[0].astype(int) * 60 + df_pivot.index.str.split(':').str[1].astype(int)
        timestamps = torch.tensor(time_in_minutes.values, dtype=torch.float32) / 60.0 # Normalize to hours

        record_id = int(self.patient_files[idx].split('.')[0])
        label = torch.tensor(self.outcomes_map[record_id], dtype=torch.long)
        
        # Pad or truncate sequences and create attention mask
        seq_len = values.shape[0]
        attention_mask = torch.ones(seq_len, dtype=torch.float32)
        if seq_len > self.max_seq_len:
            values = values[:self.max_seq_len, :]
            timestamps = timestamps[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]
        elif seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            values = F.pad(values, (0, 0, 0, pad_len), 'constant', 0)
            timestamps = F.pad(timestamps, (0, pad_len), 'constant', 0)
            attention_mask = F.pad(attention_mask, (0, pad_len), 'constant', 0)

        return {
            'inputs': values,
            'positions': timestamps.unsqueeze(-1),
            'labels': label,
            'attention_mask': attention_mask
        }

    @staticmethod
    def get_splits(data_dir: str, max_seq_len: int = 512, **kwargs):
        return (
            PhysioNet2012Dataset(data_dir, split='train', max_seq_len=max_seq_len),
            PhysioNet2012Dataset(data_dir, split='val', max_seq_len=max_seq_len),
            PhysioNet2012Dataset(data_dir, split='test', max_seq_len=max_seq_len)
        ) 