import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
# Unified split utilities
from .split_utils import get_split_sizes, DEFAULT_SPLIT_FRAC
from typing import Optional, Tuple, Dict, Any, List

# Dynamic data path resolution (e.g., Kaggle input datasets)
from . import resolve_data_path

class JenaClimateDataset(Dataset):
    """
    JenaClimateDataset for multivariate time series forecasting on the Jena Climate Kaggle dataset.
    Loads CSV, normalizes features, and provides sliding windows for input/output.
    Each item is a dict with keys:
        'input': FloatTensor[input_len, input_dim]
        'target': FloatTensor[output_len, output_dim]
    """
    def __init__(
        self,
        data: np.ndarray,
        input_len: int = 96,
        output_len: int = 24,
        target_col: int = 0,
    ) -> None:
        super().__init__()
        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        self.target_col = target_col
        self.indices = self._compute_indices()

    def _compute_indices(self) -> List[int]:
        total_len = self.input_len + self.output_len
        return [i for i in range(len(self.data) - total_len + 1)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        x = self.data[i : i + self.input_len]  # [input_len, input_dim]
        y = self.data[i + self.input_len : i + self.input_len + self.output_len, self.target_col]  # [output_len]
        if y.ndim == 1:
            y = y[:, None]
        return {
            'inputs': torch.tensor(x, dtype=torch.float32),
            'targets': torch.tensor(y, dtype=torch.float32),
        }

    @staticmethod
    def get_splits(
        csv_path: str,
        input_len: int = 96,
        output_len: int = 24,
        split_frac: Dict[str, float] | None = None,
        target_col: Optional[str] = 'T (degC)',
    ) -> Tuple['JenaClimateDataset', 'JenaClimateDataset', 'JenaClimateDataset']:
        """
        Loads CSV, splits chronologically, fits scaler on train, applies to all, returns datasets.
        """
        csv_path = resolve_data_path(csv_path)
        df = pd.read_csv(csv_path)
        # Drop date/time column if present
        for col in ['Date Time', 'date', 'datetime', 'timestamp']:
            if col in df.columns:
                df = df.drop(columns=[col])
        # Determine target column index
        if target_col is not None:
            if isinstance(target_col, str):
                target_idx = df.columns.get_loc(target_col)
            else:
                target_idx = int(target_col)
        else:
            target_idx = 0  # default to first column
        data = df.values.astype(np.float32)
        n_train, n_val, n_test = get_split_sizes(len(data), split_frac)
        # Chronological split
        train_data = data[:n_train]
        val_data = data[n_train:n_train+n_val]
        test_data = data[n_train+n_val:]
        # Normalize (fit scaler on train, apply to all)
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)
        # Build datasets
        train_ds = JenaClimateDataset(train_data, input_len, output_len, target_col=target_idx)
        val_ds = JenaClimateDataset(val_data, input_len, output_len, target_col=target_idx)
        test_ds = JenaClimateDataset(test_data, input_len, output_len, target_col=target_idx)
        
        # Attach the scaler to all datasets for denormalization during evaluation
        train_ds.scaler = scaler
        val_ds.scaler = scaler
        test_ds.scaler = scaler
        
        return train_ds, val_ds, test_ds

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test JenaClimateDataset loader.")
    parser.add_argument('--csv', type=str, required=True, help='Path to Jena CSV.')
    parser.add_argument('--input_len', type=int, default=96)
    parser.add_argument('--output_len', type=int, default=24)
    args = parser.parse_args()
    train, val, test = JenaClimateDataset.get_splits(args.csv, args.input_len, args.output_len)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    sample = train[0]
    print(f"input shape: {sample['input'].shape}, target shape: {sample['target'].shape}") 