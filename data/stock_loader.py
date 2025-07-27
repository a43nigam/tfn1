import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
# Use split helper for consistent ratios
from .split_utils import get_split_sizes, DEFAULT_SPLIT_FRAC
from typing import Optional, Tuple, Dict, Any, List

# Dynamic data path resolution (e.g., Kaggle input datasets)
from . import resolve_data_path

class StockMarketDataset(Dataset):
    """
    StockMarketDataset for multivariate time series forecasting on Kaggle stock market data.
    Loads CSV, normalizes features, and provides sliding windows for input/output.
    Each item is a dict with keys:
        'input': FloatTensor[input_len, input_dim]
        'target': FloatTensor[output_len, output_dim]
    """
    def __init__(
        self,
        data: np.ndarray,
        input_len: int = 60,
        output_len: int = 5,
        target_col: int = 3,  # default to 'close'
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
            'input': torch.tensor(x, dtype=torch.float32),
            'target': torch.tensor(y, dtype=torch.float32),
        }

    @staticmethod
    def get_splits(
        csv_path: str,
        input_len: int = 60,
        output_len: int = 5,
        split_frac: Dict[str, float] | None = None,
        target_col: Optional[str] = 'close',
        ticker: Optional[str] = None,
    ) -> Tuple['StockMarketDataset', 'StockMarketDataset', 'StockMarketDataset']:
        """
        Loads CSV, filters by ticker if provided, uses only numeric columns, splits chronologically, fits scaler on train, applies to all, returns datasets.
        """
        csv_path = resolve_data_path(csv_path)
        df = pd.read_csv(csv_path)
        # Optionally filter by ticker symbol
        if ticker is not None and 'Name' in df.columns:
            df = df[df['Name'] == ticker]
        # Use only numeric columns (open, high, low, close, volume) - case insensitive
        available_cols = [c.lower() for c in df.columns]
        target_col_lower = target_col.lower() if isinstance(target_col, str) else target_col
        
        # Find matching columns case-insensitively
        cols = []
        for col_name in ['open', 'high', 'low', 'close', 'volume']:
            if col_name in available_cols:
                # Find the original column name with correct case
                original_col = [c for c in df.columns if c.lower() == col_name][0]
                cols.append(original_col)
        
        df = df[cols]
        
        # Determine target column index
        if target_col is not None:
            if isinstance(target_col, str):
                # Find target column case-insensitively
                target_col_original = [c for c in df.columns if c.lower() == target_col_lower]
                if not target_col_original:
                    raise ValueError(f"Target column '{target_col}' not found in available columns: {list(df.columns)}")
                target_idx = df.columns.get_loc(target_col_original[0])
            else:
                target_idx = int(target_col)
        else:
            # Default to 'close' column
            close_cols = [c for c in df.columns if c.lower() == 'close']
            if not close_cols:
                raise ValueError("No 'close' column found in available columns")
            target_idx = df.columns.get_loc(close_cols[0])
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
        train_ds = StockMarketDataset(train_data, input_len, output_len, target_col=target_idx)
        val_ds = StockMarketDataset(val_data, input_len, output_len, target_col=target_idx)
        test_ds = StockMarketDataset(test_data, input_len, output_len, target_col=target_idx)
        return train_ds, val_ds, test_ds

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test StockMarketDataset loader.")
    parser.add_argument('--csv', type=str, required=True, help='Path to stock CSV.')
    parser.add_argument('--input_len', type=int, default=60)
    parser.add_argument('--output_len', type=int, default=5)
    parser.add_argument('--ticker', type=str, default=None)
    args = parser.parse_args()
    train, val, test = StockMarketDataset.get_splits(args.csv, args.input_len, args.output_len, ticker=args.ticker)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    sample = train[0]
    print(f"input shape: {sample['input'].shape}, target shape: {sample['target'].shape}") 