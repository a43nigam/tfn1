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

def _find_column_index(df: pd.DataFrame, col_name: str) -> int:
    """
    Find column index case-insensitively.
    
    Args:
        df: DataFrame to search in
        col_name: Column name to find (case-insensitive)
    
    Returns:
        Column index
    
    Raises:
        ValueError: If column not found
    """
    col_name_lower = col_name.lower()
    available_cols = [c.lower() for c in df.columns]
    
    if col_name_lower not in available_cols:
        raise ValueError(f"Column '{col_name}' not found in available columns: {list(df.columns)}")
    
    # Find the original column name with correct case
    original_col = [c for c in df.columns if c.lower() == col_name_lower][0]
    return df.columns.get_loc(original_col)

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
            'inputs': torch.tensor(x, dtype=torch.float32),
            'targets': torch.tensor(y, dtype=torch.float32),
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
        
        # Step 2: Simplify column logic using utility function
        # Use only numeric columns (open, high, low, close, volume) - case insensitive
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [c.lower() for c in df.columns]
        
        # Find matching columns case-insensitively
        cols = []
        for col_name in numeric_cols:
            if col_name in available_cols:
                # Find the original column name with correct case
                original_col = [c for c in df.columns if c.lower() == col_name][0]
                cols.append(original_col)
        
        df = df[cols]
        
        # Determine target column index using utility function
        if target_col is not None:
            if isinstance(target_col, str):
                target_idx = _find_column_index(df, target_col)
            else:
                target_idx = int(target_col)
        else:
            # Default to 'close' column
            target_idx = _find_column_index(df, 'close')
        
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
        
        # Attach the scaler to all datasets for denormalization during evaluation
        train_ds.scaler = scaler
        val_ds.scaler = scaler
        test_ds.scaler = scaler
        
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