import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List

# Consolidated split helper
from .split_utils import get_split_sizes, DEFAULT_SPLIT_FRAC
# Dynamic data path resolution (e.g., Kaggle input datasets)
from . import resolve_data_path


class NormalizationStrategy(ABC):
    """Abstract base class for normalization strategies."""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the normalizer on training data."""
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        pass
    
    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform for denormalization."""
        pass


class GlobalStandardScaler(NormalizationStrategy):
    """Global standardization (current behavior) - fit on train, apply to all splits."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def fit(self, data: np.ndarray) -> None:
        """Fit scaler on training data."""
        self.scaler.fit(data)
        self._is_fitted = True
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform for denormalization."""
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return self.scaler.inverse_transform(data)


class InstanceNormalizer(NormalizationStrategy):
    """Instance normalization - normalize each sample individually."""
    
    def __init__(self):
        self._is_fitted = True  # No fitting needed for instance normalization
    
    def fit(self, data: np.ndarray) -> None:
        """No fitting needed for instance normalization."""
        pass
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize each sample individually (mean=0, std=1).
        
        Args:
            data: Array of shape [num_samples, num_features] or [num_samples, seq_len, num_features]
        
        Returns:
            Normalized data with same shape as input
        """
        if data.ndim == 2:
            # Handle [num_samples, num_features] - normalize each sample
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
            return (data - mean) / std
        elif data.ndim == 3:
            # Handle [num_samples, seq_len, num_features] - normalize each sample across time
            mean = np.mean(data, axis=(1, 2), keepdims=True)
            std = np.std(data, axis=(1, 2), keepdims=True)
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
            return (data - mean) / std
        else:
            raise ValueError(f"InstanceNormalizer expects 2D or 3D data, got {data.ndim}D")
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Instance normalization is not easily invertible - return as-is."""
        return data


class FeatureWiseNormalizer(NormalizationStrategy):
    """Feature-wise normalization - normalize each feature independently."""
    
    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
        self._is_fitted = False
    
    def fit(self, data: np.ndarray) -> None:
        """Fit normalizer on training data."""
        self.feature_means = np.mean(data, axis=0)
        self.feature_stds = np.std(data, axis=0)
        # Avoid division by zero
        self.feature_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
        self._is_fitted = True
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self._is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        return (data - self.feature_means) / self.feature_stds
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform for denormalization."""
        if not self._is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        return data * self.feature_stds + self.feature_means


def create_normalization_strategy(strategy_name: str) -> NormalizationStrategy:
    """Factory function to create normalization strategies."""
    if strategy_name == "global":
        return GlobalStandardScaler()
    elif strategy_name == "instance":
        return InstanceNormalizer()
    elif strategy_name == "feature_wise":
        return FeatureWiseNormalizer()
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy_name}")


class ETTDataset(Dataset):
    """
    ETTDataset for univariate/multivariate time series forecasting.
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
        target_col: int = -1,
        instance_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.data = data  # shape: [num_samples, num_features]
        self.input_len = input_len
        self.output_len = output_len
        self.target_col = target_col
        self.instance_normalize = instance_normalize
        self.indices = self._compute_indices()

    def _compute_indices(self) -> List[int]:
        # Only indices where both input and output windows fit
        total_len = self.input_len + self.output_len
        return [i for i in range(len(self.data) - total_len + 1)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        x = self.data[i : i + self.input_len]  # [input_len, input_dim]
        y = self.data[i + self.input_len : i + self.input_len + self.output_len, self.target_col]  # [output_len]
        
        # Instance normalization is now applied in get_splits for efficiency
        # No need to apply it here anymore
        
        # If output is 1D, add feature dim
        if y.ndim == 1:
            y = y[:, None]
        return {
            'input': torch.tensor(x, dtype=torch.float32),
            'target': torch.tensor(y, dtype=torch.float32),
        }

    @staticmethod
    def get_splits(
        csv_path: str,
        input_len: int = 96,
        output_len: int = 24,
        split_frac: Dict[str, float] | None = None,
        target_col: Optional[str] = 'OT',
        normalization_strategy: str = "global",
        instance_normalize: bool = False,
    ) -> Tuple['ETTDataset', 'ETTDataset', 'ETTDataset']:
        """
        Loads CSV, splits chronologically, fits scaler on train, applies to all, returns datasets.
        
        Args:
            csv_path: Path to CSV file
            input_len: Input sequence length
            output_len: Output sequence length
            split_frac: Split fractions for train/val/test
            target_col: Target column name or index
            normalization_strategy: Normalization strategy ("global", "instance", "feature_wise")
            instance_normalize: Whether to apply instance normalization in __getitem__
        """
        csv_path = resolve_data_path(csv_path)
        df = pd.read_csv(csv_path)
        
        # Extract calendar features if available
        calendar_features = None
        if 'date' in df.columns:
            # Store calendar features for potential use in embeddings
            date_series = pd.to_datetime(df['date'])
            calendar_features = {
                'hour': date_series.dt.hour.values,
                'day_of_week': date_series.dt.dayofweek.values,
                'day_of_month': date_series.dt.day.values,
                'month': date_series.dt.month.values,
                'is_weekend': (date_series.dt.dayofweek >= 5).astype(int).values,
            }
            # Drop date column for feature processing
            df = df.drop(columns=['date'])
        
        # Determine target column index
        if target_col is not None:
            if isinstance(target_col, str):
                target_idx = df.columns.get_loc(target_col)
            else:
                target_idx = int(target_col)
        else:
            target_idx = -1  # last column
        
        data = df.values.astype(np.float32)
        n_train, n_val, n_test = get_split_sizes(len(data), split_frac)
        
        # Chronological split
        train_data = data[:n_train]
        val_data = data[n_train:n_train+n_val]
        test_data = data[n_train+n_val:]
        
        # Apply normalization strategy
        normalizer = create_normalization_strategy(normalization_strategy)
        
        if normalization_strategy == "global":
            # Fit on train, apply to all splits
            normalizer.fit(train_data)
            train_data = normalizer.transform(train_data)
            val_data = normalizer.transform(val_data)
            test_data = normalizer.transform(test_data)
        elif normalization_strategy == "feature_wise":
            # Fit on train, apply to all splits
            normalizer.fit(train_data)
            train_data = normalizer.transform(train_data)
            val_data = normalizer.transform(val_data)
            test_data = normalizer.transform(test_data)
        elif normalization_strategy == "instance":
            # Step 1: Vectorize instance normalization - apply to entire dataset splits
            train_data = normalizer.transform(train_data)
            val_data = normalizer.transform(val_data)
            test_data = normalizer.transform(test_data)
        
        # Apply additional instance normalization if requested (for backward compatibility)
        if instance_normalize and normalization_strategy != "instance":
            instance_normalizer = InstanceNormalizer()
            train_data = instance_normalizer.transform(train_data)
            val_data = instance_normalizer.transform(val_data)
            test_data = instance_normalizer.transform(test_data)
        
        # Build datasets
        train_ds = ETTDataset(train_data, input_len, output_len, target_col=target_idx, 
                             instance_normalize=instance_normalize)
        val_ds = ETTDataset(val_data, input_len, output_len, target_col=target_idx,
                           instance_normalize=instance_normalize)
        test_ds = ETTDataset(test_data, input_len, output_len, target_col=target_idx,
                            instance_normalize=instance_normalize)
        
        # Attach the normalizer and calendar features to all datasets
        for ds in [train_ds, val_ds, test_ds]:
            ds.normalizer = normalizer
            ds.calendar_features = calendar_features
        
        return train_ds, val_ds, test_ds

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test ETTDataset loader.")
    parser.add_argument('--csv', type=str, required=True, help='Path to ETT CSV.')
    parser.add_argument('--input_len', type=int, default=96)
    parser.add_argument('--output_len', type=int, default=24)
    args = parser.parse_args()
    train, val, test = ETTDataset.get_splits(args.csv, args.input_len, args.output_len)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    sample = train[0]
    print(f"input shape: {sample['input'].shape}, target shape: {sample['target'].shape}") 