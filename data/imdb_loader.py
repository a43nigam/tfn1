import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, Tuple
from . import resolve_data_path

class IMDBDataset(Dataset):
    """
    IMDBDataset for text classification (genre) and regression (IMDB_Rating) on IMDB Top 1000 Movies/TV Shows.
    Args:
        file_path: Path to CSV
        tokenizer_name: HuggingFace tokenizer
        max_length: Max token length
        task: 'classification' or 'regression'
        text_col: Column for input text (Overview)
        label_col: Column for genre (classification)
        regression_col: Column for IMDB_Rating (regression)
    """
    def __init__(
        self,
        file_path: str,
        tokenizer_name: str = 'bert-base-uncased',
        max_length: int = 256,
        task: str = 'classification',
        text_col: str = 'Overview',
        label_col: str = 'Genre',
        regression_col: str = 'IMDB_Rating',
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Handle both file_path and direct DataFrame input
        if df is not None:
            # Use provided DataFrame (for splits)
            self.df = df
        else:
            # Load from file path (for backward compatibility)
            file_path = resolve_data_path(file_path)
            self.df = pd.read_csv(file_path)
        
        self.texts = self.df[text_col].astype(str).tolist()
        self.task = task
        if task == 'classification':
            # Use first genre if multiple
            self.labels = [str(g).split(',')[0].strip() for g in self.df[label_col]]
            self.label2idx = {g: i for i, g in enumerate(sorted(set(self.labels)))}
            self.idx2label = {i: g for g, i in self.label2idx.items()}
            self.targets = [self.label2idx[g] for g in self.labels]
        elif task == 'regression':
            self.targets = self.df[regression_col].astype(float).tolist()
        else:
            raise ValueError(f"Unknown task: {task}")
        self.max_length = max_length
    def __len__(self) -> int:
        return len(self.texts)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        if self.task == 'classification':
            label = torch.tensor(self.targets[idx], dtype=torch.long)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}
        else:
            value = torch.tensor(self.targets[idx], dtype=torch.float32)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': value}
    
    @staticmethod
    def get_splits(
        file_path: str,
        tokenizer_name: str = 'bert-base-uncased',
        max_length: int = 256,
        split_frac: Optional[Dict[str, float]] = None,
        text_col: str = 'review',
        label_col: str = 'sentiment',
        regression_col: str = 'sentiment',
        task: str = 'classification',
        split_seed: int = 42,
    ) -> Tuple['IMDBDataset', 'IMDBDataset', 'IMDBDataset']:
        """
        Loads CSV, splits randomly, and returns train/val/test datasets.
        
        Args:
            file_path: Path to IMDB CSV file
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            split_frac: Split fractions (default: 80/10/10)
            text_col: Column name for text data
            label_col: Column name for classification labels
            regression_col: Column name for regression targets
            task: 'classification' or 'regression'
            split_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        from .split_utils import get_split_sizes, DEFAULT_SPLIT_FRAC
        
        # Use default split fractions if not provided
        if split_frac is None:
            split_frac = DEFAULT_SPLIT_FRAC
        
        # Load and prepare data
        file_path = resolve_data_path(file_path)
        df = pd.read_csv(file_path)
        
        # Validate required columns exist
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in CSV. Available columns: {list(df.columns)}")
        
        if task == 'classification' and label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV. Available columns: {list(df.columns)}")
        elif task == 'regression' and regression_col not in df.columns:
            raise ValueError(f"Regression column '{regression_col}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Shuffle data for random split
        df = df.sample(frac=1, random_state=split_seed).reset_index(drop=True)
        
        # Calculate split sizes
        n_total = len(df)
        n_train, n_val, n_test = get_split_sizes(n_total, split_frac)
        
        # Split data
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]
        
        # Create datasets
        train_ds = IMDBDataset(
            file_path=None,  # We'll pass the DataFrame directly
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            task=task,
            text_col=text_col,
            label_col=label_col,
            regression_col=regression_col,
            df=train_df  # Pass the split DataFrame
        )
        
        val_ds = IMDBDataset(
            file_path=None,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            task=task,
            text_col=text_col,
            label_col=label_col,
            regression_col=regression_col,
            df=val_df
        )
        
        test_ds = IMDBDataset(
            file_path=None,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            task=task,
            text_col=text_col,
            label_col=label_col,
            regression_col=regression_col,
            df=test_df
        )
        
        return train_ds, val_ds, test_ds 