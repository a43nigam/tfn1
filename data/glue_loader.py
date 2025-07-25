import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, Tuple

# Support dynamic path resolution (e.g., inside Kaggle)
from . import resolve_data_path

class GLUEDataset(Dataset):
    """
    GLUEDataset for GLUE tasks (e.g., SST-2, MRPC).
    Loads TSV/CSV, tokenizes, and returns dicts for DataLoader.
    """
    def __init__(
        self,
        file_path: str,
        tokenizer_name: str = 'bert-base-uncased',
        max_length: int = 128,
        split: Optional[str] = None,
        split_frac: Optional[Dict[str, float]] = None,
        split_seed: int = 42,
        text_col: str = 'sentence',
        label_col: str = 'label',
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        file_path = resolve_data_path(file_path)
        df = pd.read_csv(file_path, sep=None, engine='python')
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"CSV/TSV must have '{text_col}' and '{label_col}' columns.")
        # Optionally split
        if split_frac is not None:
            df = df.sample(frac=1, random_state=split_seed).reset_index(drop=True)
            n = len(df)
            train_end = int(split_frac.get('train', 0.8) * n)
            val_end = train_end + int(split_frac.get('val', 0.1) * n)
            if split == 'train':
                df = df.iloc[:train_end]
            elif split == 'val':
                df = df.iloc[train_end:val_end]
            elif split == 'test':
                df = df.iloc[val_end:]
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.max_length = max_length
    def __len__(self) -> int:
        return len(self.texts)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }
    @staticmethod
    def get_splits(file_path: str, tokenizer_name: str = 'bert-base-uncased', max_length: int = 128, split_frac: Optional[Dict[str, float]] = None, text_col: str = 'sentence', label_col: str = 'label'):
        return (
            GLUEDataset(file_path, tokenizer_name, max_length, split='train', split_frac=split_frac, text_col=text_col, label_col=label_col),
            GLUEDataset(file_path, tokenizer_name, max_length, split='val', split_frac=split_frac, text_col=text_col, label_col=label_col),
            GLUEDataset(file_path, tokenizer_name, max_length, split='test', split_frac=split_frac, text_col=text_col, label_col=label_col),
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test GLUEDataset loader.")
    parser.add_argument('--file', type=str, required=True, help='Path to GLUE TSV/CSV.')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()
    ds = GLUEDataset(args.file, args.tokenizer, args.max_length)
    sample = ds[0]
    print(f"input_ids shape: {sample['input_ids'].shape}, attention_mask shape: {sample['attention_mask'].shape}, label: {sample['labels']}") 