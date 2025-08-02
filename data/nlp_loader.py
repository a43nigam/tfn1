import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, Any

# Dynamic data path resolution
from . import resolve_data_path

class NLPDataset(Dataset):
    """
    NLPDataset for text classification. Loads a CSV/TSV with 'text' and 'label' columns,
    tokenizes with HuggingFace transformers, and returns dicts for DataLoader.
    """
    def __init__(
        self,
        file_path: str,
        tokenizer_name: str = 'bert-base-uncased',
        max_length: int = 128,
        split: Optional[str] = None,
        split_frac: Optional[Dict[str, float]] = None,
        split_seed: int = 42,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        file_path = resolve_data_path(file_path)
        df = pd.read_csv(file_path, sep=None, engine='python')
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV/TSV must have 'text' and 'label' columns.")
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
        self.texts = df['text'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
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
            'inputs': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }
    @staticmethod
    def get_splits(file_path: str, tokenizer_name: str = 'bert-base-uncased', max_length: int = 128, split_frac: Optional[Dict[str, float]] = None):
        return (
            NLPDataset(file_path, tokenizer_name, max_length, split='train', split_frac=split_frac),
            NLPDataset(file_path, tokenizer_name, max_length, split='val', split_frac=split_frac),
            NLPDataset(file_path, tokenizer_name, max_length, split='test', split_frac=split_frac),
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test NLPDataset loader.")
    parser.add_argument('--file', type=str, required=True, help='Path to CSV/TSV with text and label columns.')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()
    ds = NLPDataset(args.file, args.tokenizer, args.max_length)
    sample = ds[0]
    print(f"input_ids shape: {sample['input_ids'].shape}, attention_mask shape: {sample['attention_mask'].shape}, label: {sample['labels']}") 