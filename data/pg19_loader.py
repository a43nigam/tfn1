import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, Tuple

# Dynamic data path resolution (e.g., Kaggle input datasets)
from . import resolve_data_path

class PG19Dataset(Dataset):
    """
    PG19Dataset for long-form language modeling (PG-19 books).
    Loads CSV, tokenizes, and returns dicts for DataLoader.
    """
    def __init__(
        self,
        file_path: str,
        tokenizer_name: str = 'gpt2',
        max_length: int = 512,
        split: Optional[str] = None,
        split_frac: Optional[Dict[str, float]] = None,
        split_seed: int = 42,
        text_col: str = 'text',
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        file_path = resolve_data_path(file_path)
        df = pd.read_csv(file_path)
        if text_col not in df.columns:
            raise ValueError(f"CSV must have '{text_col}' column.")
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
        self.max_length = max_length
    def __len__(self) -> int:
        return len(self.texts)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        input_ids = enc['input_ids'].squeeze(0)
        # For LM: labels are input_ids shifted by 1 (next token prediction)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # ignore index for last token
        return {
            'inputs': input_ids,
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': labels,
        }
    @staticmethod
    def get_splits(file_path: str, tokenizer_name: str = 'gpt2', max_length: int = 512, split_frac: Optional[Dict[str, float]] = None, text_col: str = 'text'):
        return (
            PG19Dataset(file_path, tokenizer_name, max_length, split='train', split_frac=split_frac, text_col=text_col),
            PG19Dataset(file_path, tokenizer_name, max_length, split='val', split_frac=split_frac, text_col=text_col),
            PG19Dataset(file_path, tokenizer_name, max_length, split='test', split_frac=split_frac, text_col=text_col),
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test PG19Dataset loader.")
    parser.add_argument('--file', type=str, required=True, help='Path to PG-19 CSV.')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()
    ds = PG19Dataset(args.file, args.tokenizer, args.max_length)
    sample = ds[0]
    print(f"input_ids shape: {sample['input_ids'].shape}, attention_mask shape: {sample['attention_mask'].shape}, labels shape: {sample['labels'].shape}") 