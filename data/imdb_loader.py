import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, Any
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
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        file_path = resolve_data_path(file_path)
        df = pd.read_csv(file_path)
        self.texts = df[text_col].astype(str).tolist()
        self.task = task
        if task == 'classification':
            # Use first genre if multiple
            self.labels = [str(g).split(',')[0].strip() for g in df[label_col]]
            self.label2idx = {g: i for i, g in enumerate(sorted(set(self.labels)))}
            self.idx2label = {i: g for g, i in self.label2idx.items()}
            self.targets = [self.label2idx[g] for g in self.labels]
        elif task == 'regression':
            self.targets = df[regression_col].astype(float).tolist()
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